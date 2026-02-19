from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import ClipRequest, ClipResult, ModelAdapter
from .registry import repos_for_model_id

LOGGER = logging.getLogger(__name__)
WAN_TI2V_GUIDANCE_URL = "https://github.com/Wan-Video/Wan2.2"
WAN_PROFILE_SMOKE = "smoke"
WAN_PROFILE_QUALITY = "quality"


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _multiple_of_16(value: int) -> int:
    if value <= 16:
        return 16
    return max(16, value - (value % 16))


def _choose_wan_frame_count(target_frames: int) -> int:
    # WAN pipeline frame counts are quantized to 4n+1.
    if target_frames <= 1:
        return 1
    return ((target_frames - 1 + 3) // 4) * 4 + 1


def _apply_native_720p_map(width: int, height: int) -> tuple[int, int, bool]:
    # Wan2.2 TI2V guidance uses 1280x704 (or 704x1280) for 720P mode.
    if width == 1280 and height == 720:
        return 1280, 704, True
    if width == 720 and height == 1280:
        return 704, 1280, True
    return width, height, False


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_wan_profile(request: ClipRequest) -> str | None:
    raw_profile = (
        request.params.get("wan_profile")
        or request.params.get("profile")
        or request.global_params.get("wan_profile")
        or request.global_params.get("profile")
        or os.getenv("WAN22_PROFILE")
    )
    if raw_profile is None:
        return None
    normalized = str(raw_profile).strip().lower()
    if normalized in {WAN_PROFILE_SMOKE, WAN_PROFILE_QUALITY}:
        return normalized
    return None


def _resolve_export_quality(request: ClipRequest, config_default: int) -> int:
    raw_value = request.params.get("export_quality")
    if raw_value is None:
        raw_value = request.global_params.get("export_quality")
    if raw_value is None:
        raw_value = config_default
    try:
        quality = int(raw_value)
    except (TypeError, ValueError):
        quality = int(config_default)
    return max(0, min(10, quality))


class Wan22Adapter(ModelAdapter):
    """
    WAN 2.2 TI2V adapter using Hugging Face Diffusers WanImageToVideoPipeline.
    """

    def __init__(self, model_id: str, model_version: str, config: Any) -> None:
        super().__init__(model_id, model_version, config)
        self._pipe: Any | None = None
        self._pipe_repo_id: str | None = None
        self._torch: Any | None = None
        self._export_to_video: Any | None = None
        self._device: str | None = None
        self._dtype_name: str | None = None
        self._imageio_ffmpeg_available: bool | None = None
        self._ffmpeg_version: str | None = None

    def _import_runtime(self) -> tuple[Any, Any, Any]:
        try:
            import ftfy  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "WAN adapter requires 'ftfy'. Install it with: pip install ftfy"
            ) from exc

        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError("WAN adapter requires torch. Install torch in this environment.") from exc

        try:
            from diffusers import WanImageToVideoPipeline  # type: ignore
            from diffusers.utils import export_to_video  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "WAN adapter requires diffusers with WAN support. "
                "Install/upgrade diffusers, transformers, accelerate, and safetensors."
            ) from exc

        return torch, WanImageToVideoPipeline, export_to_video

    def _resolve_repo_candidates(self) -> list[str]:
        configured_repo = str(getattr(self.config, "wan22_repo_id", "") or "").strip()
        env_repo = str(os.getenv("WAN22_REPO_ID") or "").strip()
        candidates: list[str] = []
        if configured_repo:
            candidates.append(configured_repo)

        if env_repo and env_repo != configured_repo:
            LOGGER.warning(
                "WAN22_REPO_ID env ('%s') differs from config wan22_repo_id ('%s'); using config value.",
                env_repo,
                configured_repo or "<empty>",
            )
            if env_repo not in candidates:
                candidates.append(env_repo)

        for key in [self.model_id, "wan22_ti2v_5b"]:
            try:
                for spec in repos_for_model_id(key):
                    if spec.repo_id not in candidates:
                        candidates.append(spec.repo_id)
                if candidates:
                    break
            except ValueError:
                continue

        if not candidates:
            candidates.append("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            candidates.append("Wan-AI/Wan2.2-TI2V-5B")
        return candidates

    def _ensure_export_preflight(self) -> tuple[bool, str]:
        try:
            import imageio_ffmpeg  # type: ignore  # noqa: F401

            imageio_ffmpeg_available = True
        except Exception as exc:
            raise RuntimeError(
                "WAN export preflight failed: python package 'imageio_ffmpeg' is missing. "
                "Install with 'pip install imageio-ffmpeg'."
            ) from exc

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "WAN export preflight failed: 'ffmpeg' is not available on PATH. "
                "Install ffmpeg (e.g. 'apt-get install ffmpeg') and ensure PATH is updated."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("WAN export preflight failed: 'ffmpeg -version' timed out.") from exc

        if result.returncode != 0:
            raise RuntimeError(
                "WAN export preflight failed: 'ffmpeg -version' returned non-zero "
                f"({result.returncode}): {result.stderr.strip()}"
            )
        first_line = (result.stdout or "").splitlines()
        ffmpeg_version = first_line[0].strip() if first_line else "unknown"
        self._imageio_ffmpeg_available = imageio_ffmpeg_available
        self._ffmpeg_version = ffmpeg_version
        return imageio_ffmpeg_available, ffmpeg_version

    def _resolve_cache_dir(self) -> str | None:
        configured_cache = getattr(self.config, "hf_hub_cache", None)
        if configured_cache:
            return str(Path(configured_cache).expanduser())
        env_hub_cache = os.getenv("HF_HUB_CACHE")
        if env_hub_cache:
            return str(Path(env_hub_cache).expanduser())
        configured_home = getattr(self.config, "hf_home", None)
        if configured_home:
            return str(Path(configured_home).expanduser() / "hub")
        env_home = os.getenv("HF_HOME")
        if env_home:
            return str(Path(env_home).expanduser() / "hub")
        return None

    def _resolve_dtype_and_device(self, torch: Any) -> tuple[Any, str, str]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_env = (os.getenv("WAN22_DTYPE") or "auto").strip().lower()
        if dtype_env == "bf16":
            dtype = torch.bfloat16
            dtype_name = "bfloat16"
        elif dtype_env == "fp32":
            dtype = torch.float32
            dtype_name = "float32"
        elif dtype_env == "fp16":
            dtype = torch.float16
            dtype_name = "float16"
        elif device == "cuda":
            # Default to fp16 for broad runtime compatibility on 4090.
            dtype = torch.float16
            dtype_name = "float16"
        else:
            dtype = torch.float32
            dtype_name = "float32"
        return dtype, device, dtype_name

    def _apply_memory_opts(self, pipe: Any, device: str) -> None:
        enable_offload = _as_bool(os.getenv("WAN22_ENABLE_CPU_OFFLOAD"), default=True)
        if device == "cuda":
            if enable_offload and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        else:
            pipe.to(device)

        if hasattr(pipe, "enable_attention_slicing"):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(pipe, "enable_xformers_memory_efficient_attention") and _as_bool(
            os.getenv("WAN22_ENABLE_XFORMERS"), default=True
        ):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass
        if hasattr(pipe, "set_progress_bar_config"):
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

    def _ensure_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        torch, WanImageToVideoPipeline, export_to_video = self._import_runtime()
        self._torch = torch
        self._export_to_video = export_to_video
        torch_dtype, device, dtype_name = self._resolve_dtype_and_device(torch)
        cache_dir = self._resolve_cache_dir()
        load_errors: list[str] = []

        for repo_id in self._resolve_repo_candidates():
            kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            try:
                pipe = WanImageToVideoPipeline.from_pretrained(repo_id, **kwargs)
                self._apply_memory_opts(pipe, device)
                self._pipe = pipe
                self._pipe_repo_id = repo_id
                self._device = device
                self._dtype_name = dtype_name
                return pipe
            except Exception as exc:
                load_errors.append(f"{repo_id}: {exc}")

        raise RuntimeError(
            "Failed to load WAN 2.2 pipeline from available repos. "
            + " | ".join(load_errors)
        )

    def _apply_sampler_if_possible(self, pipe: Any, sampler_name: str | None) -> str | None:
        if not sampler_name:
            return None
        normalized = sampler_name.strip().lower().replace("-", "").replace("_", "")
        try:
            from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler  # type: ignore
        except Exception:
            return None

        scheduler_cls = None
        if normalized in {"ddim"}:
            scheduler_cls = DDIMScheduler
        elif normalized in {"euler", "eulera"}:
            scheduler_cls = EulerDiscreteScheduler
        elif normalized in {"dpmpp2m", "dpmpp", "dpm2m", "dpmsolver"}:
            scheduler_cls = DPMSolverMultistepScheduler
        else:
            return None

        try:
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            return scheduler_cls.__name__
        except Exception:
            return None

    def _timeout_callback(self, started: float, max_runtime_seconds: int):
        def _callback(_: Any, __: int, ___: Any, callback_kwargs: dict[str, Any] | None) -> dict[str, Any]:
            if time.monotonic() - started > max_runtime_seconds:
                raise TimeoutError(f"WAN adapter exceeded clip timeout ({max_runtime_seconds}s)")
            return callback_kwargs or {}

        return _callback

    def _load_image(self, input_image: Path, width: int, height: int) -> Any:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError("WAN adapter requires Pillow for loading conditioning images.") from exc

        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found for WAN adapter: {input_image}")
        image = Image.open(input_image).convert("RGB")
        if image.size != (width, height):
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            image = image.resize((width, height), resample=resampling)
        return image

    def _extract_video_frames(self, output: Any) -> list[Any]:
        frames = None
        if hasattr(output, "frames"):
            frames = output.frames
        elif isinstance(output, tuple) and output:
            frames = output[0]
        if frames is None:
            raise RuntimeError("WAN pipeline returned no frames.")
        if isinstance(frames, list) and frames and isinstance(frames[0], list):
            frames = frames[0]
        if not isinstance(frames, list) or not frames:
            raise RuntimeError("WAN pipeline produced an empty frame list.")
        return frames

    def _probe_video(self, video_path: Path, timeout_seconds: int = 30) -> dict[str, Any]:
        ffprobe_bin = getattr(self.config, "ffprobe_bin", "ffprobe")
        command = [
            ffprobe_bin,
            "-hide_banner",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,avg_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=max(5, timeout_seconds),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffprobe not found: {ffprobe_bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ffprobe timed out while validating WAN clip: {video_path}") from exc

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed ({result.returncode}): {result.stderr.strip()}")
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("ffprobe returned invalid JSON for WAN clip validation.") from exc

    def _extract_frame_count(self, probe_payload: dict[str, Any]) -> int | None:
        streams = probe_payload.get("streams") or []
        if not streams:
            return None
        stream = streams[0] if isinstance(streams[0], dict) else {}
        for key in ("nb_read_frames", "nb_frames"):
            value = stream.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                text = value.strip()
                if text.isdigit():
                    return int(text)
        return None

    def _extract_duration_seconds(self, probe_payload: dict[str, Any]) -> float | None:
        fmt = probe_payload.get("format") or {}
        if not isinstance(fmt, dict):
            return None
        value = fmt.get("duration")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_encoded_video(
        self,
        *,
        video_path: Path,
        expected_frames: int,
        fps: int,
        timeout_seconds: int,
    ) -> tuple[int, float]:
        probe_payload = self._probe_video(video_path, timeout_seconds=timeout_seconds)
        actual_frames = self._extract_frame_count(probe_payload)
        if actual_frames is None:
            raise RuntimeError("WAN clip validation failed: could not determine encoded frame count via ffprobe.")

        if abs(actual_frames - expected_frames) > 1:
            raise RuntimeError(
                "WAN clip validation failed: frame count mismatch "
                f"(expected {expected_frames}, got {actual_frames})."
            )

        actual_duration = self._extract_duration_seconds(probe_payload)
        if actual_duration is None:
            raise RuntimeError("WAN clip validation failed: could not determine encoded duration via ffprobe.")

        expected_duration = float(expected_frames) / float(max(1, fps))
        duration_tolerance = max(0.15, 2.0 / float(max(1, fps)))
        if abs(actual_duration - expected_duration) > duration_tolerance:
            raise RuntimeError(
                "WAN clip validation failed: duration mismatch "
                f"(expected ~{expected_duration:.3f}s, got {actual_duration:.3f}s, "
                f"fps={fps}, tolerance={duration_tolerance:.3f}s)."
            )
        return actual_frames, actual_duration

    def generate_clip(self, request: ClipRequest) -> ClipResult:
        if request.input_image is None:
            raise ValueError(
                "WAN 2.2 TI2V requires an input image for every clip. "
                "Check clip chaining/input_image wiring in the runner."
            )

        started = time.monotonic()
        pipe = self._ensure_pipeline()
        torch = self._torch
        export_to_video = self._export_to_video
        if torch is None or export_to_video is None:
            raise RuntimeError("WAN adapter runtime was not initialized.")
        imageio_ffmpeg_available, ffmpeg_version = self._ensure_export_preflight()

        width = _multiple_of_16(int(request.width))
        height = _multiple_of_16(int(request.height))
        original_size = (width, height)
        width, height, native_720p_applied = _apply_native_720p_map(width, height)
        native_720p_warning: str | None = None
        if native_720p_applied:
            native_720p_warning = (
                f"WAN TI2V native_720p applied: requested {original_size[0]}x{original_size[1]} "
                f"mapped to {width}x{height} per Wan2.2 guidance."
            )
            LOGGER.warning(native_720p_warning)
        conditioning_image = self._load_image(Path(request.input_image), width, height)
        target_frames = max(1, int(round(float(request.fps) * float(request.duration_seconds))))
        wan_num_frames = _choose_wan_frame_count(target_frames)

        wan_profile = _resolve_wan_profile(request)
        requested_steps = max(1, int(request.steps))
        effective_steps = requested_steps
        if wan_profile == WAN_PROFILE_QUALITY:
            effective_steps = min(60, max(45, requested_steps))
        elif wan_profile == WAN_PROFILE_SMOKE:
            effective_steps = min(16, requested_steps)

        cfg = request.params.get("cfg")
        if cfg is None:
            cfg = request.params.get("guidance_scale")
        if cfg is None:
            cfg = request.global_params.get("cfg", request.global_params.get("guidance_scale"))
        if cfg is None:
            if wan_profile == WAN_PROFILE_QUALITY:
                cfg = 6.0
            elif wan_profile == WAN_PROFILE_SMOKE:
                cfg = 4.5
            else:
                cfg = 5.0
        guidance_scale = float(cfg)
        motion_strength = _as_float(request.params.get("motion_strength"))
        if motion_strength is None:
            motion_strength = _as_float(request.global_params.get("motion_strength"))
        export_quality = _resolve_export_quality(
            request,
            int(getattr(self.config, "wan22_export_quality", 9)),
        )

        sampler_name = request.params.get("sampler") or request.global_params.get("sampler")
        scheduler_applied = self._apply_sampler_if_possible(pipe, sampler_name)

        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(request.seed))
        warnings: list[str] = []
        if native_720p_warning:
            warnings.append(native_720p_warning)
        call_kwargs: dict[str, Any] = {
            "image": conditioning_image,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "height": height,
            "width": width,
            "num_frames": wan_num_frames,
            "num_inference_steps": effective_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "pil",
            "callback_on_step_end": self._timeout_callback(started, int(request.max_runtime_seconds)),
            "callback_on_step_end_tensor_inputs": [],
        }
        if motion_strength is not None:
            call_kwargs["motion_strength"] = motion_strength
        unsupported_kwargs: list[str] = []
        pipe_prompt = str(call_kwargs.get("prompt", request.prompt))

        while True:
            try:
                output = pipe(**call_kwargs)
                break
            except TypeError as exc:
                error_text = str(exc)
                handled = False
                # Older diffusers builds may not expose callback hooks for this pipeline.
                if "callback_on_step_end" in error_text and "callback_on_step_end" in call_kwargs:
                    call_kwargs.pop("callback_on_step_end", None)
                    call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
                    unsupported_kwargs.append("callback_on_step_end")
                    warnings.append("WAN pipeline does not support callback_on_step_end; continuing without step callback.")
                    handled = True
                if "motion_strength" in error_text and "motion_strength" in call_kwargs:
                    call_kwargs.pop("motion_strength", None)
                    unsupported_kwargs.append("motion_strength")
                    warnings.append("WAN pipeline does not support motion_strength; ignoring requested value.")
                    handled = True
                if not handled:
                    raise

        if time.monotonic() - started > request.max_runtime_seconds:
            raise TimeoutError(f"WAN adapter exceeded clip timeout ({request.max_runtime_seconds}s)")

        frames = self._extract_video_frames(output)
        backend_effective_num_frames: int | None
        try:
            backend_effective_num_frames = len(frames)
        except Exception:
            backend_effective_num_frames = None
            warnings.append("Could not determine backend effective num_frames from WAN output frames payload.")
        request.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            export_to_video(
                frames,
                output_video_path=str(request.output_video_path),
                fps=max(1, int(request.fps)),
                quality=export_quality,
            )
        except TypeError as exc:
            raise RuntimeError(
                "WAN export failed: current diffusers export_to_video does not accept 'quality'. "
                "Upgrade diffusers/imageio stack or remove unsupported runtime."
            ) from exc
        actual_frames, actual_duration = self._validate_encoded_video(
            video_path=request.output_video_path,
            expected_frames=wan_num_frames,
            fps=max(1, int(request.fps)),
            timeout_seconds=min(60, int(request.max_runtime_seconds)),
        )

        runtime_seconds = time.monotonic() - started
        return ClipResult(
            output_video_path=request.output_video_path,
            model_id=self.model_id,
            model_version=self.model_version,
            runtime_seconds=runtime_seconds,
            extra_metadata={
                "repo_id": self._pipe_repo_id,
                "repo_id_used": self._pipe_repo_id,
                "dtype": self._dtype_name,
                "device": self._device,
                "scheduler": scheduler_applied,
                "wan_profile": wan_profile,
                "steps_requested": requested_steps,
                "steps_effective": effective_steps,
                "guidance_scale": guidance_scale,
                "motion_strength_requested": motion_strength,
                "motion_strength_applied": "motion_strength" in call_kwargs,
                "unsupported_kwargs": unsupported_kwargs,
                "prompt_passed_to_pipe": pipe_prompt,
                "target_frames": target_frames,
                "wan_num_frames": wan_num_frames,
                "quantization_rule": "4n+1",
                "backend_effective_num_frames": backend_effective_num_frames,
                "requested_num_frames": wan_num_frames,
                "encoded_frames": actual_frames,
                "encoded_duration_sec": actual_duration,
                "fps": int(request.fps),
                "export_quality_used": export_quality,
                "ffmpeg_version": ffmpeg_version,
                "imageio_ffmpeg_available": imageio_ffmpeg_available,
                "requested_resolution": {"width": original_size[0], "height": original_size[1]},
                "internal_resolution": {"width": width, "height": height},
                "wan_native_720p_applied": native_720p_applied,
                "wan_native_720p_guidance_url": WAN_TI2V_GUIDANCE_URL,
                "warnings": warnings,
            },
        )
