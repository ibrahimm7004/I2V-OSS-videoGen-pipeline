from __future__ import annotations

import logging
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .base import ClipRequest, ClipResult, ModelAdapter
from .registry import repos_for_model_id

LOGGER = logging.getLogger(__name__)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _multiple_of_16(value: int) -> int:
    if value <= 16:
        return 16
    return max(16, value - (value % 16))


def _resolve_guidance_scale(request: ClipRequest) -> float:
    cfg = request.params.get("cfg")
    if cfg is None:
        cfg = request.params.get("guidance_scale")
    if cfg is None:
        cfg = request.global_params.get("cfg", request.global_params.get("guidance_scale", 6.0))
    return float(cfg)


def _resolve_num_frames(request: ClipRequest) -> int:
    duration_based = max(1, int(round(float(request.duration_seconds) * float(request.fps))) + 1)
    explicit = max(1, int(request.frames))
    return max(duration_based, explicit)


def _is_pil_image(value: Any) -> bool:
    return value.__class__.__module__.startswith("PIL.") and value.__class__.__name__ == "Image"


def _to_numpy(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    if _is_pil_image(value):
        return np.asarray(value)
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _coerce_to_fhwc(raw_frames: Any) -> Any:
    import numpy as np

    if isinstance(raw_frames, (list, tuple)):
        if not raw_frames:
            raise RuntimeError("CogVideoX returned an empty frame sequence.")
        first = raw_frames[0]
        if isinstance(first, (list, tuple)) and first:
            # Common shape: batch list -> frame list
            raw_frames = first
        stacked = np.stack([_to_numpy(frame) for frame in raw_frames], axis=0)
    else:
        stacked = _to_numpy(raw_frames)

    if stacked.ndim == 5:
        # B,F,C,H,W or B,F,H,W,C -> use first batch.
        stacked = stacked[0]

    if stacked.ndim != 4:
        raise RuntimeError(f"Unexpected frame tensor rank {stacked.ndim}; expected 4D after batching.")

    # Convert to F,H,W,C
    if stacked.shape[-1] in {1, 3, 4}:
        fhwc = stacked
    elif stacked.shape[1] in {1, 3, 4}:
        fhwc = np.transpose(stacked, (0, 2, 3, 1))
    else:
        raise RuntimeError(f"Cannot infer channel axis from frame shape {tuple(stacked.shape)}.")

    if fhwc.shape[-1] == 1:
        fhwc = np.repeat(fhwc, 3, axis=-1)
    elif fhwc.shape[-1] == 4:
        fhwc = fhwc[..., :3]
    if fhwc.shape[-1] != 3:
        raise RuntimeError(f"Expected 3-channel frames after conversion, got shape {tuple(fhwc.shape)}.")
    return fhwc


def _extract_frames_payload(output: Any) -> Any:
    if output is None:
        raise RuntimeError("CogVideoX pipeline output is None.")

    for attr in ("frames", "videos", "video", "images"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if value is not None:
                return value

    if isinstance(output, dict):
        for key in ("frames", "videos", "video", "images", "sample"):
            if key in output and output[key] is not None:
                return output[key]

    if isinstance(output, (tuple, list)) and output:
        return output[0]

    return output


def _sanitize_frames_to_uint8(raw_frames: Any) -> tuple[list[Any], dict[str, Any]]:
    import numpy as np
    from PIL import Image

    fhwc = _coerce_to_fhwc(raw_frames)
    original_dtype = str(fhwc.dtype)
    original_shape = tuple(int(x) for x in fhwc.shape)

    as_float = fhwc.astype(np.float32, copy=False)
    total_values = as_float.size
    nan_count = int(np.isnan(as_float).sum()) if total_values > 0 else 0
    inf_count = int(np.isinf(as_float).sum()) if total_values > 0 else 0
    nan_fraction = float((nan_count + inf_count) / max(1, total_values))

    sanitized = np.nan_to_num(as_float, nan=0.0, posinf=1.0, neginf=0.0)
    raw_min = float(sanitized.min()) if sanitized.size else 0.0
    raw_max = float(sanitized.max()) if sanitized.size else 0.0

    value_range_assumption = "uint8"
    if not np.issubdtype(fhwc.dtype, np.integer):
        if raw_min >= -1e-5 and raw_max <= 1.00001:
            value_range_assumption = "0..1"
        elif raw_min >= -1.00001 and raw_max <= 1.00001:
            value_range_assumption = "-1..1"
            sanitized = (sanitized + 1.0) * 0.5
        elif raw_min >= -1e-5 and raw_max <= 255.0001:
            value_range_assumption = "0..255"
            sanitized = sanitized / 255.0
        else:
            value_range_assumption = "minmax"
            if math.isclose(raw_max, raw_min, rel_tol=0.0, abs_tol=1e-8):
                sanitized = np.zeros_like(sanitized, dtype=np.float32)
            else:
                sanitized = (sanitized - raw_min) / (raw_max - raw_min)
    else:
        if raw_max > 255:
            value_range_assumption = "int-minmax"
            if math.isclose(raw_max, raw_min, rel_tol=0.0, abs_tol=1e-8):
                sanitized = np.zeros_like(sanitized, dtype=np.float32)
            else:
                sanitized = (sanitized - raw_min) / (raw_max - raw_min)
        else:
            value_range_assumption = "uint8"
            sanitized = sanitized / 255.0

    sanitized = np.clip(sanitized, 0.0, 1.0)
    if not np.isfinite(sanitized).all():
        raise RuntimeError("CogVideoX frame sanitize produced non-finite values after cleanup.")

    frame0 = sanitized[0] if sanitized.shape[0] > 0 else np.zeros((1, 1, 3), dtype=np.float32)
    frame0_min = float(frame0.min())
    frame0_max = float(frame0.max())
    frame_std = float(np.std(sanitized))

    uint8_frames = np.clip(np.round(sanitized * 255.0), 0, 255).astype(np.uint8)
    if uint8_frames.ndim != 4 or uint8_frames.shape[-1] != 3:
        raise RuntimeError(f"Unexpected uint8 frame shape after sanitize: {tuple(uint8_frames.shape)}")
    if uint8_frames.shape[0] <= 0:
        raise RuntimeError("No frames after CogVideoX sanitize.")

    pil_frames = [Image.fromarray(uint8_frames[i], mode="RGB") for i in range(uint8_frames.shape[0])]
    stats = {
        "raw_dtype": original_dtype,
        "raw_shape": list(original_shape),
        "value_range_assumption": value_range_assumption,
        "nan_fraction": nan_fraction,
        "first_frame_min": frame0_min,
        "first_frame_max": frame0_max,
        "frame_std": frame_std,
        "num_frames_sanitized": int(uint8_frames.shape[0]),
    }
    return pil_frames, stats


class CogVideoX15I2VAdapter(ModelAdapter):
    """
    CogVideoX 1.5 5B I2V adapter via diffusers CogVideoXImageToVideoPipeline.
    """

    def __init__(self, model_id: str, model_version: str, config: Any) -> None:
        super().__init__(model_id, model_version, config)
        self._pipe: Any | None = None
        self._pipe_repo_id: str | None = None
        self._torch: Any | None = None
        self._export_to_video: Any | None = None
        self._device: str | None = None
        self._dtype_name: str | None = None
        self._torch_dtype: Any | None = None

    def _import_runtime(self) -> tuple[Any, Any, Any]:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError("CogVideoX adapter requires torch. Install torch in this environment.") from exc

        try:
            from diffusers import CogVideoXImageToVideoPipeline  # type: ignore
            from diffusers.utils import export_to_video  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "CogVideoX adapter requires diffusers with CogVideoX I2V support. "
                "Install/upgrade diffusers, transformers, accelerate, and safetensors."
            ) from exc

        return torch, CogVideoXImageToVideoPipeline, export_to_video

    def _resolve_repo_candidates(self) -> list[str]:
        env_repo = os.getenv("COGVX15_REPO_ID")
        candidates: list[str] = []
        if env_repo:
            candidates.append(env_repo.strip())

        for key in [self.model_id, "cogvideox15_5b_i2v"]:
            try:
                for spec in repos_for_model_id(key):
                    if spec.repo_id not in candidates:
                        candidates.append(spec.repo_id)
                if candidates:
                    break
            except ValueError:
                continue

        if "THUDM/CogVideoX1.5-5B-I2V" not in candidates:
            candidates.append("THUDM/CogVideoX1.5-5B-I2V")
        if not candidates:
            candidates.append("zai-org/CogVideoX1.5-5B-I2V")
        return candidates

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
        dtype_env = (os.getenv("COGVX15_DTYPE") or "bf16").strip().lower()
        if dtype_env == "fp16":
            dtype = torch.float16
            dtype_name = "float16"
        elif dtype_env == "fp32":
            dtype = torch.float32
            dtype_name = "float32"
        elif dtype_env == "bf16":
            dtype = torch.bfloat16
            dtype_name = "bfloat16"
        else:
            raise ValueError("Invalid COGVX15_DTYPE. Use one of: fp16, bf16, fp32.")
        if device == "cpu" and dtype_name != "float32":
            dtype = torch.float32
            dtype_name = "float32"
        return dtype, device, dtype_name

    def _try_enable_sdpa(self, torch: Any) -> None:
        if not torch.cuda.is_available():
            return
        try:
            if hasattr(torch.backends, "cuda"):
                cuda_backend = torch.backends.cuda
                if hasattr(cuda_backend, "enable_flash_sdp"):
                    cuda_backend.enable_flash_sdp(True)
                if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
                    cuda_backend.enable_mem_efficient_sdp(True)
                if hasattr(cuda_backend, "enable_math_sdp"):
                    cuda_backend.enable_math_sdp(True)
        except Exception:
            pass

    def _apply_memory_opts(self, pipe: Any, device: str, torch: Any) -> None:
        enable_offload = _as_bool(os.getenv("COGVX15_ENABLE_CPU_OFFLOAD"), default=False)
        if device == "cuda":
            if enable_offload and hasattr(pipe, "enable_model_cpu_offload"):
                # Move modules onto CUDA first; offload manager hooks from there.
                try:
                    pipe.to("cuda")
                except Exception:
                    pass
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        else:
            pipe.to(device)

        self._try_enable_sdpa(torch)
        if hasattr(pipe, "enable_attention_slicing"):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(pipe, "enable_xformers_memory_efficient_attention") and _as_bool(
            os.getenv("COGVX15_ENABLE_XFORMERS"), default=True
        ):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        if hasattr(pipe, "enable_vae_tiling"):
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass
        if hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
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

        torch, CogVideoXImageToVideoPipeline, export_to_video = self._import_runtime()
        self._torch = torch
        self._export_to_video = export_to_video
        torch_dtype, device, dtype_name = self._resolve_dtype_and_device(torch)
        cache_dir = self._resolve_cache_dir()
        hf_token = os.getenv("HF_TOKEN")
        load_errors: list[str] = []

        for repo_id in self._resolve_repo_candidates():
            kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            if hf_token:
                kwargs["token"] = hf_token
            try:
                pipe = CogVideoXImageToVideoPipeline.from_pretrained(repo_id, **kwargs)
                self._apply_memory_opts(pipe, device, torch)
                self._pipe = pipe
                self._pipe_repo_id = repo_id
                self._device = device
                self._dtype_name = dtype_name
                self._torch_dtype = torch_dtype
                return pipe
            except Exception as exc:
                load_errors.append(f"{repo_id}: {exc}")

        raise RuntimeError(
            "Failed to load CogVideoX 1.5 I2V pipeline from available repos. "
            + " | ".join(load_errors)
        )

    def _apply_sampler_if_possible(self, pipe: Any, sampler_name: str | None) -> str | None:
        if not sampler_name:
            return None
        normalized = sampler_name.strip().lower().replace("-", "").replace("_", "")
        try:
            from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler  # type: ignore
        except Exception:
            LOGGER.warning("Sampler '%s' requested but scheduler classes are unavailable; using pipeline default.", sampler_name)
            return None

        scheduler_cls = None
        if normalized in {"ddim"}:
            scheduler_cls = DDIMScheduler
        elif normalized in {"euler", "eulera"}:
            scheduler_cls = EulerDiscreteScheduler
        elif normalized in {"dpmpp2m", "dpmpp", "dpm2m", "dpmsolver"}:
            scheduler_cls = DPMSolverMultistepScheduler
        else:
            LOGGER.warning("Sampler '%s' not mapped for CogVideoX; using pipeline default.", sampler_name)
            return None

        try:
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            return scheduler_cls.__name__
        except Exception:
            LOGGER.warning("Failed to apply sampler '%s'; using pipeline default.", sampler_name)
            return None

    def _timeout_callback(self, started: float, max_runtime_seconds: int):
        def _callback(_: Any, __: int, ___: Any, callback_kwargs: dict[str, Any] | None) -> dict[str, Any]:
            if time.monotonic() - started > max_runtime_seconds:
                raise TimeoutError(f"CogVideoX adapter exceeded clip timeout ({max_runtime_seconds}s)")
            return callback_kwargs or {}

        return _callback

    def _load_image(self, input_image: Path, width: int, height: int) -> Any:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError("CogVideoX adapter requires Pillow for conditioning image loading.") from exc

        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found for CogVideoX adapter: {input_image}")
        image = Image.open(input_image).convert("RGB")
        if image.size != (width, height):
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            image = image.resize((width, height), resample=resampling)
        return image

    def generate_clip(self, request: ClipRequest) -> ClipResult:
        if request.input_image is None:
            raise ValueError(
                "CogVideoX 1.5 I2V requires an input image for every clip. "
                "Check clip chaining/input_image wiring in the runner."
            )

        started = time.monotonic()
        pipe = self._ensure_pipeline()
        torch = self._torch
        export_to_video = self._export_to_video
        if torch is None or export_to_video is None:
            raise RuntimeError("CogVideoX adapter runtime was not initialized.")

        width = _multiple_of_16(int(request.width))
        height = _multiple_of_16(int(request.height))
        num_frames = _resolve_num_frames(request)
        guidance_scale = _resolve_guidance_scale(request)
        conditioning_image = self._load_image(Path(request.input_image), width, height)

        sampler_name = request.params.get("sampler") or request.global_params.get("sampler")
        scheduler_applied = self._apply_sampler_if_possible(pipe, sampler_name)

        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(request.seed))
        call_kwargs: dict[str, Any] = {
            "image": conditioning_image,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": max(1, int(request.steps)),
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "pt",
            "callback_on_step_end": self._timeout_callback(started, int(request.max_runtime_seconds)),
            "callback_on_step_end_tensor_inputs": [],
        }

        def _run_pipe() -> Any:
            return pipe(**call_kwargs)

        try:
            if self._device == "cuda" and self._torch_dtype is not None and self._dtype_name != "float32":
                with torch.autocast(device_type="cuda", dtype=self._torch_dtype):
                    output = _run_pipe()
            else:
                with nullcontext():
                    output = _run_pipe()
        except TypeError as exc:
            if "callback_on_step_end" not in str(exc):
                raise
            call_kwargs.pop("callback_on_step_end", None)
            call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
            if self._device == "cuda" and self._torch_dtype is not None and self._dtype_name != "float32":
                with torch.autocast(device_type="cuda", dtype=self._torch_dtype):
                    output = _run_pipe()
            else:
                output = _run_pipe()

        if time.monotonic() - started > request.max_runtime_seconds:
            raise TimeoutError(f"CogVideoX adapter exceeded clip timeout ({request.max_runtime_seconds}s)")

        raw_payload = _extract_frames_payload(output)
        pil_frames, frame_stats = _sanitize_frames_to_uint8(raw_payload)
        if frame_stats["num_frames_sanitized"] <= 0:
            raise RuntimeError("CogVideoX produced zero sanitized frames.")
        if frame_stats["nan_fraction"] > 0.25 and frame_stats["frame_std"] < 0.002:
            raise RuntimeError(
                "CogVideoX frames collapsed after NaN cleanup. "
                f"nan_fraction={frame_stats['nan_fraction']:.6f} frame_std={frame_stats['frame_std']:.6f}"
            )

        request.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(pil_frames, output_video_path=str(request.output_video_path), fps=max(1, int(request.fps)))

        mp4_size_bytes = request.output_video_path.stat().st_size if request.output_video_path.exists() else 0
        if mp4_size_bytes < 8_192:
            raise RuntimeError(
                f"CogVideoX output appears invalidly small ({mp4_size_bytes} bytes). "
                "Inspect adapter_metadata frame stats."
            )

        runtime_seconds = time.monotonic() - started
        return ClipResult(
            output_video_path=request.output_video_path,
            model_id=self.model_id,
            model_version=self.model_version,
            runtime_seconds=runtime_seconds,
            extra_metadata={
                "repo_id": self._pipe_repo_id,
                "dtype": self._dtype_name,
                "device": self._device,
                "scheduler": scheduler_applied,
                "guidance_scale": guidance_scale,
                "num_frames": num_frames,
                "fps": int(request.fps),
                "mp4_size_bytes": int(mp4_size_bytes),
                "first_frame_min": frame_stats["first_frame_min"],
                "first_frame_max": frame_stats["first_frame_max"],
                "nan_fraction": frame_stats["nan_fraction"],
                "frame_dtype": frame_stats["raw_dtype"],
                "frame_shape": frame_stats["raw_shape"],
                "value_range_assumption": frame_stats["value_range_assumption"],
                "frame_std": frame_stats["frame_std"],
            },
        )
