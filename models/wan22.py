from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .base import ClipRequest, ClipResult, ModelAdapter
from .registry import repos_for_model_id


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _multiple_of_16(value: int) -> int:
    if value <= 16:
        return 16
    return max(16, value - (value % 16))


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

    def _import_runtime(self) -> tuple[Any, Any, Any]:
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
        env_repo = os.getenv("WAN22_REPO_ID")
        candidates: list[str] = []
        if env_repo:
            candidates.append(env_repo.strip())

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
            candidates.append("Wan-AI/Wan2.2-TI2V-5B")
            candidates.append("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
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

        width = _multiple_of_16(int(request.width))
        height = _multiple_of_16(int(request.height))
        conditioning_image = self._load_image(Path(request.input_image), width, height)

        cfg = request.params.get("cfg")
        if cfg is None:
            cfg = request.params.get("guidance_scale")
        if cfg is None:
            cfg = request.global_params.get("cfg", request.global_params.get("guidance_scale", 5.0))
        guidance_scale = float(cfg)

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
            "num_frames": max(1, int(request.frames)),
            "num_inference_steps": max(1, int(request.steps)),
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "pil",
            "callback_on_step_end": self._timeout_callback(started, int(request.max_runtime_seconds)),
            "callback_on_step_end_tensor_inputs": [],
        }

        try:
            output = pipe(**call_kwargs)
        except TypeError as exc:
            # Older diffusers builds may not expose callback hooks for this pipeline.
            if "callback_on_step_end" not in str(exc):
                raise
            call_kwargs.pop("callback_on_step_end", None)
            call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
            output = pipe(**call_kwargs)

        if time.monotonic() - started > request.max_runtime_seconds:
            raise TimeoutError(f"WAN adapter exceeded clip timeout ({request.max_runtime_seconds}s)")

        frames = self._extract_video_frames(output)
        request.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, output_video_path=str(request.output_video_path), fps=max(1, int(request.fps)))

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
            },
        )
