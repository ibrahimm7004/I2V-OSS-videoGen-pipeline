from __future__ import annotations

import importlib
from typing import Any

from .base import ModelAdapter

_ADAPTERS: dict[str, tuple[str, str]] = {
    "mock": ("models.mock", "MockAdapter"),
    "wan2.2": ("models.wan22", "Wan22Adapter"),
    "wan22": ("models.wan22", "Wan22Adapter"),
    "wan2.2-i2v": ("models.wan22", "Wan22Adapter"),
    "wan22_ti2v_5b": ("models.wan22", "Wan22Adapter"),
    "hunyuan-i2v": ("models.hunyuan_i2v", "HunyuanI2VAdapter"),
    "hunyuan_i2v": ("models.hunyuan_i2v", "HunyuanI2VAdapter"),
    "cogvideox1.5-i2v": ("models.cogvideox15_i2v", "CogVideoX15I2VAdapter"),
    "cogvideox15-i2v": ("models.cogvideox15_i2v", "CogVideoX15I2VAdapter"),
    "cogvideox15_5b_i2v": ("models.cogvideox15_i2v", "CogVideoX15I2VAdapter"),
}


def get_adapter(model_id: str, model_version: str, config: Any) -> ModelAdapter:
    adapter_ref = _ADAPTERS.get(model_id.lower())
    if adapter_ref is None:
        available = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(f"Unknown model adapter '{model_id}'. Known ids: {available}")
    module_name, class_name = adapter_ref
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(model_id=model_id, model_version=model_version, config=config)
