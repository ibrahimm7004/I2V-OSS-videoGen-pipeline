from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFRepoSpec:
    repo_id: str
    revision: str | None = None


MODEL_REPO_REGISTRY: dict[str, list[HFRepoSpec]] = {
    "wan22_ti2v_5b": [
        HFRepoSpec(repo_id="Wan-AI/Wan2.2-TI2V-5B"),
        # Diffusers-format weights used by the runtime adapter.
        HFRepoSpec(repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    ],
    "hunyuan_i2v": [
        HFRepoSpec(repo_id="tencent/HunyuanVideo-I2V"),
    ],
    "cogvideox15_5b_i2v": [
        HFRepoSpec(repo_id="zai-org/CogVideoX1.5-5B-I2V"),
    ],
}

MODEL_SELECTORS: dict[str, list[str]] = {
    "all": ["wan22_ti2v_5b", "hunyuan_i2v", "cogvideox15_5b_i2v"],
    "wan": ["wan22_ti2v_5b"],
    "hunyuan": ["hunyuan_i2v"],
    "cog": ["cogvideox15_5b_i2v"],
}


def model_ids_for_selector(selector: str) -> list[str]:
    key = selector.lower()
    if key not in MODEL_SELECTORS:
        valid = ", ".join(sorted(MODEL_SELECTORS.keys()))
        raise ValueError(f"Unknown selector '{selector}'. Valid values: {valid}")
    return list(MODEL_SELECTORS[key])


def repos_for_model_id(model_id: str) -> list[HFRepoSpec]:
    key = model_id.lower()
    if key not in MODEL_REPO_REGISTRY:
        valid = ", ".join(sorted(MODEL_REPO_REGISTRY.keys()))
        raise ValueError(f"Unknown model_id '{model_id}'. Known model IDs: {valid}")
    return list(MODEL_REPO_REGISTRY[key])
