from __future__ import annotations

from .base import ClipRequest, ClipResult, ModelAdapter


class HunyuanI2VAdapter(ModelAdapter):
    def generate_clip(self, request: ClipRequest) -> ClipResult:
        # TODO: Integrate Hunyuan I2V inference path.
        raise NotImplementedError("Hunyuan I2V adapter is a scaffold stub. Use model.id=mock for local dry runs.")

