from __future__ import annotations

from .base import ClipRequest, ClipResult, ModelAdapter


class CogVideoX15I2VAdapter(ModelAdapter):
    def generate_clip(self, request: ClipRequest) -> ClipResult:
        # TODO: Integrate CogVideoX 1.5 I2V inference path.
        raise NotImplementedError("CogVideoX 1.5 I2V adapter is a scaffold stub. Use model.id=mock for local dry runs.")

