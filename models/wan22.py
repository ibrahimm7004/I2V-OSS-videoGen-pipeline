from __future__ import annotations

from .base import ClipRequest, ClipResult, ModelAdapter


class Wan22Adapter(ModelAdapter):
    def generate_clip(self, request: ClipRequest) -> ClipResult:
        # TODO: Integrate WAN 2.2 I2V inference path.
        raise NotImplementedError("WAN 2.2 adapter is a scaffold stub. Use model.id=mock for local dry runs.")

