from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .utils import utc_now_iso


class ProgressTracker:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.status_dir = run_dir / "status"
        self.status_path = self.status_dir / "status.json"
        self.log_path = self.status_dir / "progress.log"

    def update(
        self,
        clip_index: int | None,
        stage: str,
        percent: float,
        message: str,
        extra: dict | None = None,
    ) -> dict:
        event = {
            "timestamp": utc_now_iso(),
            "clip_index": clip_index,
            "stage": stage,
            "percent": round(max(0.0, min(100.0, percent)), 2),
            "message": message,
        }
        if extra:
            event.update(extra)
        self._write_status_atomic(event)
        self._append_progress_line(event)
        return event

    def _write_status_atomic(self, event: dict) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.status_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(event, indent=2, sort_keys=True), encoding="utf-8")
        last_error: Exception | None = None
        for attempt in range(10):
            try:
                os.replace(tmp_path, self.status_path)
                return
            except PermissionError as exc:
                last_error = exc
                # Windows can transiently lock files while scanners/indexers read them.
                time.sleep(0.05 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def _append_progress_line(self, event: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        clip_value = event["clip_index"] if event["clip_index"] is not None else "-"
        bundle_part = ""
        if event.get("bundle_path"):
            bundle_part = f" | bundle_path={event['bundle_path']}"
        line = f"{event['timestamp']} | {event['stage']} | clip={clip_value} | msg={event['message']}{bundle_part}\n"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
