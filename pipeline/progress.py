from __future__ import annotations

import json
import os
from pathlib import Path

from .utils import utc_now_iso


class ProgressTracker:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.status_dir = run_dir / "status"
        self.status_path = self.status_dir / "status.json"
        self.log_path = self.status_dir / "progress.log"

    def update(self, clip_index: int | None, stage: str, percent: float, message: str) -> dict:
        event = {
            "timestamp": utc_now_iso(),
            "clip_index": clip_index,
            "stage": stage,
            "percent": round(max(0.0, min(100.0, percent)), 2),
            "message": message,
        }
        self._write_status_atomic(event)
        self._append_progress_line(event)
        return event

    def _write_status_atomic(self, event: dict) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.status_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(event, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.status_path)

    def _append_progress_line(self, event: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        clip_value = event["clip_index"] if event["clip_index"] is not None else "-"
        line = f"{event['timestamp']} | {event['stage']} | clip={clip_value} | msg={event['message']}\n"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
