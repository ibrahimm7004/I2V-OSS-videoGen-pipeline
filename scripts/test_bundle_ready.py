from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.runner import run_job


def _fail(message: str) -> int:
    print(f"bundle_ready test failed: {message}", file=sys.stderr)
    return 1


def _resolve_job() -> Path:
    candidates = [Path("jobs/example_mock.yaml"), Path("jobs/idea03_cogvideox.yaml")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    fallback = sorted(Path("jobs").glob("*.yaml"))
    if fallback:
        return fallback[0]
    raise RuntimeError("No job YAML found under jobs/.")


def main() -> int:
    run_id = "bundle-ready-test"
    run_dir = (Path("outputs") / run_id).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)

    run_job(
        _resolve_job(),
        overrides={
            "run_id": run_id,
            "model.id": "mock",
            "output_root": "outputs",
        },
    )

    status_path = run_dir / "status" / "status.json"
    if not status_path.exists():
        return _fail(f"missing status file: {status_path}")

    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    stage = status_data.get("stage")
    if stage != "bundle_ready":
        return _fail(f"expected stage 'bundle_ready', got '{stage}'")

    bundle_path_value = status_data.get("bundle_path")
    if not isinstance(bundle_path_value, str) or not bundle_path_value:
        return _fail("missing bundle_path in status.json")

    bundle_path = Path(bundle_path_value)
    if not bundle_path.is_absolute():
        bundle_path = run_dir / bundle_path
    if not bundle_path.exists():
        return _fail(f"bundle path does not exist: {bundle_path}")

    print(f"bundle_ready test passed: {status_path}")
    print(f"bundle path: {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
