from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.watch_status import format_pretty_line


def main() -> int:
    payload = {
        "timestamp": "2026-02-18T00:00:00Z",
        "stage": "bundle_ready",
        "clip_index": None,
        "message": "bundle ready",
        "bundle_path": "outputs/bundle-ready-test/bundle.zip",
    }
    line = format_pretty_line(payload)
    if "bundle_path=\"outputs/bundle-ready-test/bundle.zip\"" not in line:
        print(f"watch pretty test failed: bundle_path missing in line: {line}", file=sys.stderr)
        return 1
    if "bundle_ready" not in line:
        print(f"watch pretty test failed: stage missing in line: {line}", file=sys.stderr)
        return 1
    print(f"watch pretty test passed: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

