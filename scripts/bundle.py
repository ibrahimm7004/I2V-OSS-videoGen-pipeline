from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.bundling import create_run_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one bundle archive for a run directory.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory (outputs/<run_id>).")
    parser.add_argument("--out", type=Path, default=None, help="Optional archive output path.")
    parser.add_argument("--format", choices=["zip", "tar.gz"], default="zip")
    args = parser.parse_args()

    archive_path = create_run_bundle(args.run_dir, archive_path=args.out, fmt=args.format)
    print(f"Bundle created: {archive_path}")


if __name__ == "__main__":
    main()
