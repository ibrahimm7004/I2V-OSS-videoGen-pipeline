from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _build_remote_command(remote_status_dir: str, action: str) -> str:
    safe_dir = remote_status_dir.rstrip("/")
    if action == "pause":
        return (
            f"mkdir -p '{safe_dir}' && "
            f"touch '{safe_dir}/PAUSE' && "
            f"rm -f '{safe_dir}/STOP' && "
            f"printf '{{\"action\":\"pause\"}}' > '{safe_dir}/control.json'"
        )
    if action == "resume":
        return (
            f"mkdir -p '{safe_dir}' && "
            f"rm -f '{safe_dir}/PAUSE' '{safe_dir}/STOP' && "
            f"printf '{{\"action\":\"resume\"}}' > '{safe_dir}/control.json'"
        )
    if action == "stop":
        return (
            f"mkdir -p '{safe_dir}' && "
            f"touch '{safe_dir}/STOP' && "
            f"printf '{{\"action\":\"stop\"}}' > '{safe_dir}/control.json'"
        )
    raise ValueError(f"Unsupported action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send pause/resume/stop control commands for a remote run.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-path", required=True, help="Remote outputs base path.")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Execute over SSH. Otherwise only print commands.")
    args = parser.parse_args()

    actions = [name for name, enabled in [("pause", args.pause), ("resume", args.resume), ("stop", args.stop)] if enabled]
    if len(actions) != 1:
        print("Choose exactly one action: --pause, --resume, or --stop", file=sys.stderr)
        return 1

    action = actions[0]
    remote_base = args.remote_path.rstrip("/").rstrip("\\")
    remote_status_dir = f"{remote_base}/{args.run_id}/status"
    remote_cmd = _build_remote_command(remote_status_dir, action)
    ssh_cmd = ["ssh", "-i", str(args.key), f"{args.user}@{args.host}", remote_cmd]

    print("Control command")
    print(f"  action: {action}")
    print(f"  remote status dir: {remote_status_dir}")
    print("  ssh:")
    print("  " + " ".join(ssh_cmd))

    if not args.execute:
        return 0

    result = subprocess.run(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        print(f"Remote control failed ({result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return 1
    if result.stdout.strip():
        print(result.stdout.strip())
    print("Remote control command applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
