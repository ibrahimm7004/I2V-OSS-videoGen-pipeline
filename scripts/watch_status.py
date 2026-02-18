from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _ssh_run(user: str, host: str, key_path: Path, remote_cmd: str, timeout_seconds: int = 15) -> str:
    cmd = [
        "ssh",
        "-i",
        str(key_path),
        f"{user}@{host}",
        remote_cmd,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"SSH command failed ({result.returncode}): {stderr}")
    return result.stdout


def _remote_read_file(user: str, host: str, key_path: Path, remote_file: str) -> str | None:
    quoted = shlex.quote(remote_file)
    remote_cmd = f"if [ -f {quoted} ]; then cat {quoted}; else echo __MISSING__; fi"
    output = _ssh_run(user, host, key_path, remote_cmd)
    if output.strip() == "__MISSING__":
        return None
    return output


def _remote_tail_file(user: str, host: str, key_path: Path, remote_file: str, lines: int) -> str | None:
    quoted = shlex.quote(remote_file)
    remote_cmd = f"if [ -f {quoted} ]; then tail -n {lines} {quoted}; else echo __MISSING__; fi"
    output = _ssh_run(user, host, key_path, remote_cmd)
    if output.strip() == "__MISSING__":
        return None
    return output


def _extract_bundle_path(status_payload: dict) -> str | None:
    bundle_path = status_payload.get("bundle_path")
    if isinstance(bundle_path, str) and bundle_path:
        return bundle_path
    outputs = status_payload.get("outputs")
    if isinstance(outputs, dict):
        nested = outputs.get("bundle_path")
        if isinstance(nested, str) and nested:
            return nested
    return None


def format_status_one_line(status_payload: dict) -> str:
    timestamp = status_payload.get("timestamp", "-")
    stage = status_payload.get("stage", "-")
    clip = status_payload.get("clip_index", "-")
    percent = status_payload.get("percent", "-")
    message = status_payload.get("message", "")
    bundle_path = _extract_bundle_path(status_payload)
    if bundle_path:
        return f'{timestamp} stage={stage} clip={clip} pct={percent} msg="{message}" bundle_path="{bundle_path}"'
    return f'{timestamp} stage={stage} clip={clip} pct={percent} msg="{message}"'


def format_pretty_line(status_payload: dict) -> str:
    timestamp = status_payload.get("timestamp", "-")
    stage = status_payload.get("stage", "-")
    clip = status_payload.get("clip_index")
    message = status_payload.get("message", "")
    clip_label = f"clip_{clip}" if clip is not None else "-"
    bundle_path = _extract_bundle_path(status_payload)
    if bundle_path:
        return f'[{timestamp}] {stage} {clip_label} msg="{message}" bundle_path="{bundle_path}"'
    return f'[{timestamp}] {stage} {clip_label} msg="{message}"'


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll remote run status over SSH.")
    parser.add_argument("--host", required=True, help="Remote host or IP.")
    parser.add_argument("--user", required=True, help="SSH user.")
    parser.add_argument("--key", required=True, type=Path, help="Path to private SSH key.")
    parser.add_argument("--remote-path", required=True, help="Remote outputs base directory.")
    parser.add_argument("--run-id", required=True, help="Run ID under remote outputs directory.")
    parser.add_argument("--interval-sec", type=float, default=2.0, help="Polling interval (default: 2s).")
    parser.add_argument("--pretty", action="store_true", help="Pretty one-line output format.")
    parser.add_argument(
        "--tail-progress",
        action="store_true",
        help="On status change, also print tail of remote status/progress.log.",
    )
    parser.add_argument(
        "--tail-stdout",
        action="store_true",
        help="On status change, also print tail of remote status/stdout.log.",
    )
    parser.add_argument("--tail-lines", type=int, default=20, help="Tail line count (default: 20).")
    args = parser.parse_args()

    run_base = args.remote_path.rstrip("/").rstrip("\\")
    remote_status_json = f"{run_base}/{args.run_id}/status/status.json"
    remote_progress_log = f"{run_base}/{args.run_id}/status/progress.log"
    remote_stdout_log = f"{run_base}/{args.run_id}/status/stdout.log"

    last_serialized = ""
    last_progress_tail = ""
    last_stdout_tail = ""
    warned_missing = False

    while True:
        try:
            content = _remote_read_file(args.user, args.host, args.key, remote_status_json)
            if content is None:
                if not warned_missing:
                    print(f"Waiting for remote status: {remote_status_json}")
                    warned_missing = True
                time.sleep(max(0.2, args.interval_sec))
                continue
            warned_missing = False
            payload = json.loads(content)
            serialized = json.dumps(payload, sort_keys=True)
        except (json.JSONDecodeError, RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"Status poll error: {exc}", file=sys.stderr)
            time.sleep(max(0.2, args.interval_sec))
            continue

        if serialized != last_serialized:
            if args.pretty:
                print(format_pretty_line(payload))
            else:
                print(format_status_one_line(payload))
            last_serialized = serialized

            if args.tail_progress:
                try:
                    tail = _remote_tail_file(args.user, args.host, args.key, remote_progress_log, args.tail_lines)
                    tail_value = (tail or "").rstrip()
                    if tail_value and tail_value != last_progress_tail:
                        print("---- progress.log (tail) ----")
                        print(tail_value)
                        last_progress_tail = tail_value
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    print(f"Progress tail error: {exc}", file=sys.stderr)

            if args.tail_stdout:
                try:
                    tail = _remote_tail_file(args.user, args.host, args.key, remote_stdout_log, args.tail_lines)
                    tail_value = (tail or "").rstrip()
                    if tail_value and tail_value != last_stdout_tail:
                        print("---- stdout.log (tail) ----")
                        print(tail_value)
                        last_stdout_tail = tail_value
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    print(f"Stdout tail error: {exc}", file=sys.stderr)

            if payload.get("stage") in {"bundle_ready", "complete", "failed", "aborted_timeout", "aborted_user"}:
                break

        time.sleep(max(0.2, args.interval_sec))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
