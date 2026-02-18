from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.registry import HFRepoSpec, model_ids_for_selector, repos_for_model_id

SMOKE_ALLOW_PATTERNS = [
    "README*",
    "*.json",
    "*.txt",
    "*.md",
]


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for prefetch. Install it with: pip install huggingface_hub"
        ) from exc
    return snapshot_download


def _effective_cache_dir(cache_dir_override: Path | None) -> Path:
    if cache_dir_override is not None:
        return cache_dir_override.resolve()
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser().resolve()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for item in root.rglob("*") if item.is_file())


def _snapshot_worker(
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    allow_patterns: list[str] | None,
    result_path: str,
) -> None:
    result_payload: dict[str, Any]
    try:
        snapshot_download = _load_snapshot_download()
        kwargs: dict[str, Any] = {"repo_id": repo_id}
        if revision:
            kwargs["revision"] = revision
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        local_path = snapshot_download(**kwargs)
        result_payload = {"status": "ok", "local_path": str(local_path)}
    except Exception as exc:
        result_payload = {"status": "error", "error": str(exc)}

    try:
        Path(result_path).write_text(json.dumps(result_payload), encoding="utf-8")
    except Exception:
        pass


def _run_snapshot(repo: HFRepoSpec, cache_dir: Path | None = None) -> str:
    snapshot_download = _load_snapshot_download()
    kwargs: dict[str, Any] = {"repo_id": repo.repo_id}
    if repo.revision:
        kwargs["revision"] = repo.revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    local_path = snapshot_download(**kwargs)
    return str(local_path)


def _run_snapshot_smoke(repo: HFRepoSpec, cache_dir: Path | None, timeout_seconds: int) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        result_file = Path(temp.name)

    process = ctx.Process(
        target=_snapshot_worker,
        args=(
            repo.repo_id,
            repo.revision,
            str(cache_dir) if cache_dir is not None else None,
            SMOKE_ALLOW_PATTERNS,
            str(result_file),
        ),
    )
    process.start()
    process.join(timeout=max(1, timeout_seconds))

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        result_file.unlink(missing_ok=True)
        return {
            "repo_id": repo.repo_id,
            "status": "timeout",
            "detail": f"Stopped after {timeout_seconds}s smoke window.",
        }

    result: dict[str, Any] = {"repo_id": repo.repo_id, "status": "unknown", "detail": "No status returned."}
    if result_file.exists():
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            payload = {"status": "error", "error": "Could not parse worker result file."}
        if payload.get("status") == "ok":
            result["status"] = "ok"
            result["detail"] = payload.get("local_path")
        else:
            result["status"] = "error"
            result["detail"] = payload.get("error", "unknown error")
    result_file.unlink(missing_ok=True)
    return result


def _selected_repos(model_selector: str) -> list[tuple[str, HFRepoSpec]]:
    repos: list[tuple[str, HFRepoSpec]] = []
    seen: set[tuple[str, str | None]] = set()
    for model_id in model_ids_for_selector(model_selector):
        for repo in repos_for_model_id(model_id):
            key = (repo.repo_id, repo.revision)
            if key in seen:
                continue
            seen.add(key)
            repos.append((model_id, repo))
    return repos


def _print_plan(
    *,
    selector: str,
    repos: list[tuple[str, HFRepoSpec]],
    cache_dir: Path,
    smoke: bool,
    smoke_seconds: int,
) -> None:
    print("Prefetch plan")
    print(f"  models selector: {selector}")
    print(f"  smoke mode: {smoke}")
    if smoke:
        print(f"  smoke seconds: {smoke_seconds}")
    print(f"  HF_HOME: {os.getenv('HF_HOME')}")
    print(f"  HF_HUB_CACHE: {os.getenv('HF_HUB_CACHE')}")
    print(f"  effective cache dir: {cache_dir}")
    print("  repos:")
    for model_id, repo in repos:
        revision = repo.revision or "default"
        print(f"    - model={model_id} repo={repo.repo_id} revision={revision}")


def _print_registry_self_test() -> None:
    print("Prefetch registry self-test")
    for model_id in model_ids_for_selector("all"):
        repos = repos_for_model_id(model_id)
        repo_ids = ", ".join(repo.repo_id for repo in repos)
        print(f"  {model_id} -> {repo_ids}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download-only Hugging Face prefetch stage.")
    parser.add_argument(
        "--models",
        choices=["all", "wan", "hunyuan", "cog", "cogvideox", "cogvideox15"],
        default="all",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache directory override.")
    parser.add_argument("--smoke", action="store_true", help="Run short smoke prefetch and stop.")
    parser.add_argument("--smoke-seconds", type=int, default=20, help="Smoke window in seconds (default: 20).")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without downloading.")
    parser.add_argument("--print-registry", action="store_true", help="Print model_id -> repo_id mapping and exit.")
    args = parser.parse_args()

    if args.print_registry:
        _print_registry_self_test()
        return 0

    repos = _selected_repos(args.models)
    cache_dir = _effective_cache_dir(args.cache_dir)
    _print_plan(
        selector=args.models,
        repos=repos,
        cache_dir=cache_dir,
        smoke=args.smoke,
        smoke_seconds=args.smoke_seconds,
    )

    if args.dry_run:
        print("Dry run only. No downloads started.")
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    files_before = _count_files(cache_dir)

    failures: list[str] = []
    if args.smoke:
        total_seconds = max(1, int(args.smoke_seconds))
        per_repo_seconds = max(1, total_seconds // max(1, len(repos)))
        for _, repo in repos:
            result = _run_snapshot_smoke(repo, cache_dir, per_repo_seconds)
            print(f"Smoke repo={repo.repo_id} status={result['status']} detail={result['detail']}")
            if result["status"] == "error":
                failures.append(f"{repo.repo_id}: {result['detail']}")
    else:
        for _, repo in repos:
            try:
                local_path = _run_snapshot(repo, cache_dir)
                print(f"Prefetched repo={repo.repo_id} -> {local_path}")
            except Exception as exc:
                message = f"{repo.repo_id}: {exc}"
                failures.append(message)
                print(f"Prefetch failed: {message}", file=sys.stderr)

    files_after = _count_files(cache_dir)
    created = files_after > files_before
    print(f"Cache files before={files_before} after={files_after} created_any={created}")

    if failures:
        print("Prefetch completed with failures:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
