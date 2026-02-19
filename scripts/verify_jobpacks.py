from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import sys

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.output_schema import ManifestSchema
from pipeline.runner import run_job

try:
    from scripts._job_loading import (
        ALLOWED_MODEL_IDS,
        JobPackSpec,
        collect_job_pack_issues,
        convert_job_pack_to_runtime_job,
        load_job_pack,
    )
except ModuleNotFoundError:
    from _job_loading import (
        ALLOWED_MODEL_IDS,
        JobPackSpec,
        collect_job_pack_issues,
        convert_job_pack_to_runtime_job,
        load_job_pack,
    )


def _image_info(path: Path) -> tuple[int, int] | None:
    try:
        import cv2  # type: ignore

        image = cv2.imread(str(path))
        if image is not None:
            return int(image.shape[1]), int(image.shape[0])
    except Exception:
        return None
    return None


def _fail(messages: list[str], text: str) -> None:
    messages.append(text)


def _warn(warnings: list[str], text: str) -> None:
    warnings.append(text)


def _idea_name_from_job(job_path: Path) -> str | None:
    match = re.search(r"(idea\d+)", job_path.stem.lower())
    return match.group(1) if match else None


def _find_ref_image(asset_dir: Path, ref_name: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = asset_dir / f"{ref_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _convert_to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _run_mock_smoke(job_path: Path, pack: JobPackSpec, failures: list[str]) -> tuple[bool, bool]:
    run_id = f"verify-{job_path.stem}-mock"
    run_dir = (REPO_ROOT / "outputs" / run_id).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)

    runtime_job = convert_job_pack_to_runtime_job(
        pack,
        run_id=run_id,
        model_id_override="mock",
        output_root_override="outputs",
        dry_run=True,
        fast_mode=True,
        source_job_path=job_path,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8", dir=str(REPO_ROOT)) as tmp:
        temp_path = Path(tmp.name)
        import yaml

        yaml.safe_dump(runtime_job, tmp, sort_keys=False)

    try:
        out_dir = run_job(temp_path, overrides={"run_id": run_id, "model.id": "mock", "output_root": "outputs"})
    except Exception as exc:
        _fail(failures, f"{job_path.name}: mock run failed: {exc}")
        temp_path.unlink(missing_ok=True)
        return False, False
    finally:
        temp_path.unlink(missing_ok=True)

    status_path = out_dir / "status" / "status.json"
    if not status_path.exists():
        _fail(failures, f"{job_path.name}: missing status file at {status_path}")
        return False, False

    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    if status_data.get("stage") != "bundle_ready":
        _fail(
            failures,
            f"{job_path.name}: expected final stage bundle_ready, got {status_data.get('stage')}",
        )
        return False, False

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        _fail(failures, f"{job_path.name}: missing manifest.json")
        return False, False

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest = ManifestSchema.model_validate(manifest_data)
    except ValidationError as exc:
        _fail(failures, f"{job_path.name}: manifest validation failed: {exc}")
        return False, False

    bundle_value = manifest.outputs.bundle_path
    if not bundle_value:
        _fail(failures, f"{job_path.name}: manifest missing outputs.bundle_path")
        return True, False
    bundle_path = Path(bundle_value)
    if not bundle_path.is_absolute():
        bundle_path = out_dir / bundle_path
    if not bundle_path.exists():
        _fail(failures, f"{job_path.name}: bundle not found at {bundle_path}")
        return True, False

    required_present = {"manifest.json", "final_stitched.mp4"}
    with ZipFile(bundle_path, "r") as zf:
        members = set(zf.namelist())
    if not required_present.issubset(members):
        missing = sorted(required_present - members)
        _fail(failures, f"{job_path.name}: bundle missing required files: {missing}")
        return True, False
    if not any(item.startswith("clips/") for item in members):
        _fail(failures, f"{job_path.name}: bundle missing clips/")
        return True, False
    if not any(item.startswith("frames/") for item in members):
        _fail(failures, f"{job_path.name}: bundle missing frames/")
        return True, False
    if not any(item.startswith("logs/") for item in members):
        _fail(failures, f"{job_path.name}: bundle missing logs/")
        return True, False

    return True, True


def _default_job_paths() -> list[Path]:
    return [
        REPO_ROOT / "jobs" / "idea01_wan.yaml",
        REPO_ROOT / "jobs" / "idea02_hunyuan.yaml",
        REPO_ROOT / "jobs" / "idea03_cogvideox.yaml",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify job-pack YAMLs, assets, and mock smoke runs.")
    parser.add_argument("--jobs", nargs="+", type=Path, default=None, help="Job-pack YAML paths to verify.")
    args = parser.parse_args()

    if args.jobs:
        job_paths = [path.resolve() for path in args.jobs]
    else:
        job_paths = _default_job_paths()

    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    assets_root = REPO_ROOT / "assets"
    jobs_root = REPO_ROOT / "jobs"
    if not assets_root.exists():
        _fail(failures, "assets/ directory not found.")
    if not jobs_root.exists():
        _fail(failures, "jobs/ directory not found.")

    print("Discovered job YAMLs:")
    for path in job_paths:
        print(f"- {_convert_to_repo_relative(path)}")
        if not path.exists():
            _fail(failures, f"Missing required job file: {path}")

    for job_path in job_paths:
        row = {
            "job": job_path.name,
            "schema_ok": False,
            "assets_ok": False,
            "mock_run_ok": False,
            "bundle_ok": False,
        }
        if not job_path.exists():
            rows.append(row)
            continue

        before_schema_failures = len(failures)
        try:
            pack = load_job_pack(job_path)
        except Exception as exc:
            _fail(failures, f"{job_path.name}: YAML/schema parse failed: {exc}")
            rows.append(row)
            continue

        for issue in collect_job_pack_issues(pack):
            _fail(failures, f"{job_path.name}: {issue}")
        row["schema_ok"] = len(failures) == before_schema_failures

        before_assets_failures = len(failures)
        idea_name = _idea_name_from_job(job_path)
        if not idea_name:
            _fail(failures, f"{job_path.name}: cannot infer idea folder name from filename")
            rows.append(row)
            continue

        asset_dir = assets_root / idea_name
        if not asset_dir.exists():
            _fail(failures, f"{job_path.name}: missing asset folder {asset_dir}")
            rows.append(row)
            continue

        ref_01 = _find_ref_image(asset_dir, "ref_01")
        ref_02 = _find_ref_image(asset_dir, "ref_02")
        if ref_01 is None:
            _fail(failures, f"{job_path.name}: missing required {asset_dir}/ref_01.(png|jpg|jpeg|webp)")
        if ref_02 is None:
            _warn(warnings, f"{job_path.name}: optional ref_02 missing under {asset_dir}")

        for rel in pack.inputs.initial_images:
            resolved = (REPO_ROOT / rel).resolve()
            if not resolved.exists():
                _fail(failures, f"{job_path.name}: referenced image not found: {rel}")
                continue
            size = resolved.stat().st_size
            if size <= 0:
                _fail(failures, f"{job_path.name}: referenced image has zero size: {rel}")
            dims = _image_info(resolved)
            dims_text = f"{dims[0]}x{dims[1]}" if dims else "unknown"
            print(f"  {job_path.name} image: {rel} size={size} bytes dims={dims_text}")

        if pack.inputs.initial_images:
            first_img = (REPO_ROOT / pack.inputs.initial_images[0]).resolve()
            if first_img.exists() and first_img.stat().st_size <= 0:
                _fail(failures, f"{job_path.name}: first image is empty: {first_img}")

        row["assets_ok"] = len(failures) == before_assets_failures

        if pack.run.model_id not in ALLOWED_MODEL_IDS:
            _fail(failures, f"{job_path.name}: model_id '{pack.run.model_id}' is not recognized")
            rows.append(row)
            continue

        mock_ok, bundle_ok = _run_mock_smoke(job_path, pack, failures)
        row["mock_run_ok"] = mock_ok
        row["bundle_ok"] = bundle_ok
        rows.append(row)

    print("\nPer-job summary:")
    for row in rows:
        print(
            f"- {row['job']}: schema={row['schema_ok']} assets={row['assets_ok']} "
            f"mock_run={row['mock_run_ok']} bundle={row['bundle_ok']}"
        )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll job packs verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
