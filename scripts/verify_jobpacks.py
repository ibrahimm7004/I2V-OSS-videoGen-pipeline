from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import sys

from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.job_schema import JobSpec
from pipeline.output_schema import ManifestSchema
from pipeline.runner import run_job
from pipeline.utils import load_structured_file


ALLOWED_MODEL_IDS = {"wan22_ti2v_5b", "hunyuan_i2v", "cogvideox15_5b_i2v"}


class HFCacheSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    HF_HOME: str
    HF_HUB_CACHE: str


class SeedStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    base_seed: int
    per_clip_offsets: list[int]


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    model_id: str
    output_dir: str
    seed_strategy: SeedStrategy
    hf_cache: HFCacheSpec


class VideoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    clip_duration_sec: float = Field(gt=0)
    num_clips: int = Field(gt=0)


class GenerationDefaults(BaseModel):
    model_config = ConfigDict(extra="allow")
    steps: int = Field(gt=0)
    guidance_scale: float | int
    sampler: str


class RifeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    boundary_sec: float | int


class StitchingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    mode: str
    crossfade_sec: float | int
    rife_boundary_smoothing: RifeSpec


class PromptsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    global_prompt: str
    negative_prompt: str
    continuity_rules: list[str]


class InputsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_images: list[str]


class ShotPack(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    prompt: str
    name: str | None = None


class JobPackSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int | str
    run: RunSpec
    hf_cache: dict[str, Any] | None = None
    video: VideoSpec
    generation_defaults: GenerationDefaults
    stitching: StitchingSpec
    prompts: PromptsSpec
    inputs: InputsSpec
    shots: list[ShotPack]


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


def _convert_to_runtime_job(pack: JobPackSpec, run_id: str) -> dict[str, Any]:
    frames = max(1, int(round(pack.video.clip_duration_sec * pack.video.fps)))
    fast_fps = 4
    fast_duration = 0.5
    fast_frames = max(1, int(round(fast_duration * fast_fps)))

    shots: list[dict[str, Any]] = []
    for idx, shot in enumerate(pack.shots):
        seed_offset = pack.run.seed_strategy.per_clip_offsets[idx]
        shots.append(
            {
                "prompt": shot.prompt,
                "negative_prompt": pack.prompts.negative_prompt,
                "duration_seconds": fast_duration,
                "steps": min(8, int(pack.generation_defaults.steps)),
                "fps": fast_fps,
                "frames": fast_frames,
                "seed": int(pack.run.seed_strategy.base_seed + seed_offset),
                "width": min(512, int(pack.video.width)),
                "height": min(512, int(pack.video.height)),
                "params": {
                    "cfg": pack.generation_defaults.guidance_scale,
                    "sampler": pack.generation_defaults.sampler,
                },
            }
        )

    runtime = {
        "job_name": run_id,
        "run_id": run_id,
        "model": {"id": "mock", "version": "VERIFY_MOCK"},
        "output_root": pack.run.output_dir,
        "dry_run": True,
        "global_params": {
            "cfg": pack.generation_defaults.guidance_scale,
            "sampler": pack.generation_defaults.sampler,
        },
        "constants": {
            "global_prompt": pack.prompts.global_prompt,
            "continuity_rules": pack.prompts.continuity_rules,
        },
        "input_image": pack.inputs.initial_images[0] if pack.inputs.initial_images else None,
        "shots": shots,
    }
    # Ensure JobSpec compatibility using the repo schema.
    JobSpec.model_validate(runtime)
    return runtime


def _run_mock_smoke(job_path: Path, pack: JobPackSpec, failures: list[str]) -> tuple[bool, bool]:
    run_id = f"verify-{job_path.stem}-mock"
    run_dir = (REPO_ROOT / "outputs" / run_id).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)

    runtime_job = _convert_to_runtime_job(pack, run_id)
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


def main() -> int:
    job_paths = [
        REPO_ROOT / "jobs" / "idea01_wan.yaml",
        REPO_ROOT / "jobs" / "idea02_hunyuan.yaml",
        REPO_ROOT / "jobs" / "idea03_cogvideox.yaml",
    ]

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
        print(f"- {path.relative_to(REPO_ROOT).as_posix()}")
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

        try:
            raw = load_structured_file(job_path)
            pack = JobPackSpec.model_validate(raw)
        except Exception as exc:
            _fail(failures, f"{job_path.name}: YAML/schema parse failed: {exc}")
            rows.append(row)
            continue

        before_schema_failures = len(failures)
        # Required field semantics checks.
        if pack.run.model_id not in ALLOWED_MODEL_IDS:
            _fail(failures, f"{job_path.name}: model_id '{pack.run.model_id}' not in {sorted(ALLOWED_MODEL_IDS)}")
        if pack.video.num_clips != len(pack.shots):
            _fail(failures, f"{job_path.name}: video.num_clips={pack.video.num_clips} but shots={len(pack.shots)}")
        if len(pack.run.seed_strategy.per_clip_offsets) != pack.video.num_clips:
            _fail(
                failures,
                f"{job_path.name}: per_clip_offsets length={len(pack.run.seed_strategy.per_clip_offsets)} "
                f"!= num_clips={pack.video.num_clips}",
            )
        if not pack.prompts.global_prompt.strip():
            _fail(failures, f"{job_path.name}: prompts.global_prompt is empty")
        if not pack.prompts.negative_prompt.strip():
            _fail(failures, f"{job_path.name}: prompts.negative_prompt is empty")
        if not pack.prompts.continuity_rules:
            _fail(failures, f"{job_path.name}: prompts.continuity_rules is empty")
        if not pack.stitching.enabled:
            _fail(failures, f"{job_path.name}: stitching.enabled must be true")
        if pack.stitching.mode != "concat":
            _fail(failures, f"{job_path.name}: stitching.mode must be 'concat'")
        if not isinstance(pack.stitching.crossfade_sec, (int, float)):
            _fail(failures, f"{job_path.name}: stitching.crossfade_sec must be numeric")

        row["schema_ok"] = len(failures) == before_schema_failures

        # Asset folder + required refs checks.
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

        # Validate inputs.initial_images paths relative to repo root.
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
