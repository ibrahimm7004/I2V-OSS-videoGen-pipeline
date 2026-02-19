# WAN Next Run Patch Report

Date: 2026-02-19  
Repo: `C:\Users\hp\Desktop\github-projects\I2V-OSS-videoGen-pipeline`

## Completed Checklist

- [x] WAN frame quantization changed to `4n+1` (nearest `>= target_frames`)
- [x] WAN metadata now includes `quantization_rule: "4n+1"` and best-effort `backend_effective_num_frames`
- [x] WAN repo-id selection made config-driven (`WAN22_REPO_ID`, default Diffusers repo)
- [x] `repo_id_used` recorded in clip metadata and propagated to run manifest `model_runtime`
- [x] WAN export quality knob added (`generation_defaults.export_quality` / `WAN22_EXPORT_QUALITY`, default `9`)
- [x] WAN export preflight added (`imageio_ffmpeg` import + `ffmpeg -version` check)
- [x] WAN export metadata added (`export_quality_used`, `ffmpeg_version`, `imageio_ffmpeg_available`)
- [x] Profile-aware post-clip static validator defaults added (WAN quality default `0.003` unless overridden)
- [x] Runner writes effective validator thresholds into metadata
- [x] Optional WAN continuity chaining added (`generation_defaults.chain_last_frame`)
- [x] Continuity debug artifacts added under `outputs/<run_id>/debug/continuity/`
- [x] `jobs/idea01_wan.yaml` updated for quality + native WAN 720p settings (`1280x704`)
- [x] README updated for all new WAN knobs and behavior
- [x] Added fast non-GPU test script for quantization + config precedence

## Files Changed

- `models/wan22.py`
- `pipeline/config.py`
- `pipeline/runner.py`
- `scripts/_job_loading.py`
- `jobs/idea01_wan.yaml`
- `README.md`
- `scripts/test_wan_patch_basics.py` (new)

## Key Implementation Notes

### 1) Frame quantization rule

File: `models/wan22.py`

```python
def _choose_wan_frame_count(target_frames: int) -> int:
    if target_frames <= 1:
        return 1
    return ((target_frames - 1 + 3) // 4) * 4 + 1
```

And runtime metadata now includes:
- `target_frames`
- `wan_num_frames`
- `quantization_rule: "4n+1"`
- `backend_effective_num_frames` (best effort)
- `encoded_frames`
- `encoded_duration_sec`
- `fps`

### 2) Repo id source of truth

File: `pipeline/config.py`
- Added:
  - `wan22_repo_id` defaulting to `WAN22_REPO_ID` env or `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
  - `wan22_export_quality`

File: `models/wan22.py`
- Candidate selection now starts from `config.wan22_repo_id`.
- Legacy candidates remain for backward compatibility.
- If env differs from config, warning is logged.

Manifest propagation:
- File: `pipeline/runner.py`
- `manifest["model_runtime"]["repo_id_used"]` is now set from adapter metadata.

### 3) Export quality + preflight

File: `models/wan22.py`
- Export quality resolution precedence:
  1. `shot.params.export_quality`
  2. `global_params.export_quality`
  3. `config.wan22_export_quality` (env/default)

- Added export preflight:
  - `import imageio_ffmpeg` must succeed
  - `ffmpeg -version` must succeed
  - hard-fail with actionable errors otherwise

- Export call now passes:
  - `quality=<resolved_quality>`
  - `fps=<requested_fps>`

### 4) Static validation defaults (quality-aware)

File: `pipeline/config.py`
- `post_clip_min_frame_diff` now supports unset (`None`) instead of forcing `0.0`.

File: `pipeline/runner.py`
- Effective min frame diff logic:
  - Explicit `POST_CLIP_MIN_FRAME_DIFF` override wins.
  - If WAN + `wan_profile=quality` -> default `0.003`
  - If WAN + `wan_profile=smoke` -> default `0.0`

- Effective values are stored in clip metadata:
  - `post_clip_validation.min_frame_diff_effective`
  - `post_clip_validation.min_size_bytes_effective`

### 5) Optional continuity chaining

File: `scripts/_job_loading.py`
- Maps `generation_defaults.chain_last_frame` into runtime `global_params`.

File: `pipeline/runner.py`
- For WAN:
  - `chain_last_frame=true`: clip `i>0` uses previous last frame
  - `false` (default): clip `i>0` uses initial image again
- Writes continuity copies to:
  - `outputs/<run_id>/debug/continuity/last_frame_clip_XXX.png`
- Metadata includes:
  - `chain_last_frame_enabled`
  - `init_image_used_path`
  - `previous_clip_last_frame_path`

## Job YAML Update

File: `jobs/idea01_wan.yaml`

- `video.height: 704` (with `width: 1280`)
- `generation_defaults.wan_profile: "quality"`
- `generation_defaults.steps: 52`
- `generation_defaults.guidance_scale: 5.5`
- `generation_defaults.motion_strength: 0.65`
- `generation_defaults.export_quality: 9`
- `fps=24`, `clip_duration_sec=5`, `num_clips=8` retained

## Validation Results

Executed successfully:

```powershell
python -m compileall pipeline models scripts
python scripts/test_wan_patch_basics.py
python scripts/test_run_all_schema_parity.py
python scripts/verify_jobpacks.py --jobs jobs/idea01_wan.yaml
python scripts/run_all.py --jobs jobs/idea01_wan.yaml --out outputs --dry-run
```

Observed:
- compileall: pass
- WAN patch basics: pass
- schema parity: pass
- verify_jobpacks (idea01_wan): pass
- run_all dry-run (mock override): pass

## Next Paid Vast Run: Recommended Commands

```bash
cd /workspace/I2V-OSS-videoGen-pipeline
source .venv/bin/activate

export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export WAN22_REPO_ID="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
export WAN22_EXPORT_QUALITY=9
export POST_CLIP_VALIDATION_ENABLED=true
# Optional explicit override (otherwise WAN quality defaults to 0.003):
# export POST_CLIP_MIN_FRAME_DIFF=0.003

python scripts/prefetch.py --models wan
python scripts/run_job.py jobs/idea01_wan.yaml
```

## Expected Outcomes (Quality Run)

- Frame target per clip: `round(24 * 5) = 120`
- WAN quantized frames: `121` (`4n+1`)
- Resolution path: `1280x704` (native WAN 720p lane)
- Export quality used: `9`
- Static validator active for WAN quality profile by default (`min_frame_diff_effective=0.003` unless overridden)
- Explicit metadata for troubleshooting:
  - `repo_id_used`
  - `quantization_rule`
  - `wan_num_frames`
  - `backend_effective_num_frames`
  - `encoded_frames`
  - `encoded_duration_sec`
  - `export_quality_used`
  - `ffmpeg_version`
  - `imageio_ffmpeg_available`

