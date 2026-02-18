# I2V OSS Video-Gen Test Pipeline

Lightweight Python scaffold for deterministic, job-spec-driven video-gen testing.
No Docker, no heavy model inference code yet.

## What This Includes

- Job spec schema (`YAML`/`JSON`) with `pydantic` validation.
- Local runner with:
  - Per-clip generation
  - Last-frame chaining (`clip_i` uses `last_frame_{i-1}` as input)
  - Per-clip logs + run manifest
  - Progress tracking via `status/status.json` + `status/progress.log`
  - Guardrails:
    - Abort if any clip exceeds 10 minutes
    - Abort if total wall time exceeds `planned_clip_count * 10 minutes`
- Model adapter interface + stubs for:
  - `wan22_ti2v_5b`
  - `hunyuan_i2v`
  - `cogvideox15_5b_i2v`
- Deterministic `mock` adapter for dry runs (creates dummy MP4 quickly).
- Stitching stub with deterministic concat placeholder.
- Bundling utility to create one archive per run.

## Repo Layout

```text
pipeline/
models/
jobs/
scripts/
outputs/
```

## Setup

### Windows (local)

```powershell
.\scripts\setup_local.ps1
.\.venv\Scripts\Activate.ps1
```

### Vast.ai Linux

```bash
bash scripts/setup_vast.sh
source .venv/bin/activate
```

## FFmpeg Local Folder

Place FFmpeg binaries under:

```text
ffmpeg/bin/ffmpeg(.exe)
ffmpeg/bin/ffprobe(.exe)
```

The resolver priority is:

1. `FFMPEG_BIN` / `FFPROBE_BIN` env vars
2. repo-local `ffmpeg/bin` binaries
3. `ffmpeg` / `ffprobe` on `PATH`

Validation commands:

```powershell
python scripts\ffmpeg_check.py
python scripts\self_check.py --with-ffmpeg
```

## Hugging Face Prefetch

Set an HF token if your repos require auth:

```powershell
$env:HF_TOKEN="hf_xxx"
```

```bash
export HF_TOKEN="hf_xxx"
```

Point cache to a persistent disk on Vast.ai (example):

```bash
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/hf-cache"
```

Run a 20-second smoke prefetch:

```powershell
python scripts\prefetch.py --models all --smoke --smoke-seconds 20
```

Optional cache override:

```powershell
python scripts\prefetch.py --models hunyuan --cache-dir D:\hf-cache --dry-run
```

## Commands

### a) Local dry run with mock adapter

```powershell
python scripts\run_job.py jobs\example_mock.yaml --set dry_run=true
```

### b) Prefetch dry run

```powershell
python scripts\prefetch.py --models all --dry-run
```

### b2) Prefetch selected model family

```powershell
python scripts\prefetch.py --models wan
```

### c) Run job invocation

```powershell
python scripts\run_job.py jobs\example_mock.yaml
```

Alternative CLI form (used by remote helper):

```powershell
python -m scripts.run_job --job jobs/example_mock.yaml --out outputs
```

### d) Bundle one run into a single archive

```powershell
python scripts\bundle.py --run-dir outputs\<run_id> --format zip
```

### Watch progress/status

Remote over SSH (Windows PowerShell):

```powershell
python scripts\watch_status.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id>
```

With progress and stdout tails on change:

```powershell
python scripts\watch_status.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --tail-progress --tail-stdout --tail-lines 20
```

On instance (Linux), start a run with stdout logging to `outputs/<run_id>/status/stdout.log`:

```bash
bash scripts/remote_run.sh
```

### Run scaffold self-check (no pytest)

```powershell
python scripts\self_check.py
```

### Run scaffold self-check with ffmpeg validation

```powershell
python scripts\self_check.py --with-ffmpeg
```

## Path portability

Validate that manifest/log path fields are relative-friendly:

```powershell
python scripts\path_check.py
```

## Output Location

Each run writes to:

```text
outputs/<run_id>/
```

Expected artifacts:

- `clips/clip_000.mp4`
- `frames/last_frame_000.png`
- `logs/log_000.json`
- `manifest.json`
- `status/status.json`
- `status/progress.log`
- `status/stdout.log` (when using `scripts/remote_run.sh`)
- `final_stitched.mp4`

## Notes

- Real model inference is intentionally not implemented in this scaffold.
- Adapter files contain TODO placeholders for integration.
- `manifest.json` records `HF_HOME` and `HF_HUB_CACHE` values from environment/config.
- `progress.log` format is: `ISO8601 | stage | clip=<index|-> | msg=...`
