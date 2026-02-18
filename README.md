# I2V OSS Video-Gen Test Pipeline

Lightweight Python scaffold for deterministic, job-spec-driven video-gen testing.
No Docker; WAN 2.2 TI2V adapter is implemented, while other model adapters remain stubs.

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
- Model adapter interface with:
  - `wan22_ti2v_5b` implemented (Diffusers WAN I2V)
  - `hunyuan_i2v` stub
  - `cogvideox15_5b_i2v` stub
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

WAN-only full prefetch for runtime weights/components:

```powershell
python scripts\prefetch.py --models wan
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

## Run One Job

Local:

```powershell
python scripts\run_job.py jobs\example_mock.yaml --set run_id=my-run
```

Linux/Vast:

```bash
python -m scripts.run_job --job jobs/example_wan.yaml --out outputs
```

## Run All Jobs Sequentially

Uses `jobs/idea01_wan.yaml jobs/idea02_hunyuan.yaml jobs/idea03_cogvideox.yaml` if present; otherwise falls back to `jobs/example_mock.yaml` x3.

```powershell
python scripts\run_all.py --out outputs
```

Generated summary:

```text
outputs/run_all_<timestamp>/run_all_manifest.json
```

### d) Bundle one run into a single archive

```powershell
python scripts\bundle.py --run-dir outputs\<run_id> --format zip
```

## Pause / Resume / Stop

Print control command (safe preview):

```powershell
python scripts\control_run.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --pause
```

Execute immediately on remote:

```powershell
python scripts\control_run.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --resume --execute
```

Stop:

```powershell
python scripts\control_run.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --stop --execute
```

## Watch Status (SSH)

```powershell
python scripts\watch_status.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --pretty
```

With progress/stdout tails:

```powershell
python scripts\watch_status.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --pretty --tail-progress --tail-stdout --tail-lines 20
```

Run helper on instance (writes `status/stdout.log`):

```bash
bash scripts/remote_run.sh
```

## Download Bundle To Local

Single download:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_bundle.ps1 -HostName <INSTANCE_IP> -User root -KeyPath C:\keys\vast.pem -RemoteOutputsPath /workspace/I2V-OSS-videoGen-pipeline/outputs -RunId <run_id> -LocalDir C:\downloads\i2v
```

Auto-watch and download when ready:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\watch_and_download.ps1 -HostName <INSTANCE_IP> -User root -KeyPath C:\keys\vast.pem -RemoteOutputsPath /workspace/I2V-OSS-videoGen-pipeline/outputs -RunId <run_id> -LocalDir C:\downloads\i2v
```

## Implementation Guide (Bundle Ready Flow)

1) Start unattended run-all on instance:

```bash
python scripts/run_all.py --out outputs
```

2) Watch from local with pretty status:

```powershell
python scripts\watch_status.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --pretty
```

3) When stage is `bundle_ready`, download:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_bundle.ps1 -HostName <INSTANCE_IP> -User root -KeyPath C:\keys\vast.pem -RemoteOutputsPath /workspace/I2V-OSS-videoGen-pipeline/outputs -RunId <run_id> -LocalDir C:\downloads\i2v
```

4) Downloading a finished bundle does not interfere with subsequent runs.

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

## WAN Adapter Smoke (Vast)

```bash
python scripts/prefetch.py --models wan
python scripts/smoke_adapter_wan22.py --input-image assets/idea01/ref_01.png --out-dir outputs/_adapter_smoke/wan22
```

## Notes

- WAN 2.2 TI2V inference is implemented.
- Hunyuan and CogVideoX adapter files still contain TODO placeholders.
- `manifest.json` records `HF_HOME` and `HF_HUB_CACHE` values from environment/config.
- `progress.log` format is: `ISO8601 | stage | clip=<index|-> | msg=...`
- Full deployment runbook: `docs/vast_runbook.md`
