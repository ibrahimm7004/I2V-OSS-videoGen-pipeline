# Vast.ai Runbook

Automated WAN run workflow for fresh Vast Ubuntu instances.

## Quick start (one command chain)

```bash
ssh -i /path/to/key root@<INSTANCE_IP>
cd /workspace
git clone <YOUR_REPO_URL> I2V-OSS-videoGen-pipeline
cd I2V-OSS-videoGen-pipeline
bash scripts/vast/run_all_steps.sh --repo-url <YOUR_REPO_URL> --prefetch-wan
```

This runs:
- `scripts/vast/00_check_instance.sh`
- `scripts/vast/01_setup_env.sh`
- `scripts/vast/02_smoke_and_preflight.sh`
- `scripts/vast/03_run_wan.sh`

## Manual step-by-step

### 1) Instance check

```bash
bash scripts/vast/00_check_instance.sh --min-free-gb 60
```

### 2) Setup env and cache

```bash
bash scripts/vast/01_setup_env.sh --repo-url <YOUR_REPO_URL> --prefetch-wan
source /workspace/I2V_ENV.sh
```

### 3) Preflight and optional smoke

```bash
bash scripts/vast/02_smoke_and_preflight.sh --job jobs/wan/idea01.yaml
```

Optional 1-clip smoke run:

```bash
bash scripts/vast/02_smoke_and_preflight.sh --job jobs/wan/idea01.yaml --run-smoke --smoke-num-clips 1 --smoke-duration-sec 2 --smoke-steps 12
```

### 4) Launch full WAN run in tmux

```bash
bash scripts/vast/03_run_wan.sh --job jobs/wan/idea01.yaml
```

Optional runtime overrides (without editing YAML):

```bash
bash scripts/vast/03_run_wan.sh --job jobs/wan/idea01.yaml --num-clips 8 --clip-duration-sec 5 --run-id wan-main-001
```

## Runtime environment used by scripts

The scripts export and use:

```bash
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache
export WAN22_REPO_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers
export WAN22_EXPORT_QUALITY=9
export POST_CLIP_VALIDATION_ENABLED=true
export POST_CLIP_MIN_FRAME_DIFF=0.003
```

## Monitor and download from local Windows PC

Watch run status:

```powershell
python scripts\watch_status.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --pretty --tail-progress --tail-stdout
```

Download bundle after `bundle_ready`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_bundle.ps1 -HostName <INSTANCE_IP> -User root -KeyPath C:\keys\vast.pem -RemoteOutputsPath /workspace/I2V-OSS-videoGen-pipeline/outputs -RunId <run_id> -LocalDir C:\downloads\i2v
```

Auto-watch + download:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\watch_and_download.ps1 -HostName <INSTANCE_IP> -User root -KeyPath C:\keys\vast.pem -RemoteOutputsPath /workspace/I2V-OSS-videoGen-pipeline/outputs -RunId <run_id> -LocalDir C:\downloads\i2v
```

## Cleanup and billing reminder

Delete run files:

```bash
bash scripts/cleanup_run.sh --run-id <run_id>
```

Delete run + cache (dangerous):

```bash
bash scripts/cleanup_run.sh --run-id <run_id> --delete-cache --cache-path /workspace/hf-cache
```

Deleting files does not stop billing. Stop or destroy the instance in Vast UI/CLI to stop charges.
