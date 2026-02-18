# Vast.ai Runbook

This runbook is for unattended runs on a Vast.ai Linux instance, with control and downloads from a local Windows PC.

## 1) SSH into instance

```bash
ssh -i /path/to/key root@<INSTANCE_IP>
```

## 2) Clone repo and bootstrap

```bash
git clone <YOUR_REPO_URL>
cd I2V-OSS-videoGen-pipeline
bash scripts/vast_bootstrap.sh
source .venv/bin/activate
```

## 3) Set Hugging Face cache to persistent storage

Example using `/workspace`:

```bash
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/hf-cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
```

Set token if required by model repos:

```bash
export HF_TOKEN="hf_xxx"
```

## 4) Prefetch weights (download-only)

Dry run:

```bash
python scripts/prefetch.py --models all --dry-run
```

Smoke prefetch (20 seconds):

```bash
python scripts/prefetch.py --models all --smoke --smoke-seconds 20
```

Full prefetch:

```bash
python scripts/prefetch.py --models all
```

## 5) Run jobs

Single job:

```bash
python -m scripts.run_job --job jobs/example_wan.yaml --out outputs
```

Unattended all-jobs sequence:

```bash
python scripts/run_all.py --out outputs
```

Remote helper with stdout log:

```bash
bash scripts/remote_run.sh
```

## 6) Control running jobs (pause/resume/stop)

From local machine (print SSH command only):

```powershell
python scripts\control_run.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --pause
```

Execute immediately:

```powershell
python scripts\control_run.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --stop --execute
```

## 7) Monitor from local PC

```powershell
python scripts\watch_status.py --host <INSTANCE_IP> --user root --key C:\keys\vast.pem --remote-path /workspace/I2V-OSS-videoGen-pipeline/outputs --run-id <run_id> --pretty --tail-progress --tail-stdout
```

## 8) Download bundle(s) to local PC

Single run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_bundle.ps1 -HostName <INSTANCE_IP> -User root -KeyPath C:\keys\vast.pem -RemoteOutputsPath /workspace/I2V-OSS-videoGen-pipeline/outputs -RunId <run_id> -LocalDir C:\downloads\i2v
```

Watch and auto-download on completion:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\watch_and_download.ps1 -HostName <INSTANCE_IP> -User root -KeyPath C:\keys\vast.pem -RemoteOutputsPath /workspace/I2V-OSS-videoGen-pipeline/outputs -RunId <run_id> -LocalDir C:\downloads\i2v
```

## 9) Cleanup

Delete one run directory:

```bash
bash scripts/cleanup_run.sh --run-id <run_id>
```

Delete run + cache (dangerous):

```bash
bash scripts/cleanup_run.sh --run-id <run_id> --delete-cache --cache-path /workspace/hf-cache
```

## 10) Billing warning

- Deleting files reduces storage usage.
- Deleting files **does not stop instance billing**.
- To stop charges, you must stop/destroy the instance in Vast UI/CLI.
