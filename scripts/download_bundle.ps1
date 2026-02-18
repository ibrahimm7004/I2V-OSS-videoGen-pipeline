param(
  [Parameter(Mandatory = $true)][string]$HostName,
  [Parameter(Mandatory = $true)][string]$User,
  [Parameter(Mandatory = $true)][string]$KeyPath,
  [Parameter(Mandatory = $true)][string]$RemoteOutputsPath,
  [Parameter(Mandatory = $true)][string]$RunId,
  [Parameter(Mandatory = $true)][string]$LocalDir,
  [switch]$ComputeHash
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

$remoteRunDir = ($RemoteOutputsPath.TrimEnd('/') + "/" + $RunId)
$remoteManifest = $remoteRunDir + "/manifest.json"

$manifestRaw = & ssh -i $KeyPath "$User@$HostName" "cat '$remoteManifest'"
if (-not $manifestRaw) {
  throw "Could not read remote manifest: $remoteManifest"
}

$manifest = $manifestRaw | ConvertFrom-Json
$bundleRel = $manifest.outputs.bundle_path
if (-not $bundleRel) {
  throw "manifest.outputs.bundle_path is empty for run '$RunId'"
}

$remoteBundle = $bundleRel
if (-not ($bundleRel -match '^/')) {
  $remoteBundle = $remoteRunDir + "/" + $bundleRel
}

$fileName = [System.IO.Path]::GetFileName($remoteBundle)
$localPath = Join-Path $LocalDir $fileName

Write-Host "Downloading bundle..."
Write-Host "  Remote: $remoteBundle"
Write-Host "  Local:  $localPath"

& scp -i $KeyPath "$User@$HostName:$remoteBundle" $localPath

if (-not (Test-Path $localPath)) {
  throw "Download failed: file not found locally at $localPath"
}

$item = Get-Item $localPath
if ($item.Length -le 0) {
  throw "Download failed: file is empty ($localPath)"
}

Write-Host "Download complete: $localPath ($($item.Length) bytes)"

if ($ComputeHash) {
  $hash = Get-FileHash -Algorithm SHA256 -Path $localPath
  Write-Host "SHA256: $($hash.Hash)"
}
