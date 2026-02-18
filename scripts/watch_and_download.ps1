param(
  [Parameter(Mandatory = $true)][string]$HostName,
  [Parameter(Mandatory = $true)][string]$User,
  [Parameter(Mandatory = $true)][string]$KeyPath,
  [Parameter(Mandatory = $true)][string]$RemoteOutputsPath,
  [Parameter(Mandatory = $true)][string]$RunId,
  [Parameter(Mandatory = $true)][string]$LocalDir,
  [int]$IntervalSec = 5,
  [switch]$ComputeHash
)

$ErrorActionPreference = "Stop"

$remoteRunDir = ($RemoteOutputsPath.TrimEnd('/') + "/" + $RunId)
$remoteStatus = $remoteRunDir + "/status/status.json"
$remoteManifest = $remoteRunDir + "/manifest.json"

$lastStatus = ""
while ($true) {
  try {
    $statusRaw = & ssh -i $KeyPath "$User@$HostName" "if [ -f '$remoteStatus' ]; then cat '$remoteStatus'; else echo __MISSING__; fi"
    if ($statusRaw.Trim() -eq "__MISSING__") {
      Write-Host "Waiting for remote status file: $remoteStatus"
      Start-Sleep -Seconds $IntervalSec
      continue
    }

    if ($statusRaw -ne $lastStatus) {
      $status = $statusRaw | ConvertFrom-Json
      $clip = $status.clip_index
      if ($null -eq $clip) { $clip = "-" }
      Write-Host "[$($status.timestamp)] stage=$($status.stage) clip=$clip msg=$($status.message)"
      $lastStatus = $statusRaw
    }

    $manifestRaw = & ssh -i $KeyPath "$User@$HostName" "if [ -f '$remoteManifest' ]; then cat '$remoteManifest'; else echo __MISSING__; fi"
    if ($manifestRaw.Trim() -ne "__MISSING__") {
      $manifest = $manifestRaw | ConvertFrom-Json
      $bundleRel = $manifest.outputs.bundle_path
      if ($bundleRel) {
        $remoteBundle = $bundleRel
        if (-not ($bundleRel -match '^/')) {
          $remoteBundle = $remoteRunDir + "/" + $bundleRel
        }
        $exists = & ssh -i $KeyPath "$User@$HostName" "if [ -f '$remoteBundle' ]; then echo YES; else echo NO; fi"
        if ($exists.Trim() -eq "YES") {
          $downloadScript = Join-Path $PSScriptRoot "download_bundle.ps1"
          if ($ComputeHash) {
            & $downloadScript -HostName $HostName -User $User -KeyPath $KeyPath -RemoteOutputsPath $RemoteOutputsPath -RunId $RunId -LocalDir $LocalDir -ComputeHash
          } else {
            & $downloadScript -HostName $HostName -User $User -KeyPath $KeyPath -RemoteOutputsPath $RemoteOutputsPath -RunId $RunId -LocalDir $LocalDir
          }
          Write-Host "Auto-download complete."
          exit 0
        }
      }
    }
  }
  catch {
    Write-Host "Watch error: $($_.Exception.Message)"
  }

  Start-Sleep -Seconds $IntervalSec
}
