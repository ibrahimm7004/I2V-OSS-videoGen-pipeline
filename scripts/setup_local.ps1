param(
  [string]$PythonExe = "python",
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment at $VenvDir"
& $PythonExe -m venv $VenvDir

$PythonPath = Join-Path $VenvDir "Scripts\\python.exe"
if (!(Test-Path $PythonPath)) {
  throw "Virtualenv python not found at $PythonPath"
}

Write-Host "Upgrading pip"
& $PythonPath -m pip install --upgrade pip

Write-Host "Installing requirements"
& $PythonPath -m pip install -r requirements.txt

Write-Host "Done. Activate with: .\\$VenvDir\\Scripts\\Activate.ps1"

