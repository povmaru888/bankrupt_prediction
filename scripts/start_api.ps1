param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Virtual environment Python was not found." }
$env:PYTHONPATH = (Join-Path $ProjectRoot "src")
Push-Location -LiteralPath $ProjectRoot
try {
    & $python -m uvicorn bankruptcy_prediction.api.app:create_runtime_app --factory --host 127.0.0.1 --port $Port
} finally {
    Pop-Location
}
