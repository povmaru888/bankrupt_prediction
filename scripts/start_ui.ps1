param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8502,
    [int]$ApiPort = 8000
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Virtual environment Python was not found." }
$env:BANKRUPTCY_API_BASE_URL = "http://127.0.0.1:$ApiPort"
$env:PYTHONPATH = (Join-Path $ProjectRoot "src")
$entryPoint = Join-Path $ProjectRoot "streamlit_app.py"
Push-Location -LiteralPath $ProjectRoot
try {
    & $python -m streamlit run $entryPoint --server.address 127.0.0.1 --server.port $Port
} finally {
    Pop-Location
}
