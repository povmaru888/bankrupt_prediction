param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8503
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Virtual environment Python was not found." }
$entryPoint = Join-Path $ProjectRoot "deploy\research_app.py"
Push-Location -LiteralPath $ProjectRoot
try {
    & $python -m streamlit run $entryPoint --server.address 127.0.0.1 --server.port $Port
} finally {
    Pop-Location
}
