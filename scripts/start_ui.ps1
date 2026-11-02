param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8501,
    [int]$ApiPort = 8000
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Virtual environment Python was not found." }
$env:BANKRUPTCY_API_BASE_URL = "http://127.0.0.1:$ApiPort"
& $python -m streamlit run (Join-Path $ProjectRoot "src\bankruptcy_prediction\ui\app.py") --server.address 127.0.0.1 --server.port $Port
