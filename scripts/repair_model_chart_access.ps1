# Restore read-only access to the existing candidate model charts.
# Run from an elevated PowerShell window. This never edits model data or binaries.
# The script is deliberately restricted to one known candidate directory.
#Requires -RunAsAdministrator
[CmdletBinding()]
param(
    [string]$ReadUser = "$env:COMPUTERNAME\ittraining"
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$candidate = Join-Path $root 'models\candidate-20261022T105726Z-155ceb0b'
$item = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
if (-not $item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
    throw 'Candidate path is not a normal directory; refusing to change permissions.'
}
$aclTool = Join-Path $env:SystemRoot 'System32\icacls.exe'
$ownerTool = Join-Path $env:SystemRoot 'System32\takeown.exe'
$grant = "${ReadUser}:(OI)(CI)RX"

Write-Host "Scope: $candidate"
Write-Host "Granting read-and-execute access ONLY to $ReadUser"
& $aclTool $candidate /grant $grant /T /C
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'ACL could not be modified directly. Recovering ownership of this candidate subtree only.'
    & $ownerTool /F $candidate /A /R /D Y
    if ($LASTEXITCODE -ne 0) {
        throw 'Ownership recovery failed. No broader permissions have been changed.'
    }
    & $aclTool $candidate /grant $grant /T /C
    if ($LASTEXITCODE -ne 0) {
        throw 'The candidate directory remains inaccessible. Inspect its effective permissions manually.'
    }
}

$files = @('precision_recall.png','calibration.png','confusion_matrix.png','feature_importance.png')
foreach ($name in $files) {
    $path = Join-Path (Join-Path $candidate 'charts') $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Write-Warning "Chart missing: $name"
    } else {
        Write-Host "Available chart: $name"
    }
}
Write-Host 'Permission repair complete. Relaunch Streamlit or refresh the browser.'
