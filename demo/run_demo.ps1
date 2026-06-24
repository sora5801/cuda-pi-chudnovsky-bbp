# =============================================================================
#  run_demo.ps1  --  Build (if needed) and run the narrated demo
# =============================================================================
#
#  The easiest way to see everything working. Builds the project with nvcc if the
#  demo executable is missing, then runs it.
#
#  USAGE:
#    powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$demoExe = Join-Path $root "build\pi_demo.exe"

if (-not (Test-Path $demoExe)) {
    Write-Host "[run_demo] pi_demo.exe not found; building first..." -ForegroundColor Yellow
    & "$root\scripts\build.ps1"
}

Write-Host "[run_demo] Launching the demo...`n" -ForegroundColor Cyan
& $demoExe
