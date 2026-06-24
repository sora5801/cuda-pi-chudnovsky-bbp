# =============================================================================
#  gen_vs.ps1  --  Generate a Visual Studio 2026 solution with CMake
# =============================================================================
#
#  Produces build\PiDigits.sln (and the .vcxproj projects) that you can open and
#  build inside Visual Studio 2026. CMake handles the CUDA build customization and
#  the correct platform toolset (v145) for you.
#
#  USAGE:
#    powershell -ExecutionPolicy Bypass -File scripts\gen_vs.ps1
#  Then either:
#    * open build\PiDigits.sln in Visual Studio 2026, or
#    * build from the command line:  cmake --build build --config Release
# =============================================================================

param(
    [string]$Arch = "75"   # CUDA architecture number (75 = Turing/RTX 2080)
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# The CUDA toolkit root (so CMake selects the matching MSBuild integration).
$cuda = $env:CUDA_PATH
if (-not $cuda) { $cuda = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" }

Write-Host "[gen] Generating Visual Studio 2026 solution into build\ ..." -ForegroundColor Cyan
& cmake -S $root -B (Join-Path $root "build") `
    -G "Visual Studio 18 2026" -A x64 `
    -T cuda="$cuda" `
    "-DCMAKE_CUDA_ARCHITECTURES=$Arch"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "[gen] Done. Open build\PiDigits.sln in Visual Studio 2026," -ForegroundColor Green
Write-Host "      or run:  cmake --build build --config Release" -ForegroundColor Green
