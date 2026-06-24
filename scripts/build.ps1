# =============================================================================
#  build.ps1  --  One-command build with nvcc (no CMake / no IDE required)
# =============================================================================
#
#  This is the most robust way to build on Windows: it imports the Visual Studio
#  environment (so nvcc can find cl.exe) and compiles the three executables
#  directly with nvcc. It does not depend on CMake or any project files.
#
#  USAGE:
#    powershell -ExecutionPolicy Bypass -File scripts\build.ps1
#    powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Arch sm_89 -Config Debug
#
#  Output: build\pi.exe, build\pi_selftest.exe, build\pi_demo.exe
# =============================================================================

param(
    [string]$Arch   = "sm_75",     # GPU architecture (Turing/RTX 2080 = sm_75)
    [string]$Config = "Release"    # Release or Debug
)

$ErrorActionPreference = "Stop"

# Project root is the parent of this script's folder.
$root = Split-Path -Parent $PSScriptRoot

# Import the MSVC x64 environment so nvcc can locate cl.exe and the SDK headers.
. "$PSScriptRoot\vcenv.ps1"

$out = Join-Path $root "build"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$inc = Join-Path $root "include"

# The shared library sources (everything except the three entry points).
$libSrc = @(
    "src\bignum.cpp", "src\bignum_div.cpp", "src\ntt_cpu.cpp", "src\ntt_cuda.cu",
    "src\chudnovsky.cpp", "src\bbp_cpu.cpp", "src\bbp_cuda.cu"
) | ForEach-Object { Join-Path $root $_ }

# Optimization flags per configuration. -G adds device debug info.
# (We deliberately do NOT use --use_fast_math: BBP relies on accurate IEEE-754
#  double division, and the NTT is pure integer math that fast-math cannot help.)
$optFlags = if ($Config -eq "Debug") { @("-G", "-O0") } else { @("-O2") }
$common = @("-arch=$Arch", "-std=c++17", "-I", $inc) + $optFlags

function Build-Exe([string]$mainFile, [string]$exeName) {
    Write-Host "[build] $exeName ($Config, $Arch)" -ForegroundColor Cyan
    & nvcc @common @libSrc $mainFile -o (Join-Path $out $exeName)
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed building $exeName" }
}

Build-Exe (Join-Path $root "src\main.cpp")        "pi.exe"
Build-Exe (Join-Path $root "tests\self_test.cu")  "pi_selftest.exe"
Build-Exe (Join-Path $root "demo\demo.cu")        "pi_demo.exe"

Write-Host "[build] Done. Executables are in: $out" -ForegroundColor Green
Write-Host "        Try:  build\pi.exe --digits 1000 --verify" -ForegroundColor Green
