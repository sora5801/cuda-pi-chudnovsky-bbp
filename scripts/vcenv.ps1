# =============================================================================
# vcenv.ps1  --  Import the Visual Studio "Developer" environment into the
#                current PowerShell session so that `nvcc` (and `cl.exe`) work.
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# NVIDIA's CUDA compiler driver, nvcc, does NOT compile host (CPU) C++ code by
# itself. On Windows it shells out to Microsoft's C++ compiler, cl.exe. For that
# to succeed, three things must be true in the environment nvcc runs in:
#
#   1. cl.exe must be on PATH.
#   2. The INCLUDE environment variable must point at the MSVC C++ standard
#      library headers AND the Windows SDK headers.
#   3. The LIB environment variable must point at the matching import libraries.
#
# Visual Studio ships a batch file, vcvars64.bat, that sets all of these for a
# 64-bit (x64) build. The problem: vcvars64.bat is a *cmd.exe* script, and the
# environment changes it makes do not automatically propagate back to a parent
# PowerShell process. This helper bridges that gap: it runs vcvars64.bat inside
# a child cmd.exe, dumps the resulting environment with `set`, and copies every
# variable back into the *current* PowerShell session's environment.
#
# After dot-sourcing this file ( `. .\scripts\vcenv.ps1` ) you can call nvcc and
# cl.exe directly in the same session.
#
# It locates Visual Studio with vswhere.exe (the official, version-independent
# way to find VS installs), so it keeps working across VS 2019/2022/2026 without
# hard-coding a path.
# =============================================================================

# Find vswhere.exe -- it lives in a fixed location regardless of which VS edition
# (Community/Professional/Enterprise) or version is installed.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at $vswhere. Is Visual Studio installed?"
}

# Ask vswhere for the newest install that has the VC++ toolset component.
# -latest         : newest version
# -prerelease     : also consider preview builds (VS 2026 ships as 18.x)
# -products *     : any edition
# -requires ...VC.Tools... : must include the C++ compiler
# -property installationPath : print just the install root
$vsRoot = & $vswhere -latest -prerelease -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsRoot) {
    # Fall back to "any install" if the component query came up empty.
    $vsRoot = & $vswhere -latest -prerelease -products * -property installationPath
}
if (-not $vsRoot) { throw "Could not locate a Visual Studio installation via vswhere." }

$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

# Run vcvars64.bat in a child cmd, then `set` to print the fully-populated
# environment. We parse each "NAME=VALUE" line and mirror it into PowerShell.
# The `2>&1` keeps cmd's banner from polluting the variable stream; we only
# import lines that look like real assignments.
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]
        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

Write-Host "[vcenv] Imported MSVC x64 environment from: $vsRoot" -ForegroundColor Cyan
