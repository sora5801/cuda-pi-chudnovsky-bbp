@echo off
rem ============================================================================
rem  build.bat  --  One-command build with nvcc from a plain cmd.exe prompt
rem ============================================================================
rem  Imports the Visual Studio environment via vcvars64.bat (located with
rem  vswhere), then compiles the three executables with nvcc. Equivalent to
rem  build.ps1 but for users who prefer a batch file / double-clicking.
rem
rem  Usage:  scripts\build.bat            (defaults: sm_75, Release)
rem ============================================================================
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
set "ARCH=sm_75"

rem --- Locate and enter the Visual Studio x64 build environment --------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Is Visual Studio installed?
    exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -property installationPath`) do set "VSROOT=%%i"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo ERROR: vcvars64.bat failed & exit /b 1 )

if not exist "%ROOT%\build" mkdir "%ROOT%\build"

set "INC=%ROOT%\include"
set "LIB=%ROOT%\src\bignum.cpp %ROOT%\src\bignum_div.cpp %ROOT%\src\ntt_cpu.cpp %ROOT%\src\ntt_cuda.cu %ROOT%\src\chudnovsky.cpp %ROOT%\src\bbp_cpu.cpp %ROOT%\src\bbp_cuda.cu"
set "FLAGS=-arch=%ARCH% -O2 -std=c++17 -I "%INC%""

echo [build] pi.exe
nvcc %FLAGS% %LIB% "%ROOT%\src\main.cpp" -o "%ROOT%\build\pi.exe" || exit /b 1
echo [build] pi_selftest.exe
nvcc %FLAGS% %LIB% "%ROOT%\tests\self_test.cu" -o "%ROOT%\build\pi_selftest.exe" || exit /b 1
echo [build] pi_demo.exe
nvcc %FLAGS% %LIB% "%ROOT%\demo\demo.cu" -o "%ROOT%\build\pi_demo.exe" || exit /b 1

echo [build] Done. Executables are in %ROOT%\build
echo         Try:  build\pi.exe --digits 1000 --verify
endlocal
