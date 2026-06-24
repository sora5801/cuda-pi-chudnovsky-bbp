# 07 — Building and Running

This document is the practical, hands-on companion to the algorithm docs. It tells
you how to turn the source tree into three working executables and how to drive
them. It is written for **this** machine's exact toolchain, with side-notes for
people on different hardware.

> **The reference stack** (what the code was developed and tested on):
>
> | Component         | Version on this machine                         |
> |-------------------|-------------------------------------------------|
> | CUDA Toolkit      | **13.3** (`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3`) |
> | Host compiler     | **Visual Studio 2026** (v18), MSVC **toolset v145** |
> | GPU               | **NVIDIA RTX 2080 SUPER**, Turing, **sm_75** (`compute_75`) |
> | C++ standard      | **C++17** (host *and* device)                   |
> | Target platform   | **x64** only                                    |

If you are on a different GPU or a different VS version, you do not need to edit
any files — every build path takes the architecture and the compiler as a
parameter. The places to change are called out in [Troubleshooting](#troubleshooting).

There are **three** ways to build, in rough order of robustness:

1. **[nvcc scripts](#1-the-nvcc-scripts-most-robust)** — `scripts/build.ps1` / `scripts/build.bat`. No CMake, no IDE, no project files. The most reliable, and the one the demo/self-test runners fall back to.
2. **[CMake → Visual Studio solution](#2-cmake--a-visual-studio-2026-solution)** — `scripts/gen_vs.ps1` or a raw `cmake` command, or just *Open Folder* in VS using [`CMakePresets.json`](../CMakePresets.json). The recommended way to get an IDE solution and the only path that wires up `ctest`.
3. **[Hand-written `vs/PiDigits.sln`](#3-the-hand-written-visual-studio-solution)** — a checked-in solution with projects `pi` and `pi_demo`, for people who want a `.sln` in version control without running CMake.

All three compile the **same** seven library sources into the **same** three
programs. Here is the dependency picture they all share:

```
            include/*.h  (public headers)
                  |
   ---------------+-----------------------------------------
   |   the "pidigits" math library (7 translation units)   |
   |                                                        |
   |   src/bignum.cpp        big-integer core (host)        |
   |   src/bignum_div.cpp    division + decimal conv (host) |
   |   src/ntt_cpu.cpp       CPU number-theoretic transform |
   |   src/ntt_cuda.cu       GPU NTT (device)               |
   |   src/chudnovsky.cpp    binary-splitting series (host) |
   |   src/bbp_cpu.cpp       BBP on CPU (host)              |
   |   src/bbp_cuda.cu       BBP on GPU (device)            |
   ---------------+-----------------------------------------
                  |
      +-----------+------------+--------------------+
      |                        |                    |
  src/main.cpp          tests/self_test.cu     demo/demo.cu
   -> pi.exe            -> pi_selftest.exe      -> pi_demo.exe
```

`.cpp` files are host C++ compiled by Microsoft's `cl.exe`. `.cu` files contain
device code and are compiled by NVIDIA's `nvcc` (which itself shells out to
`cl.exe` for the host half — that detail drives everything below).

---

## Why building CUDA on Windows is fiddly (read this once)

`nvcc` is a *compiler driver*, not a complete compiler. It splits each `.cu`
file into device code (which it compiles to PTX/SASS itself) and host code
(which it hands to the platform's native C++ compiler). On Windows that native
compiler is **`cl.exe`**, shipped with Visual Studio.

For `nvcc` to find and successfully run `cl.exe`, three environment things must
be true (this is the exact list from [`scripts/vcenv.ps1`](../scripts/vcenv.ps1)):

1. `cl.exe` must be on `PATH`.
2. `INCLUDE` must point at the MSVC C++ standard-library headers **and** the Windows SDK headers.
3. `LIB` must point at the matching import libraries.

Visual Studio provides a batch file, `vcvars64.bat`, that sets all three for an
x64 build. The catch: it is a `cmd.exe` script, and its changes do **not**
propagate back into a parent PowerShell session. The three build paths each
solve this in their own way:

- The **nvcc scripts** import that environment explicitly (PowerShell via `vcenv.ps1`, batch via a direct `call`).
- **CMake** and the **VS solution** rely on MSBuild, which already runs inside the developer environment when you build from the IDE or `cmake --build`.

Keep this in mind: almost every "it won't build" problem on Windows is really
"`nvcc` couldn't find `cl.exe`."

---

## 1. The nvcc scripts (most robust)

> Files: [`scripts/build.ps1`](../scripts/build.ps1), [`scripts/build.bat`](../scripts/build.bat), [`scripts/vcenv.ps1`](../scripts/vcenv.ps1)

This path needs **only** the CUDA Toolkit and Visual Studio installed. No CMake,
no project files, no IDE. It is the most robust because it controls the
environment itself instead of trusting a generator to do it.

### Run it

From the project root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

or from a plain `cmd.exe` prompt:

```bat
scripts\build.bat
```

Either one produces three executables in `build\`:

```
build\pi.exe           the command-line calculator
build\pi_selftest.exe  the correctness test suite
build\pi_demo.exe      the narrated guided tour
```

The PowerShell version takes two optional parameters (the batch file is fixed at
`sm_75` / Release):

```powershell
# Build Debug, and target an Ada (RTX 40-series) GPU instead of Turing:
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Arch sm_89 -Config Debug
```

The defaults come straight from the `param(...)` block in `build.ps1`:

```powershell
param(
    [string]$Arch   = "sm_75",     # GPU architecture (Turing/RTX 2080 = sm_75)
    [string]$Config = "Release"    # Release or Debug
)
```

### How `vcenv.ps1` makes `nvcc` find `cl.exe`

The first real line of `build.ps1` dot-sources the environment helper:

```powershell
. "$PSScriptRoot\vcenv.ps1"
```

`vcenv.ps1` does four things, all to satisfy the "find `cl.exe`" requirement:

1. **Locate Visual Studio with `vswhere.exe`.** `vswhere` lives at a fixed path
   (`${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe`)
   regardless of edition or version, so the script never hard-codes a VS path.
   It asks for the newest install that ships the C++ toolset:

   ```powershell
   $vsRoot = & $vswhere -latest -prerelease -products * `
       -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
       -property installationPath
   ```

   The `-prerelease` flag is what lets it find **VS 2026**, which ships as
   version `18.x`. If the component query comes back empty it falls back to
   "any install."

2. **Find `vcvars64.bat`** under `VC\Auxiliary\Build\` of that install.

3. **Run it in a child `cmd` and harvest the environment.** This is the bridge
   that ordinary dot-sourcing can't cross:

   ```powershell
   cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
       if ($_ -match '^([^=]+)=(.*)$') {
           Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
       }
   }
   ```

   It runs `vcvars64.bat` (silencing its banner), then `set` to dump the
   *fully populated* environment, and copies every `NAME=VALUE` line back into
   the **current** PowerShell session. After this returns, `PATH`, `INCLUDE`,
   and `LIB` are all set, and `nvcc` can find `cl.exe`.

The batch file `build.bat` does the equivalent with a direct
`call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"` — simpler, because a batch
file *is* already a `cmd` script, so the environment just persists.

### What the compile command actually is

After the environment is live, `build.ps1` builds a common flag list and calls
`nvcc` once per executable. The flags are worth understanding:

```powershell
$optFlags = if ($Config -eq "Debug") { @("-G", "-O0") } else { @("-O2") }
$common = @("-arch=$Arch", "-std=c++17", "-I", $inc) + $optFlags
```

| Flag            | Meaning                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `-arch=sm_75`   | Generate SASS for the Turing GPU (RTX 2080 SUPER). Change for other GPUs. |
| `-std=c++17`    | Compile **both** host and device halves as C++17. Required — `pi_reference.h` uses C++17 inline variables. |
| `-I include`    | The public headers live in `include/`.                                  |
| `-O2`           | Release optimization (host side).                                       |
| `-G -O0`        | Debug build: `-G` embeds *device* debug info so you can step kernels in Nsight. |

Every source file is recompiled and linked in a single `nvcc` invocation per
executable — there is no separate "library" object here; the seven library
`.cpp`/`.cu` files are simply added to each command:

```powershell
& nvcc @common @libSrc $mainFile -o (Join-Path $out $exeName)
```

#### A deliberate non-flag: no `--use_fast_math`

The comment in `build.ps1` is a load-bearing engineering note, not boilerplate:

```text
# (We deliberately do NOT use --use_fast_math: BBP relies on accurate IEEE-754
#  double division, and the NTT is pure integer math that fast-math cannot help.)
```

`--use_fast_math` would relax IEEE-754 rounding (flushing denormals, using
approximate reciprocals). The BBP digit-extraction routine
(see [BBP](05_bbp.md)) accumulates a fractional sum in `double` and depends on
each modular division being correctly rounded; fast-math would silently corrupt
extracted digits. The NTT (see [the NTT](03_ntt.md)) is **pure integer**
arithmetic modulo a 64-bit prime — fast-math touches only floating point, so it
could not help even if it were safe. Leaving it off is the correct, honest
default.

---

## 2. CMake → a Visual Studio 2026 solution

> Files: [`CMakeLists.txt`](../CMakeLists.txt), [`scripts/gen_vs.ps1`](../scripts/gen_vs.ps1), [`CMakePresets.json`](../CMakePresets.json)

This is the **recommended** path for IDE work. CMake auto-detects the compilers
and the CUDA toolkit, picks the right MSBuild integration and platform toolset
for you, and — uniquely among the three paths — wires up `ctest`. It builds all
three executables (the hand-written `.sln` builds only two).

### The structure of `CMakeLists.txt`

A few decisions in the listfile are worth reading because they explain the whole
build:

- **Languages and standard.** The project declares `LANGUAGES CXX CUDA` and pins
  C++17 for both:

  ```cmake
  project(PiDigits VERSION 1.0.0 LANGUAGES CXX CUDA)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  ```

- **A single static library, mixed-language.** All seven math sources go into
  one `pidigits` static library. CMake compiles each file with the correct
  compiler automatically — `cl.exe` for `.cpp`, `nvcc` for `.cu`:

  ```cmake
  add_library(pidigits STATIC
      src/bignum.cpp src/bignum_div.cpp src/ntt_cpu.cpp src/ntt_cuda.cu
      src/chudnovsky.cpp src/bbp_cpu.cpp src/bbp_cuda.cu)
  target_link_libraries(pidigits PUBLIC CUDA::cudart)
  ```

  Linking `CUDA::cudart` `PUBLIC` means anything that links `pidigits`
  automatically pulls in the CUDA runtime — so the three executables don't have
  to mention it.

- **The default architecture is 75.** If you don't pass one, you get Turing:

  ```cmake
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  ```

- **Tests.** `enable_testing()` plus `add_test(NAME selftest COMMAND pi_selftest)`
  is what makes `ctest` able to run the self-test.

> **Note the numbering convention.** The nvcc scripts spell the architecture
> `sm_75` (NVCC syntax). CMake wants the **bare number** `75`
> (`CMAKE_CUDA_ARCHITECTURES`). Same GPU, two spellings — don't pass `sm_75` to
> CMake.

### Option A — generate the solution with `gen_vs.ps1`

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gen_vs.ps1
```

This wraps the CMake configure step and adds two important details. First, it
locates the CUDA toolkit so CMake selects the matching MSBuild integration,
falling back to the v13.3 default if `CUDA_PATH` is unset:

```powershell
$cuda = $env:CUDA_PATH
if (-not $cuda) { $cuda = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" }
```

Then it configures with the VS 2026 generator, x64, the CUDA toolset, and the
architecture:

```powershell
& cmake -S $root -B (Join-Path $root "build") `
    -G "Visual Studio 18 2026" -A x64 `
    -T cuda="$cuda" `
    "-DCMAKE_CUDA_ARCHITECTURES=$Arch"
```

It takes one parameter, `-Arch`, defaulting to `"75"`:

```powershell
# Generate a solution targeting an Ampere (sm_86) GPU:
powershell -ExecutionPolicy Bypass -File scripts\gen_vs.ps1 -Arch 86
```

The result is `build\PiDigits.sln`. Open it in Visual Studio 2026, or build it
headless:

```powershell
cmake --build build --config Release
```

### Option B — invoke CMake directly

The same thing, by hand, exactly as documented in the header of `CMakeLists.txt`:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release      # run the self-test
```

`Visual Studio 18 2026` is the generator name for VS 2026 (v18). If you are on
VS 2022, use `-G "Visual Studio 17 2022"`.

### Option C — Open Folder in VS using `CMakePresets.json`

Visual Studio 2026 can configure the project with **no command line at all**:
*File → Open → Folder…*, pick the project root, and VS reads
[`CMakePresets.json`](../CMakePresets.json) automatically.

The presets file defines two configure presets:

| Preset          | Generator              | Notes                                         |
|-----------------|------------------------|-----------------------------------------------|
| `vs2026`        | `Visual Studio 18 2026` | x64, `CMAKE_CUDA_ARCHITECTURES=75`, output to `build/`. The IDE-friendly one. |
| `ninja-release` | `Ninja`                | Command-line build into `build-ninja/`. **Must** be run from a VS Developer prompt (so `cl.exe` is on `PATH`). |

There is a matching build preset and a test preset (`vs2026-release`) that runs
`ctest` with `outputOnFailure` on. From the command line you can drive presets
directly:

```powershell
cmake --preset vs2026
cmake --build --preset vs2026-release
ctest --preset vs2026-release
```

The `ninja-release` preset is the fastest incremental build, but note its
caveat: Ninja does **not** set up the MSVC environment for you, so you must
launch it from a *Developer PowerShell for VS* (or after dot-sourcing
`scripts\vcenv.ps1`). Otherwise `nvcc` won't find `cl.exe` — the same trap as
everywhere else.

---

## 3. The hand-written Visual Studio solution

> Files: [`vs/PiDigits.sln`](../vs/PiDigits.sln), [`vs/pi.vcxproj`](../vs/pi.vcxproj), `vs/pi_demo.vcxproj`

If you want a `.sln` checked into the repo without running CMake, open
`vs/PiDigits.sln` directly. It contains **two** projects:

- **`pi`** → `build\vs\<Config>\pi.exe` (the CLI; from `src/main.cpp` + the library sources)
- **`pi_demo`** → the narrated demo

Both build Debug|x64 and Release|x64. (The self-test is **not** in this
solution — for `pi_selftest` use the nvcc script or CMake.)

### How CUDA plugs into MSBuild

The header comment in `pi.vcxproj` lays out the three-part dance that makes
`nvcc` a first-class citizen inside MSBuild, and the project body follows it
exactly:

1. **Import `CUDA 13.3.props`** in the `ExtensionSettings` group. This teaches
   MSBuild about the `CudaCompile` item type and where `nvcc` is:

   ```xml
   <ImportGroup Label="ExtensionSettings">
     <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.3.props" />
   </ImportGroup>
   ```

2. **Classify each source.** `.cu` files are listed as `CudaCompile` (built by
   `nvcc`); `.cpp` files are ordinary `ClCompile` (built by `cl.exe`):

   ```xml
   <ItemGroup>   <!-- host, cl.exe -->
     <ClCompile Include="..\src\bignum.cpp" /> ... <ClCompile Include="..\src\main.cpp" />
   </ItemGroup>
   <ItemGroup>   <!-- device, nvcc -->
     <CudaCompile Include="..\src\ntt_cuda.cu" />
     <CudaCompile Include="..\src\bbp_cuda.cu" />
   </ItemGroup>
   ```

3. **Import `CUDA 13.3.targets`** in the `ExtensionTargets` group, which **must
   be last** in the file — it wires up the actual build steps:

   ```xml
   <ImportGroup Label="ExtensionTargets">
     <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.3.targets" />
   </ImportGroup>
   ```

These `.props`/`.targets` files were dropped into VS's
`v180\BuildCustomizations` folder by the **CUDA 13.3 installer's "Visual Studio
Integration" component**, so `$(VCTargetsPath)` resolves them automatically. If
that component was skipped you get the `CUDA 13.3.props was not found` error
(see [Troubleshooting](#troubleshooting)).

### Toolset, architecture, and the C++17 note

The project pins the VS 2026 toolset in both configurations:

```xml
<PlatformToolset>v145</PlatformToolset>
```

`v145` is the toolset that ships with Visual Studio 2026 (v18). The GPU target is
set in the `CudaCompile` item-definition for each config:

```xml
<CodeGeneration>compute_75,sm_75</CodeGeneration>
```

The most easily-overlooked line is the **`-std=c++17` on `CudaCompile`**:

```xml
<CudaCompile>
  <CodeGeneration>compute_75,sm_75</CodeGeneration>
  <!-- Tell nvcc to compile device/host code as C++17 (pi_reference.h uses
       C++17 inline variables). ClCompile's LanguageStandard does not reach nvcc. -->
  <AdditionalOptions>-std=c++17 %(AdditionalOptions)</AdditionalOptions>
</CudaCompile>
```

This matters because the `<LanguageStandard>stdcpp17</LanguageStandard>` set on
`ClCompile` configures **`cl.exe` only** — it does **not** propagate to `nvcc`.
Without explicitly passing `-std=c++17` to `CudaCompile`, the `.cu` files would
compile with `nvcc`'s default standard, and the C++17 inline variables in
`pi_reference.h` (the trusted reference-digit tables) would fail to compile.
This is the single most common mistake when hand-authoring CUDA `.vcxproj`
files.

Outputs land under `build\vs\<Config>\` (kept out of the repo) via the
`OutDir`/`IntDir` overrides in the project.

---

## Running `pi` — full CLI reference

> Source of truth: [`src/main.cpp`](../src/main.cpp)

The `pi` executable is a tiny, dependency-free command-line front-end. It exposes
**two** algorithms behind one interface. Here is the complete option table,
transcribed from `print_usage()` and the parser in `main()`:

### Common options

| Option              | Argument                                          | Default | Meaning |
|---------------------|---------------------------------------------------|---------|---------|
| `--algo`            | `chudnovsky` \| `bbp`                             | `chudnovsky` | Which algorithm to run. |
| `--backend`         | `auto` \| `schoolbook` \| `karatsuba` \| `ntt-cpu` \| `ntt-cuda` | `auto` | Big-integer multiply backend used by Chudnovsky. `auto` picks the fastest by operand size (and prefers CUDA). |
| `--out`             | `<file>`                                          | stdout  | Write the digits to a file instead of standard out. |
| `--verify`          | (flag)                                            | off     | Check the result against the built-in reference digits. |
| `--quiet`           | (flag)                                            | off     | Print only the digits — no banners, no timing. |
| `--help`, `-h`      | (flag)                                            | —       | Show usage and exit. |

### Chudnovsky options (decimal digits)

| Option       | Argument | Default | Meaning |
|--------------|----------|---------|---------|
| `--digits`   | `<N>`    | `1000`  | Number of digits **after** the decimal point. |

### BBP options (hexadecimal digit extraction)

| Option          | Argument        | Default | Meaning |
|-----------------|-----------------|---------|---------|
| `--bbp-start`   | `<P>`           | `1`     | 1-based hex position to start at. |
| `--bbp-count`   | `<C>`           | `32`    | How many hex digits to extract. |
| `--bbp-device`  | `cpu` \| `gpu`  | `gpu`   | Where to run BBP (falls back to CPU if no GPU is available). |

### Worked examples

These are the examples printed by `--help`, plus a couple more to show the
backend knob:

```powershell
# 10,000 decimal digits, verified against the reference table:
build\pi.exe --digits 10000 --verify

# One million digits with the GPU NTT multiply, written to a file:
build\pi.exe --backend ntt-cuda --digits 1000000 --out pi.txt

# Extract 16 hex digits of pi starting at hex position 1,000,000, on the GPU:
build\pi.exe --algo bbp --bbp-start 1000000 --bbp-count 16 --bbp-device gpu

# Compare multiply backends on the same problem (watch the timing line):
build\pi.exe --digits 50000 --backend schoolbook
build\pi.exe --digits 50000 --backend ntt-cuda
```

### What you'll see

For Chudnovsky, with banners on, `main()` prints the digit count, the resolved
backend, the number of series **terms**, and the wall-clock **time** before the
digits themselves:

```
Chudnovsky series + binary splitting
  digits : 10000
  backend: auto (GPU available)
  terms  : 710
  time   : 0.042 s
  verify : PASS (checked 10000 digits against reference)
3.1415926535...
```

The backend line is computed from `g_mul.backend`; when it is `Auto`, the text
also reports whether a GPU was found via `cuda_available()`. `--verify` compares
the fractional digits against `PI_REF_DEC` and, on a mismatch, prints the
**first** fractional position that differs — handy when debugging a backend.

For BBP, the verify path only fires when the requested range lies within
positions `1..64` (the span covered by the `PI_REF_HEX` prefix); outside that
range there is no built-in reference to check against, so `--verify` is silently
a no-op for those positions.

> **Exit codes.** `0` on success. `2` for a usage error — unknown option,
> unknown `--algo`, unknown `--backend`, or a flag missing its value. There is no
> exit-code-1 path in `main`.

For the algorithms behind these options, see
[Chudnovsky binary splitting](04_chudnovsky.md) and [BBP](05_bbp.md); for the
`--backend` choices see [multiplication backends](01_bignum.md) and
[the NTT](03_ntt.md).

---

## Running the demo

> File: [`demo/run_demo.ps1`](../demo/run_demo.ps1)

The demo is the fastest way to see everything working end to end. The runner is
self-bootstrapping — if `build\pi_demo.exe` doesn't exist yet, it builds the
whole project with the nvcc script first, then launches the demo:

```powershell
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
```

The logic is short and worth knowing, because it shows how the build paths
compose — the demo runner *delegates* to `scripts\build.ps1`:

```powershell
$demoExe = Join-Path $root "build\pi_demo.exe"
if (-not (Test-Path $demoExe)) {
    Write-Host "[run_demo] pi_demo.exe not found; building first..." -ForegroundColor Yellow
    & "$root\scripts\build.ps1"
}
& $demoExe
```

So a clean checkout needs only one command. If you've already built (via any
path) and `build\pi_demo.exe` exists, it skips straight to running.

---

## Running the self-test

> File: `tests/self_test.cu` → `pi_selftest.exe`

The self-test is the project's correctness gate. There are two ways to run it.

### Via `ctest` (CMake builds only)

`CMakeLists.txt` registers it with `add_test(NAME selftest COMMAND pi_selftest)`,
so after a CMake build:

```powershell
ctest --test-dir build -C Release
# or, with the preset:
ctest --preset vs2026-release        # prints output on failure
```

`ctest` is the right choice in CI: it reports pass/fail as a process exit code
and (with the preset) dumps the test's output when something breaks.

### Directly (any build)

The nvcc scripts and a direct compile produce a plain executable you can just
run:

```powershell
build\pi_selftest.exe
```

This works no matter how you built — it does not depend on CMake or `ctest`.

---

## Troubleshooting

### `cl.exe not found` (or `nvcc` fails compiling host code)

This is the canonical Windows CUDA failure: `nvcc` could not locate the MSVC
host compiler. Fixes, in order of preference:

- **Using the nvcc scripts?** Make sure you're invoking `build.ps1`/`build.bat`
  rather than calling `nvcc` yourself — they dot-source
  [`scripts/vcenv.ps1`](../scripts/vcenv.ps1) to import the environment. If
  `vcenv.ps1` itself throws `vswhere.exe not found`, Visual Studio isn't
  installed (or not where expected).
- **Building with Ninja or calling `nvcc` by hand?** Run from a *Developer
  PowerShell for VS 2026*, or dot-source the helper first:
  ```powershell
  . .\scripts\vcenv.ps1
  ```
  After it prints `[vcenv] Imported MSVC x64 environment from: ...`, `cl.exe`,
  `INCLUDE`, and `LIB` are all set for that session.
- **Building inside Visual Studio?** The IDE already provides the developer
  environment, so this error there usually means a *different* problem — read on.

### `CUDA 13.3.props was not found`

Seen when building the hand-written `vs/PiDigits.sln` (or any project that
imports `CUDA 13.3.props` / `CUDA 13.3.targets`). It means MSBuild can't find the
CUDA build-customization files under `$(VCTargetsPath)\BuildCustomizations\`.

- **Cause:** the CUDA Toolkit's **"Visual Studio Integration"** component was
  not installed (or was installed before VS, so it couldn't drop the files into
  VS's `v180\BuildCustomizations` folder).
- **Fix:** re-run the **CUDA 13.3 installer** and select the *Visual Studio
  Integration* component. This is exactly what the comment at the top of
  `pi.vcxproj` instructs.
- **Different CUDA version?** The filename encodes the version. If you have, say,
  CUDA 12.6 installed, the files are `CUDA 12.6.props` / `CUDA 12.6.targets`, and
  the two `<Import>` lines in `pi.vcxproj` must be edited to match. CMake avoids
  this entirely by detecting the version for you — prefer it if you're not on
  13.3.

### Targeting a different GPU

The default everywhere is Turing (`sm_75` / `75`) for the RTX 2080 SUPER. To
build for another GPU, set the architecture for whichever path you're using:

| Build path        | How to change architecture                                    |
|-------------------|---------------------------------------------------------------|
| nvcc script       | `build.ps1 -Arch sm_89` (NVCC syntax, `sm_` prefix)           |
| CMake (direct)    | `cmake ... -DCMAKE_CUDA_ARCHITECTURES=89` (bare number)       |
| CMake (`gen_vs`)  | `gen_vs.ps1 -Arch 89`                                          |
| CMake preset      | edit `CMAKE_CUDA_ARCHITECTURES` in [`CMakePresets.json`](../CMakePresets.json) |
| hand-written `.vcxproj` | edit `<CodeGeneration>compute_75,sm_75</CodeGeneration>` (both Debug and Release blocks) |

Common values: `75` Turing (RTX 20-series), `86` Ampere (RTX 30-series), `89`
Ada (RTX 40-series). CMake 3.24+ also accepts `-DCMAKE_CUDA_ARCHITECTURES=native`
to auto-detect the GPU in the machine doing the build — convenient, but it bakes
in *that* machine's architecture, so don't use it for a binary you'll ship
elsewhere.

> Mind the spelling difference once more: the nvcc scripts use `sm_75`; CMake and
> the rest use the bare `75`. Passing `sm_75` to `CMAKE_CUDA_ARCHITECTURES` will
> error.

### Note on `-allow-unsupported-compiler`

You may have seen advice to pass `nvcc -allow-unsupported-compiler` (or
`-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler`) to force CUDA to accept a
newer MSVC than it officially supports. **This project deliberately does not use
it**, and on this stack you do not need it: **CUDA 13.3 officially supports the
Visual Studio 2026 / toolset v145 host compiler**, so the supported-compiler
check passes cleanly.

That flag is a foot-gun: it tells `nvcc` to proceed past a compatibility check
that exists for a reason. If you find yourself reaching for it, the right move is
almost always to install a CUDA Toolkit version that officially supports your
MSVC, or vice versa — not to silence the check. If you are pinned to a mismatched
pair and accept the risk, prefer adding it narrowly (e.g. as a `CudaCompile`
`AdditionalOptions` entry) rather than baking it into the scripts.

---

## Honest limitations

A few caveats that affect what you'll observe when running the programs — stated
plainly so nothing surprises you:

- **BBP relies on `double` precision.** The hex-extraction path accumulates in
  IEEE-754 `double`. This is fast and correct for moderate positions, but the
  reliability of the *last* extracted hex digits degrades as `--bbp-start` grows
  very large, because the available mantissa bits run out. This is why
  `--use_fast_math` is forbidden and why `--verify` only covers positions 1–64.
  See [BBP](05_bbp.md) for the precision analysis.
- **Division and decimal conversion are still quadratic.** The multiplication is
  asymptotically fast (NTT), but the final big-integer **division** and the
  **binary→decimal conversion** in the Chudnovsky path are not yet
  sub-quadratic. For very large `--digits` they come to dominate the runtime, so
  the speedup from `--backend ntt-cuda` is real but not the whole story. See
  [bignum division](01_bignum.md).
- **x64 only.** Every build path hard-targets the `x64` platform. There is no
  32-bit or ARM configuration.

---

## See also

- [01 — Big-integer core and multiply backends](01_bignum.md)
- [03 — The number-theoretic transform (NTT)](03_ntt.md)
- [04 — Chudnovsky via binary splitting](04_chudnovsky.md)
- [05 — BBP hexadecimal digit extraction](05_bbp.md)
- Source: [`src/main.cpp`](../src/main.cpp), [`CMakeLists.txt`](../CMakeLists.txt), [`scripts/build.ps1`](../scripts/build.ps1), [`scripts/vcenv.ps1`](../scripts/vcenv.ps1), [`scripts/gen_vs.ps1`](../scripts/gen_vs.ps1), [`vs/pi.vcxproj`](../vs/pi.vcxproj)
