# 0001 - Initial import: Chudnovsky + BBP π calculator in CUDA C++

_Date: 2026-06-24_

## Summary

The first public version of the project: a complete, heavily-commented, and
**verified** implementation that computes the digits of π two ways — the
**Chudnovsky series** (decimal digits, via binary splitting on a from-scratch
big-integer library) and the **Bailey–Borwein–Plouffe formula** (hexadecimal
digit extraction) — with the heavy lifting accelerated on an NVIDIA GPU using
CUDA C++. It builds three ways (nvcc scripts, CMake → Visual Studio 2026, and a
hand-written `.sln`), ships a narrated demo, and passes an 83,000-check self-test.

## What was added

### The math engine
- **`include/bignum.h` + `src/bignum.cpp`** — an arbitrary-precision signed
  integer (`BigInt`) in sign-magnitude form with base-2³² little-endian limbs:
  addition/subtraction with carry/borrow, bit shifts, **schoolbook** O(n²) and
  **Karatsuba** O(n¹·⁵⁸⁵) multiplication, a size-based multiply **dispatcher**,
  decimal parsing, and powers of ten.
- **`src/bignum_div.cpp`** — **Knuth Algorithm D** long division (exact, the
  workhorse), **Newton** integer square root with a bit-by-bit reference, and
  base-10 output by repeated division by 10⁹.
- **`include/goldilocks.h`** — arithmetic in the **Goldilocks field**
  GF(2⁶⁴ − 2³² + 1): the branch-light **Solinas `reduce128`**, modular add/sub/mul/
  pow/inverse, and roots of unity. Written `__host__ __device__` so the *same*
  code runs on CPU and GPU.
- **`include/ntt.h` + `src/ntt_cpu.cpp` + `src/ntt_cuda.cu`** — big-integer
  multiplication by a **Number Theoretic Transform**: 16-bit repacking (so a
  single prime suffices — no CRT), an iterative radix-2 Cooley–Tukey transform,
  pointwise product, inverse transform, and carry propagation. The CUDA version
  runs each transform stage as a kernel, with a precomputed twiddle table and a
  graceful CPU fallback.

### The two π algorithms
- **`include/chudnovsky.h` + `src/chudnovsky.cpp`** — Chudnovsky series with
  **binary splitting** (`P`, `Q`, `T` triples merged up a balanced tree), and the
  fixed-point final assembly `π = 426880·√10005·Q/T`.
- **`include/bbp_kernel.h` + `bbp.h` + `src/bbp_cpu.cpp` + `src/bbp_cuda.cu`** —
  BBP hexadecimal digit extraction via exact modular exponentiation, with the GPU
  version running **one thread per digit position**.

### Front-ends, tests, reference data
- **`src/main.cpp`** — the `pi` command-line tool (choose algorithm, digit count,
  multiply backend, BBP position/count, output file, and `--verify`).
- **`tests/self_test.cu`** — the correctness suite (`pi_selftest`).
- **`demo/demo.cu`** — a narrated guided tour (`pi_demo`).
- **`include/pi_reference.h`** — 3,500 independently-verified decimal digits and
  64 hex digits of π used as golden test vectors.

### Build system
- **`CMakeLists.txt` + `CMakePresets.json`** — CMake build that also generates a
  Visual Studio 2026 solution.
- **`vs/PiDigits.sln` + `vs/pi.vcxproj` + `vs/pi_demo.vcxproj`** — a hand-written
  VS 2026 solution (toolset `v145`, CUDA 13.3 build customization).
- **`scripts/`** — `build.ps1`/`build.bat` (nvcc), `vcenv.ps1` (imports the MSVC
  environment), `gen_vs.ps1` (CMake → VS), and `new_changelog.ps1`.

### Documentation
- **`README.md`** and a nine-part **`docs/`** deep dive (overview, Chudnovsky,
  binary splitting, BBP, big-integer library, NTT/Goldilocks, CUDA design,
  build & run, and an honest scaling analysis).

## Why

The goal is **study material**: a correct, end-to-end, richly-explained example of
high-precision computation and the CUDA patterns that accelerate it. Two
algorithms were chosen because they illustrate two different kinds of GPU
parallelism — a structured transform (NTT) versus an embarrassingly-parallel map
(BBP) — and two different number bases (decimal vs hexadecimal).

## How it ties in

Chudnovsky's binary splitting reduces computing π to a tree of big-integer
multiplications; those multiplications are exactly what the Goldilocks-field NTT
accelerates on the GPU. BBP is independent and serves both as a second algorithm
and as a cross-check on specific digits. The big-integer library underlies
Chudnovsky; the field arithmetic underlies the NTT.

## How to try it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
build\pi.exe --digits 1000 --verify
build\pi_selftest.exe
build\pi_demo.exe
build\pi.exe --algo bbp --bbp-start 1000000 --bbp-count 16
```

## Verified on

NVIDIA GeForce RTX 2080 SUPER (`sm_75`, 8 GB) · CUDA 13.3 · Visual Studio 2026
(toolset `v145`) · Windows 11. Self-test: **83,213 checks passed, 0 failed**.
