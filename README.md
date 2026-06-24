# π to billions of digits with CUDA C++ — Chudnovsky + BBP

A heavily-commented, didactic implementation of two classic ways to compute the
digits of **π (pi)**, accelerated on the GPU with CUDA C++:

| Algorithm | What it computes | The interesting CUDA part |
|-----------|------------------|---------------------------|
| **Chudnovsky series** (with binary splitting) | the first *N* **decimal** digits | a **Number Theoretic Transform** big-integer multiply, run on the GPU |
| **Bailey–Borwein–Plouffe (BBP)** | any **hexadecimal** digit, directly, *without the preceding ones* | **one GPU thread per digit position** (embarrassingly parallel) |

This repository is built as **study material for CUDA C++ and the mathematics of
high-precision arithmetic**. Almost every line is commented to explain *what* it
does, *why* it is done that way, and *how* it ties into the whole. If you want to
learn how π is actually computed to millions of digits — and how a GPU makes the
bottleneck (huge multiplications) fast — start at [`docs/00_overview.md`](docs/00_overview.md).

> **Honesty up front.** "Billions of digits" is the *algorithmic* ceiling, not
> what an 8 GB desktop GPU will do in practice. On the RTX 2080 SUPER this was
> developed on, it comfortably computes **millions** of digits. Reaching billions
> needs tens of GB of memory and the sub-quadratic division / out-of-core
> techniques described honestly in [`docs/08_scaling_to_billions.md`](docs/08_scaling_to_billions.md).
> Everything here is **verified correct** against independent reference digits.

---

## Why this exists

Computing π to high precision is a beautiful tour through systems and numerical
programming:

- **Arbitrary-precision integers** — numbers millions of bits long, with their own
  memory layout, carry propagation, and multiplication algorithms.
- **Fast multiplication** — the whole performance story. Schoolbook is O(n²);
  the **Number Theoretic Transform (NTT)** is O(n log n) and maps beautifully onto
  a GPU. This project lives under a directory literally named `CUDA_MATH_FAST_MUL`
  for a reason.
- **Finite-field arithmetic** — the NTT runs in the **Goldilocks field**
  GF(2⁶⁴ − 2³² + 1), whose special structure gives a division-free modular
  reduction that is ideal for GPUs.
- **Two complementary algorithms** — Chudnovsky (best for *all* digits) and BBP
  (the only way to get a *single far-off* digit cheaply), each illustrating a
  different CUDA parallelism pattern.

## Results on the development machine

Measured on an **NVIDIA GeForce RTX 2080 SUPER** (Turing, `sm_75`, 8 GB), CUDA
13.3, Visual Studio 2026. Reproduce them with `pi_demo` (see below).

**Multiplying two ~578,000-digit integers** (60,000 × 32-bit limbs):

| Backend | Time | Complexity |
|---------|------|-----------|
| schoolbook | ~2,900 ms | O(n²) |
| Karatsuba | ~325 ms | O(n¹·⁵⁸⁵) |
| NTT (CPU) | ~73 ms | O(n log n) |
| **NTT (CUDA)** | **~70 ms** | O(n log n) |

For larger operands the GPU pulls far ahead — multiplying two ~1.9-million-digit
numbers takes **~20 ms on the GPU vs ~420 ms on the CPU NTT (~21×)**.

**Chudnovsky:** 100,000 verified decimal digits in **~2.2 s** (all backends auto-selected).

**BBP:** the first 4,096 hex digits — **GPU 61 ms vs CPU 3,612 ms (~59×)**; and the
single hex digit at position **1,000,000** computed directly (no preceding digits)
in well under a second, matching Bailey's published value `26C65E52CB…`.

**Correctness:** the self-test runs **83,000+ checks** — field algebra, NTT (CPU
& CUDA) vs schoolbook, division/sqrt/decimal invariants, Chudnovsky vs an
independent reference, and BBP vs known digits — all passing.

---

## Quick start

You need the **CUDA Toolkit** (developed against 13.3), a host C++ compiler
(**Visual Studio 2026 / MSVC** on Windows), and an NVIDIA GPU. The fast,
dependency-light path is the bundled nvcc build script:

```powershell
# From the repository root, in PowerShell:
powershell -ExecutionPolicy Bypass -File scripts\build.ps1

# Compute and verify 1,000 digits:
build\pi.exe --digits 1000 --verify

# Run the narrated guided tour (shows everything):
build\pi.exe --help
.\build\pi_demo.exe

# Run the correctness suite:
build\pi_selftest.exe
```

Prefer **Visual Studio**? Two options, both documented in
[`docs/07_build_and_run.md`](docs/07_build_and_run.md):

```powershell
# (a) Let CMake generate a VS 2026 solution (builds pi, pi_demo, pi_selftest):
powershell -ExecutionPolicy Bypass -File scripts\gen_vs.ps1
#   then open build\PiDigits.sln, or:  cmake --build build --config Release

# (b) Use the checked-in hand-written solution:
#   open vs\PiDigits.sln in Visual Studio 2026 and press F5.
```

> The build scripts import the Visual Studio environment automatically (via
> `scripts\vcenv.ps1`) so `nvcc` can find `cl.exe`. No `-allow-unsupported-compiler`
> is needed: CUDA 13.3 officially supports the VS 2026 toolchain.

### Command-line examples

```powershell
pi --digits 10000 --verify                       # 10k decimal digits, checked vs reference
pi --backend ntt-cuda --digits 1000000 --out pi.txt   # 1M digits to a file, GPU multiply
pi --backend schoolbook --digits 2000            # force the slow O(n^2) multiply (for comparison)
pi --algo bbp --bbp-start 1000000 --bbp-count 16 # hex digits 1,000,000.. directly
pi --algo bbp --bbp-start 1 --bbp-count 64 --verify   # verify the first 64 hex digits
```

---

## Repository layout

```
PI_DIGITS/
├─ include/                 # heavily-commented headers (the public APIs)
│  ├─ bignum.h              #   arbitrary-precision integer type + multiply backends
│  ├─ goldilocks.h          #   GF(2^64-2^32+1) field arithmetic (host + device)
│  ├─ ntt.h                 #   shared NTT helpers (repack / carry / sizing)
│  ├─ chudnovsky.h          #   Chudnovsky + binary splitting interface
│  ├─ bbp.h / bbp_kernel.h  #   BBP interface + the shared host/device math
│  └─ pi_reference.h        #   trusted reference digits (golden test vector)
├─ src/
│  ├─ bignum.cpp            #   add/sub/shift/schoolbook/Karatsuba/dispatch
│  ├─ bignum_div.cpp        #   Knuth division, integer sqrt, decimal output
│  ├─ ntt_cpu.cpp           #   CPU Number Theoretic Transform multiply
│  ├─ ntt_cuda.cu           #   GPU NTT multiply (the fast-multiply accelerator)
│  ├─ chudnovsky.cpp        #   binary splitting + final assembly
│  ├─ bbp_cpu.cpp / bbp_cuda.cu  #  BBP digit extraction (CPU ref + GPU kernel)
│  └─ main.cpp              #   the `pi` command-line program
├─ tests/self_test.cu       # the correctness suite (pi_selftest)
├─ demo/demo.cu             # the narrated guided tour (pi_demo)
├─ docs/                    # the deep-dive didactic documentation (start at 00_overview)
│  └─ changelog/            # one Markdown note per push (see below)
├─ vs/                      # hand-written Visual Studio 2026 solution
├─ scripts/                 # build.ps1/.bat, vcenv.ps1, gen_vs.ps1, new_changelog.ps1
├─ CMakeLists.txt           # CMake build (generates the VS solution too)
└─ CMakePresets.json        # "Open Folder" support for Visual Studio
```

## The documentation

The `docs/` folder is the heart of the study material — read it in order:

0. [Overview & architecture](docs/00_overview.md)
1. [The Chudnovsky series](docs/01_chudnovsky.md)
2. [Binary splitting](docs/02_binary_splitting.md)
3. [The BBP formula & digit extraction](docs/03_bbp.md)
4. [The big-integer library](docs/04_bignum.md)
5. [The NTT & the Goldilocks field](docs/05_ntt_goldilocks.md)
6. [CUDA design & engineering](docs/06_cuda_design.md)
7. [Building & running](docs/07_build_and_run.md)
8. [Scaling toward billions (honestly)](docs/08_scaling_to_billions.md)

## How correctness is established

This is study material, so it must be *trustworthy*:

- The fast paths are checked against the slow, obviously-correct ones: **NTT
  (CPU and CUDA) vs schoolbook**, **Karatsuba vs schoolbook**, **Newton isqrt vs a
  bit-by-bit reference**, **Knuth division vs the `q·b + r == a, 0 ≤ r < b`
  invariant**.
- The **Goldilocks field** is validated with algebraic identities (distributivity,
  inverses, Fermat) and root-of-unity orders.
- **Chudnovsky** output is compared against an independent π computed on this
  machine with Machin's formula (whose first 1,000 digits match the SHA-256 of
  published tables), embedded in [`include/pi_reference.h`](include/pi_reference.h).
- **BBP** output is compared against the known hex prefix and Bailey's
  position-1,000,000 checkpoint.

Run `pi_selftest` to re-verify everything on your machine.

## A note on the changelog convention

Every time something new is pushed to GitHub, a short Markdown file is added under
[`docs/changelog/`](docs/changelog/) explaining **what was added and why**. The
helper `scripts\new_changelog.ps1 -Title "..."` scaffolds the next entry. The very
first entry, describing the initial import, is
[`0001-initial-import.md`](docs/changelog/0001-initial-import.md).

## Developed against

- **GPU:** NVIDIA GeForce RTX 2080 SUPER (Turing, compute capability 7.5, `sm_75`, 8 GB)
- **CUDA Toolkit:** 13.3
- **Host compiler:** Visual Studio 2026 (toolset `v145`, MSVC 14.51)
- **OS:** Windows 11

For other GPUs, pass the right architecture (e.g. `scripts\build.ps1 -Arch sm_89`
or `-DCMAKE_CUDA_ARCHITECTURES=89`).

## License

[MIT](LICENSE).
