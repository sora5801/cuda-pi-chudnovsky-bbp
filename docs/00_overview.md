# 00 -- Project Overview and Map

Welcome. This document is the front door to the project: what it computes, the
two algorithms it uses, why a GPU shows up at all, and where every piece of code
lives. If you read only one doc before diving into the source, read this one --
then follow the cross-links into the deeper notes.

The project is a **multi-digit pi calculator** written in C++17 and CUDA. It
computes the digits of pi two completely different ways, and it exists mainly as
a study vehicle for one idea: **fast big-integer multiplication, accelerated on
the GPU.** That is why the repository lives under a directory named
`CUDA_MATH_FAST_MUL`.

---

## 1. What the project does

It computes pi to (potentially) very many digits, by two independent methods you
select on the command line:

1. **Chudnovsky series + binary splitting** -- produces the first *N* **decimal**
   digits (`3.14159...`). This is the workhorse used for real pi records into
   the trillions of digits. See [src/chudnovsky.cpp](../src/chudnovsky.cpp) and
   [include/chudnovsky.h](../include/chudnovsky.h).

2. **BBP (Bailey-Borwein-Plouffe) formula** -- extracts a block of **hexadecimal**
   digits starting at an arbitrary position, *without* computing the digits that
   come before it. See [src/bbp_cpu.cpp](../src/bbp_cpu.cpp),
   [src/bbp_cuda.cu](../src/bbp_cuda.cu) and the shared math in
   [include/bbp_kernel.h](../include/bbp_kernel.h).

The front-end that ties them together is [src/main.cpp](../src/main.cpp), built
into an executable called `pi`. A self-test suite (`pi_selftest`) proves the
fast paths agree with slow reference paths and that the digits match a trusted
table, and a narrated demo (`pi_demo`) walks through the layers.

A first taste of the CLI (`pi --help` prints the full menu, defined in
`print_usage()` inside [src/main.cpp](../src/main.cpp)):

```text
pi --digits 10000 --verify
pi --backend ntt-cuda --digits 1000000 --out pi.txt
pi --algo bbp --bbp-start 1000000 --bbp-count 16 --bbp-device gpu
```

---

## 2. The two algorithms at a glance

The two algorithms answer *different questions* and have *different shapes*. It
is worth internalizing the contrast, because it explains the whole architecture.

| Aspect | Chudnovsky (decimal) | BBP (hexadecimal) |
| --- | --- | --- |
| Question it answers | "Give me the first N digits." | "Give me the digit at position P." |
| Base of the output | base 10 (decimal) | base 16 (hexadecimal) |
| Must it compute earlier digits? | **Yes** -- to know digit N you compute all of them. | **No** -- digit P is computed directly. |
| Core arithmetic | exact big-integer math (millions of bits) | tiny per-digit math in 64-bit ints + `double` |
| Memory | O(N) -- holds the whole number | **O(1)** per digit |
| Parallelism | inside one big multiply (the NTT) | **embarrassingly parallel** across positions |
| Where the GPU helps | the giant multiplications near the splitting tree's root | one GPU thread per output digit |
| Reliability | exact; verified to the last digit | `double`-precision; reliable for the *leading* hex digit only (see caveat) |

### 2a. Chudnovsky in one breath

The Chudnovsky brothers' 1988 series adds about **14.18 correct decimal digits
per term**:

```
1/pi = 12 * sum_{k>=0} (-1)^k (6k)! (13591409 + 545140134 k)
                       -----------------------------------------
                       (3k)! (k!)^3 640320^(3k + 3/2)
```

Rather than sum term by term (which would force full precision the whole way),
the code uses **binary splitting**: it keeps an exact rational `T / Q` (with a
helper product `P`) for a range of terms and merges ranges in a balanced binary
tree. The merge rule, straight from the header comment in
[include/chudnovsky.h](../include/chudnovsky.h), is:

```
P = P_L * P_R
Q = Q_L * Q_R
T = T_L * Q_R + P_L * T_R
```

The crucial consequence: **the only truly enormous multiplications happen near
the root of the tree**, where both operands have nearly all the digits. Those
are exactly the multiplications the fast NTT/CUDA backend accelerates. After the
tree is built, a final assembly evaluates

```
pi = 426880 * sqrt(10005) * Q / T
```

in fixed-point integer arithmetic. See [01_chudnovsky.md](01_chudnovsky.md) and
[02_binary_splitting.md](02_binary_splitting.md) for the full derivation.

### 2b. BBP in one breath

The BBP formula (1995) is almost magic:

```
pi = sum_{k>=0} (1/16^k) ( 4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6) )
```

Because of the `1/16^k`, you can isolate the hex digit at position `d` by
computing the *fractional part* of `16^d * pi` using modular exponentiation for
the "head" terms (`k <= d`, done exactly with `bbp_modpow16`) and a short
floating-point tail. Each position is independent of every other, which is why
the GPU path (`bbp_hex_digits_cuda`) launches **one thread per position**. The
math lives entirely in [include/bbp_kernel.h](../include/bbp_kernel.h), marked
`BBP_HD` so the identical code compiles for CPU and GPU. See
[03_bbp.md](03_bbp.md).

> **Honest caveat.** The BBP head/tail sums accumulate in IEEE-754 `double`
> (53-bit mantissa). That is enough for roughly the first ~10 hex digits of
> `{16^d pi}` to be correct, and since each thread extracts only the *leading*
> hex digit it has ~10 digits of slack. It is reliable across the range we use
> and the self-test confirms it, but extracting *many* correct digits at a single
> position would require extended precision. This limitation is documented
> candidly in the header comment of [include/bbp_kernel.h](../include/bbp_kernel.h).

---

## 3. The role of CUDA fast multiplication

Everything performance-critical in the Chudnovsky path reduces to one operation:
**multiply two integers that are millions of bits long.** A 64-bit
`unsigned long long` holds ~19 decimal digits; a million-digit number needs
~52,000 32-bit limbs. So the project builds its own big-integer type, `BigInt`
in [include/bignum.h](../include/bignum.h), storing the magnitude as a
little-endian `std::vector<uint32_t>` (`mag[0]` is the least-significant limb,
base `B = 2^32`).

Multiplication is offered at four levels, and the project ships *all of them* so
the fast ones can be checked against the slow ones:

| Backend (`MulBackend`) | Complexity | Role |
| --- | --- | --- |
| `Schoolbook` | O(n^2) | the definition of multiplication; ground-truth reference |
| `Karatsuba` | O(n^1.585) | classic divide-and-conquer; the sub-quadratic middle ground |
| `NttCpu` | O(n log n) | Number Theoretic Transform on CPU threads |
| `NttCuda` | O(n log n) | the *same* NTT, with the transforms running on the GPU |

The NTT is an FFT done with **integers modulo a prime** instead of complex
floats, so every intermediate value is exact -- essential when you want millions
of *correct* digits. The prime is the famous **Goldilocks prime**
`p = 2^64 - 2^32 + 1` (`GL_P` in [include/goldilocks.h](../include/goldilocks.h)),
chosen because a field element fits in one `uint64_t`, transform lengths up to
`2^32` are available, and its special shape allows a division-free "Solinas"
modular reduction -- a big win on a GPU, where 64-bit integer division is
emulated and slow. Packing the big integer into 16-bit digits keeps every
convolution coefficient below `p`, so a **single prime recovers the exact
product with no Chinese-Remainder step**. The full story is in
[05_ntt_goldilocks.md](05_ntt_goldilocks.md) and [06_cuda_ntt.md](06_cuda_ntt.md).

A run-time dispatcher, `mul_dispatch()` in [src/bignum.cpp](../src/bignum.cpp),
chooses a backend automatically by operand size in `Auto` mode (thresholds
`karatsuba_threshold = 32` and `ntt_threshold = 256` limbs, from the `MulConfig`
struct), or honors whatever the user forced via `--backend`. When an NTT is
chosen and `prefer_cuda` is set, it uses the GPU **if `cuda_available()` returns
true**; otherwise it falls back to the CPU NTT. If the project was built without
CUDA, [src/ntt_cuda.cu](../src/ntt_cuda.cu)'s role is filled by a stub that
forwards to the CPU path, so the code always works.

---

## 4. Repository layout -- a guided tour

```
PI_DIGITS/
├── CMakeLists.txt        # the recommended cross-platform build
├── CMakePresets.json     # presets so "Open Folder" in VS configures itself
├── LICENSE  .gitignore
├── include/              # public headers (the API + shared host/device math)
├── src/                  # the implementation (.cpp host, .cu host+device)
├── tests/                # the correctness self-test (built as pi_selftest)
├── demo/                 # a narrated guided tour (built as pi_demo)
├── docs/                 # these documents (+ docs/changelog/)
├── vs/                   # hand-written Visual Studio 2026 .sln/.vcxproj
├── scripts/              # one-command build helpers (.bat / .ps1)
├── build/                # CMake/nvcc build output (generated; gitignored)
└── build_vs/             # a CMake build tree for VS (generated; gitignored)
```

### include/ -- the headers

| File | Purpose |
| --- | --- |
| [include/bignum.h](../include/bignum.h) | The `BigInt` type, the `MulBackend` enum and global `g_mul` config, and declarations for every multiply backend, `mul_dispatch`, division (`divmod_knuth`, `div_fast`), `isqrt`, `pow10`, and decimal conversion. |
| [include/chudnovsky.h](../include/chudnovsky.h) | The `PQT` triple, `chudnovsky_bs(a,b)` binary splitting, `compute_pi_chudnovsky(digits)`, and `chudnovsky_terms_for_digits`. |
| [include/bbp.h](../include/bbp.h) | Host-facing BBP entry points: `bbp_hex_digit`, `bbp_hex_digits_cpu`, `bbp_hex_digits_cuda`. |
| [include/bbp_kernel.h](../include/bbp_kernel.h) | The actual BBP math (`bbp_mulmod`, `bbp_modpow16`, `bbp_series`, `bbp_digit_value`), written once with `BBP_HD` to run on CPU and GPU. |
| [include/goldilocks.h](../include/goldilocks.h) | Finite-field arithmetic in GF(p) with the Goldilocks prime: `gl_add`, `gl_sub`, `gl_mul`, `gl_reduce128`, `gl_pow`, `gl_inv`, `gl_root_of_unity`. All `GL_HD` (host+device). |
| [include/ntt.h](../include/ntt.h) | Shared NTT helpers used by *both* transform implementations so they agree bit-for-bit: `repack_to16`, `carry_propagate`, `ntt_length`, `ntt_bit_reverse`. |
| [include/pi_reference.h](../include/pi_reference.h) | Trusted "golden" digits: 3500 decimal digits (`PI_REF_DEC`) and 64 hex digits (`PI_REF_HEX`) for `--verify` and the self-test. |

### src/ -- the implementation

| File | Purpose | Compiled by |
| --- | --- | --- |
| [src/main.cpp](../src/main.cpp) | CLI front-end: argument parsing, backend selection, running the chosen algorithm, `--verify`, and output. Builds into `pi`. | host (cl.exe) |
| [src/bignum.cpp](../src/bignum.cpp) | `BigInt` construction/normalization/compare, signed add/sub, `operator*`, `mul_dispatch`, schoolbook & Karatsuba multiply, `mul_small`, `pow10`. | host |
| [src/bignum_div.cpp](../src/bignum_div.cpp) | Division and roots: Knuth Algorithm D (`divmod_knuth`), Newton-based fast division (`div_fast`), integer square root (`isqrt`, `isqrt_bitwise`), decimal conversion (`to_decimal`). | host |
| [src/ntt_cpu.cpp](../src/ntt_cpu.cpp) | The CPU NTT multiply (`mul_ntt_cpu`): forward/inverse butterflies in GF(p). | host |
| [src/ntt_cuda.cu](../src/ntt_cuda.cu) | The CUDA NTT multiply (`mul_ntt_cuda`) and `cuda_available()`: the same transform in GPU kernels. | nvcc |
| [src/chudnovsky.cpp](../src/chudnovsky.cpp) | `chudnovsky_bs` binary splitting and `compute_pi_chudnovsky` final assembly (`pi = 426880 * sqrt(10005) * Q / T`). | host |
| [src/bbp_cpu.cpp](../src/bbp_cpu.cpp) | `bbp_hex_digits_cpu` / `bbp_hex_digit`: loop over positions on the CPU. | host |
| [src/bbp_cuda.cu](../src/bbp_cuda.cu) | `bbp_hex_digits_cuda`: one GPU thread per position, with CPU fallback. | nvcc |

### tests/, demo/, docs/, vs/, scripts/

| Path | Purpose |
| --- | --- |
| [tests/self_test.cu](../tests/self_test.cu) | The full correctness suite (`pi_selftest`). Tests bottom-to-top: Goldilocks algebra, NTT (CPU+CUDA) equals schoolbook, division/`isqrt`/decimal, Chudnovsky vs reference (incl. a GPU-NTT run), and BBP hex vs reference. Registered with CTest via `add_test(NAME selftest ...)`. |
| [demo/demo.cu](../demo/demo.cu) | A narrated guided tour (`pi_demo`) showing each layer in action. `demo/run_demo.ps1` runs it. |
| `docs/` | These study documents. `docs/changelog/` holds per-push changelog entries (see `scripts/new_changelog.ps1`). |
| `vs/` | A hand-written Visual Studio 2026 solution: `PiDigits.sln`, `pi.vcxproj`, `pi_demo.vcxproj`. An alternative to letting CMake generate the solution. |
| `scripts/` | One-command build helpers: `build.bat` and `build.ps1` (compile all three exes with `nvcc`), `gen_vs.ps1`, `vcenv.ps1`, and `new_changelog.ps1`. |
| `CMakeLists.txt` | The recommended build. Defines the static library `pidigits` from the seven library sources, then `pi`, `pi_selftest`, and `pi_demo` linked against it. Defaults to `CMAKE_CUDA_ARCHITECTURES=75` (Turing); override for other GPUs. |

---

## 5. Data-flow: from `main` down to the GPU

The most important path to understand is what happens when you ask for decimal
digits. Here is the call chain, grounded in the actual function names:

```
main()                              [src/main.cpp]
  │  parses --digits, --backend; sets the global g_mul
  ▼
compute_pi_chudnovsky(digits)       [src/chudnovsky.cpp]
  │  N = chudnovsky_terms_for_digits(digits)   (~digits / 14.1816)
  ▼
chudnovsky_bs(0, N)                 [src/chudnovsky.cpp]
  │  recursive balanced binary splitting; returns PQT {P, Q, T}
  │  each merge does  T = T_L*Q_R + P_L*T_R,  P = P_L*P_R,  Q = Q_L*Q_R
  ▼
BigInt operator*(a, b)             [src/bignum.cpp]   ← every "*" above lands here
  │  applies the sign rule, then calls...
  ▼
mul_dispatch(a.mag, b.mag)         [src/bignum.cpp]
  │  Auto mode: < 32 limbs → schoolbook
  │             < 256 limbs → Karatsuba
  │             ≥ 256 limbs → NTT  (CUDA if prefer_cuda && cuda_available(), else CPU)
  ▼
mul_ntt_cuda(a, b)  ─or─  mul_ntt_cpu(a, b)
  [src/ntt_cuda.cu]          [src/ntt_cpu.cpp]
       │  both use the shared helpers in include/ntt.h
       │  (repack_to16 → forward NTT → pointwise gl_mul → inverse NTT → carry_propagate)
       ▼
   GF(p) arithmetic in include/goldilocks.h  (gl_mul, gl_reduce128, ...)
```

After `chudnovsky_bs` returns `{P, Q, T}`, the **final assembly** in
`compute_pi_chudnovsky` does (see [src/chudnovsky.cpp](../src/chudnovsky.cpp)):

1. `isqrt(10005 * 10^(2*prec))` to get `sqrt(10005)` scaled to full precision
   (constants `R_CONST = 10005`, `L_CONST = 426880`).
2. `numerator = 426880 * Q * sqrtC`, then `div_fast(numerator, T)` to get
   `floor(pi * 10^prec)`.
3. `pi_scaled.to_decimal()` to format the digits as `"3.14159..."`.

The BBP path is simpler and bypasses `BigInt` entirely: `main` calls
`bbp_hex_digits_cuda` (or `..._cpu`), which fans out across positions, each
running `bbp_digit_value` from [include/bbp_kernel.h](../include/bbp_kernel.h).

> **Honest caveat on the "still-quadratic" tails.** The multiply is sub-quadratic,
> but a couple of supporting steps are not yet fully optimized. `div_fast` uses
> Newton's method on top of the fast multiply, but falls back to the exact-but
> O(n^2) `divmod_knuth` for small inputs; and the decimal conversion in
> `to_decimal`, while built on a divide-and-conquer scheme, still has
> quadratic-ish behavior at the very largest sizes. For the digit counts in this
> study project these are not the bottleneck, but they are why this is a learning
> tool rather than a record-setting engine. See [04_division_sqrt.md](04_division_sqrt.md).

---

## 6. Where to go next

These siblings drill into each layer (create them as you study; the links are
the intended reading order):

- [01_chudnovsky.md](01_chudnovsky.md) -- the Chudnovsky series, its constants, and the final assembly.
- [02_binary_splitting.md](02_binary_splitting.md) -- why and how the `PQT` tree works.
- [03_bbp.md](03_bbp.md) -- the BBP formula, modular exponentiation, and the precision caveat.
- [04_division_sqrt.md](04_division_sqrt.md) -- Knuth division, Newton `div_fast`, and `isqrt`.
- [05_ntt_goldilocks.md](05_ntt_goldilocks.md) -- the NTT, the Goldilocks prime, and Solinas reduction.
- [06_cuda_ntt.md](06_cuda_ntt.md) -- how the transform maps onto GPU kernels.
- [07_bignum.md](07_bignum.md) -- the `BigInt` memory layout and arithmetic in depth.

---

## 7. Quick start

### Build with CMake (recommended)

```bash
# Configure + build (Windows, Visual Studio 2026)
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release

# Run the self-test (registered with CTest)
ctest --test-dir build -C Release
```

For a different GPU, override the architecture, e.g.
`-DCMAKE_CUDA_ARCHITECTURES=89` (Ada / RTX 40-series) or `=native` to auto-detect.

### Build with one command (nvcc, no CMake)

```bat
scripts\build.bat          :: defaults to sm_75, Release; output in .\build
```

### Run

```bash
# 1000 decimal digits, checked against the reference table
build/pi --digits 1000 --verify

# A million digits using the GPU NTT, written to a file
build/pi --backend ntt-cuda --digits 1000000 --out pi.txt

# 16 hex digits starting at hex position 1,000,000, on the GPU
build/pi --algo bbp --bbp-start 1000000 --bbp-count 16 --bbp-device gpu
```

### Self-test directly

```bash
build/pi_selftest          # exits non-zero if anything is wrong
```

A passing `pi_selftest` proves the GPU-accelerated paths agree with the slow
reference paths and that the digits match the trusted table -- which is the
whole point: **fast, and provably correct.**
