# 06 — CUDA Engineering: Mapping the NTT and BBP onto the GPU

This document is about the *engineering* of the two GPU code paths in this
project:

1. [src/ntt_cuda.cu](../src/ntt_cuda.cu) — the Number Theoretic Transform that
   powers our fast big-integer multiply, and
2. [src/bbp_cuda.cu](../src/bbp_cuda.cu) — the Bailey–Borwein–Plouffe digit
   extractor.

The *math* of each lives elsewhere: the NTT in
[05_ntt_goldilocks.md](05_ntt_goldilocks.md) and the field arithmetic in
[include/goldilocks.h](../include/goldilocks.h); the BBP formula in
[03_bbp.md](03_bbp.md) and its kernel in
[include/bbp_kernel.h](../include/bbp_kernel.h). Here we assume you already know
*what* an NTT or BBP series computes, and we focus on *how* it lands on
thousands of CUDA threads — safely, portably, and fast.

The throughline of the whole chapter: **the GPU only ever makes things faster;
it never decides correctness.** Every kernel has a CPU twin, and any CUDA hiccup
silently falls back to it.

---

## Table of contents

- [1. The shape of an iterative NTT and why the GPU likes it](#1-the-shape-of-an-iterative-ntt-and-why-the-gpu-likes-it)
- [2. One kernel launch per stage = the global barrier](#2-one-kernel-launch-per-stage--the-global-barrier)
- [3. The five NTT kernels, line by line](#3-the-five-ntt-kernels-line-by-line)
- [4. Host orchestration and host↔device transfers](#4-host-orchestration-and-hostdevice-transfers)
- [5. Graceful CPU fallback](#5-graceful-cpu-fallback)
- [6. The BBP kernel: embarrassingly parallel](#6-the-bbp-kernel-embarrassingly-parallel)
- [7. Portability: no `__int128` on Windows, and the HD macro trick](#7-portability-no-__int128-on-windows-and-the-hd-macro-trick)
- [8. Occupancy, threads-per-block, and memory footprint](#8-occupancy-threads-per-block-and-memory-footprint)
- [9. Measured speedups — and when the GPU does *not* help](#9-measured-speedups--and-when-the-gpu-does-not-help)
- [10. Honest limitations](#10-honest-limitations)

---

## 1. The shape of an iterative NTT and why the GPU likes it

A radix-2 NTT (the integer-field cousin of the FFT) over `N` points proceeds in
`log2(N)` **stages**. Stage `s` combines elements in blocks of size
`len = 2, 4, 8, …, N`. The atomic unit of work is the **butterfly**: it reads
two values `u` and `v`, multiplies `v` by a *twiddle factor* (a power of a root
of unity), and writes back `u+v` and `u−v`. Each stage performs exactly `N/2`
butterflies.

Two structural facts decide the entire GPU design, and the header comment in
[src/ntt_cuda.cu](../src/ntt_cuda.cu) states them plainly:

> Within a stage all N/2 butterflies are completely independent, so each maps to
> its own GPU thread. BUT stage s+1 reads outputs of stage s, so stages must be
> separated by a global barrier.

So:

- **Inside a stage:** total parallelism. `N/2` butterflies, zero data
  dependencies, perfect for `N/2` GPU threads.
- **Between stages:** a hard dependency. Every butterfly in stage `s+1` may read
  a value any butterfly in stage `s` just wrote.

That second fact is the crux of the whole mapping, and it leads directly to the
"one launch per stage" pattern in the next section.

```
N = 8 example (3 stages). Each '|' pair is one butterfly.

stage len=2:  [0 1][2 3][4 5][6 7]      4 butterflies
stage len=4:  [0 . 2 .][4 . 6 .]        4 butterflies (j=0,1 per block)
stage len=8:  [0 . . . 4 . . .]         4 butterflies (j=0..3)
                  ^ each stage = N/2 = 4 independent butterflies
                  ^ but stage k+1 needs stage k finished first
```

---

## 2. One kernel launch per stage = the global barrier

CUDA gives you a cheap barrier *within a thread block* (`__syncthreads()`), but
there is **no cheap, reliable barrier across all blocks of a grid** in the
general case. (Cooperative groups exist but impose occupancy limits and device
constraints.) The simplest, most portable global barrier is:

> end the kernel and launch the next one.

When a kernel finishes, the CUDA runtime guarantees every write from that launch
is visible to the *next* launch on the same stream. That is exactly the
inter-stage dependency we need. So the host loop in `run_transform()` issues one
`k_butterfly` launch **per stage**:

```cpp
// Stages 1..logn: butterflies for len = 2, 4, ..., n.
uint32_t halfN = n >> 1;
for (uint32_t len = 2; len <= n; len <<= 1) {
    uint32_t stride = n / len;                // index step into the twiddle table
    k_butterfly<<<(halfN + TPB - 1) / TPB, TPB>>>(d_a, n, len, W, stride);
    CUDA_TRY(cudaGetLastError());
}
```

Note the grid sizing: every stage launches `halfN = N/2` threads (rounded up to
a whole number of `TPB`-sized blocks). The number of *useful* butterflies is
always `N/2` regardless of `len`; what changes per stage is only how each thread
*decodes* its index into a block and a twiddle (see `k_butterfly` below). That
uniformity is deliberate — one kernel body, `log2(N)` launches, identical
launch geometry each time.

The cost is real but small: `log2(N)` kernel launches plus the bit-reversal and
(for the inverse) a scale launch. For `N = 2^20` that is ~20 launches per
transform. Launch overhead is on the order of microseconds; the butterfly work
dwarfs it.

---

## 3. The five NTT kernels, line by line

All five device kernels live in [src/ntt_cuda.cu](../src/ntt_cuda.cu). Each one
calls the `gl_*` field operations from
[include/goldilocks.h](../include/goldilocks.h), which compile for the device
because they are marked `__host__ __device__` (see §7).

### 3.1 Bit-reversal — `k_bit_reverse` (using `__brev`)

An iterative NTT requires the input in **bit-reversed order**: the element at
index `i` belongs at index `bitrev(i)`. CUDA has a single-instruction intrinsic
`__brev` that reverses all 32 bits of a word. Since we only want the low `logn`
bits reversed, we shift the reversed value back down:

```cpp
__device__ __forceinline__ uint32_t dev_bitrev(uint32_t i, int logn) {
    return __brev(i) >> (32 - logn);
}
```

The permutation kernel then has each thread own one index `i` and swap with
`j = bitrev(i)` **only when `i < j`**:

```cpp
__global__ void k_bit_reverse(uint64_t* a, uint32_t n, int logn) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t j = dev_bitrev(i, logn);
    if (i < j) { uint64_t tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
}
```

The `i < j` guard is the trick that makes this race-free: `bitrev` is an
involution (`bitrev(bitrev(i)) == i`), so the pair `{i, j}` is visited by exactly
two threads, and only the one with the smaller index performs the swap. No two
threads touch the same slot. No `__syncthreads`, no atomics.

### 3.2 The twiddle table — `k_build_twiddles` (indexed by stride)

A butterfly multiplies by `omega_len^j`, a power of a root of unity. Recomputing
that power inside every butterfly would add an `O(log N)` modular exponentiation
to each of the `N/2 · log N` butterflies. Instead we precompute **once** a table

```
W[t] = omega_N^t,   t in [0, N/2)
```

where `omega_N` is the primitive `N`-th root from `gl_root_of_unity(N)`. The
beauty is that one table serves *every* stage. The twiddle a stage of block size
`len` needs at position `j` is

```
omega_len^j = omega_N^{ j · (N/len) }
```

so a stage just indexes `W` with a **stride** of `N/len`. That is precisely the
`stride = n / len` passed to `k_butterfly`. The table itself is built in
parallel, one entry per thread, via square-and-multiply:

```cpp
__global__ void k_build_twiddles(uint64_t* W, uint32_t half, uint64_t omega) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= half) return;
    W[t] = gl_pow(omega, (uint64_t)t);
}
```

We build **two** tables per multiply — a forward one with `omega` and an inverse
one with `gl_inv(omega)` — so the inverse transform reuses the identical
butterfly kernel.

### 3.3 The butterfly — `k_butterfly`

This is the workhorse. Thread `b` handles butterfly `b`. The thread decodes `b`
into which size-`len` block it belongs to and its position `j` within that
block's first half, fetches the twiddle with the per-stage stride, and writes
both outputs:

```cpp
__global__ void k_butterfly(uint64_t* a, uint32_t n, uint32_t len,
                            const uint64_t* W, uint32_t stride) {
    uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = len >> 1;
    if (b >= (n >> 1)) return;

    uint32_t block = b / half;            // which size-`len` block
    uint32_t j     = b % half;            // position in the block's first half
    uint32_t i     = block * len;         // base index of the block

    uint64_t w = W[(uint64_t)j * stride]; // omega_n^{ j*(n/len) } = omega_len^j
    uint64_t u = a[i + j];
    uint64_t v = gl_mul(a[i + j + half], w);
    a[i + j]        = gl_add(u, v);
    a[i + j + half] = gl_sub(u, v);
}
```

Every arithmetic op (`gl_mul`, `gl_add`, `gl_sub`) is exact Goldilocks-field
arithmetic — no floating point, no rounding, which is *why* the product comes
out bit-exact (see [05_ntt_goldilocks.md](05_ntt_goldilocks.md)). The
`gl_mul` here is the only place the device needs a 64×64→128 multiply, which is
the portability story of §7.

### 3.4 Pointwise product — `k_pointwise`

Once both operands are in the frequency domain, convolution becomes elementwise
multiplication (the convolution theorem). One thread per element:

```cpp
__global__ void k_pointwise(uint64_t* a, const uint64_t* b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = gl_mul(a[i], b[i]);
}
```

### 3.5 Inverse normalization — `k_scale`

An inverse NTT comes out scaled by `N`, so the final step multiplies every
element by `N^{-1}` in the field. `gl_inv(N)` is computed once **on the host**
(it is cheap, one Fermat exponentiation) and passed in as a scalar:

```cpp
__global__ void k_scale(uint64_t* a, uint32_t n, uint64_t ninv) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = gl_mul(a[i], ninv);
}
```

### Kernel summary

| Kernel             | Threads launched | Job                                          |
| ------------------ | ---------------- | -------------------------------------------- |
| `k_bit_reverse`    | `N`              | Permute into bit-reversed order (`__brev`)   |
| `k_build_twiddles` | `N/2`            | Fill `W[t] = omega^t`                         |
| `k_butterfly`      | `N/2` per stage  | One radix-2 butterfly each (`log2 N` launches) |
| `k_pointwise`      | `N`              | `a[i] *= b[i]` in `GF(p)`                     |
| `k_scale`          | `N`              | `a[i] *= N^{-1}` (inverse only)              |

---

## 4. Host orchestration and host↔device transfers

`mul_ntt_cuda()` is the public GPU multiply. Its structure mirrors the CPU
version exactly; only the three transforms and the pointwise product move to the
device. The full pipeline:

1. **Pick the transform length.** `n = ntt_length(a.size(), b.size())` (the next
   power of two large enough to hold the convolution), `logn = ntt_log2(n)`.

2. **Repack on the host.** Each operand is sliced into 16-bit transform digits
   with `repack_to16(a, n)`, producing zero-padded `std::vector<uint64_t>` of
   length `n`. (Why 16-bit digits, and why a single prime suffices, is the "no
   CRT needed" argument in [goldilocks.h](../include/goldilocks.h): each
   convolution coefficient stays below `p`, so the field result *equals* the
   integer result.)

3. **Allocate four device buffers** — two operands of `n` elements and two
   twiddle tables of `n/2` elements:

   ```cpp
   const size_t bytesN = (size_t)n * sizeof(uint64_t);
   const size_t bytesH = (size_t)(n / 2) * sizeof(uint64_t);
   CUDA_TRY(cudaMalloc(&d_a, bytesN));
   CUDA_TRY(cudaMalloc(&d_b, bytesN));
   CUDA_TRY(cudaMalloc(&d_Wf, bytesH));   // forward twiddles
   CUDA_TRY(cudaMalloc(&d_Wi, bytesH));   // inverse twiddles
   ```

4. **Copy operands host→device** with two `cudaMemcpy(..., cudaMemcpyHostToDevice)`.

5. **Build both twiddle tables on the device** with `k_build_twiddles`, using
   `omega = gl_root_of_unity(n)` for the forward table and `gl_inv(omega)` for
   the inverse.

6. **Run the three transforms.** `run_transform(d_a, …, d_Wf, false)` and the
   same for `d_b`, then `k_pointwise(d_a, d_b, n)`, then
   `run_transform(d_a, …, d_Wi, true)` for the inverse (which also runs
   `k_scale`).

7. **Synchronize.** A single `cudaDeviceSynchronize()` waits for the whole
   asynchronous pipeline and surfaces any launch error that the earlier
   `cudaGetLastError()` checks could not catch (kernel launches are async, so a
   runtime fault can appear only at the sync point).

8. **Copy the result device→host** (`cudaMemcpyDeviceToHost`) and finish on the
   host with `carry_propagate(hc)`, which turns the raw convolution coefficients
   back into a normalized big integer.

9. **Free everything** unconditionally — `cudaFree` on all four buffers runs even
   on the failure path, so there is no leak when we bail out.

A subtle but important detail: **all kernel launches and `cudaMemcpy` calls go
to the default stream in program order.** That means each transform's
`log2(N)` butterfly launches are implicitly serialized behind one another (the
inter-stage barrier of §2), and the final device→host copy cannot start until
the inverse transform's writes have completed. We get the ordering we need for
free from default-stream semantics, plus the one explicit
`cudaDeviceSynchronize()` before reading results back.

---

## 5. Graceful CPU fallback

The robustness contract is stated at the top of the file:

> If there is no CUDA device, or any CUDA call fails …, `mul_ntt_cuda()`
> transparently falls back to the CPU NTT so the program always produces a
> correct answer.

Two mechanisms implement it.

**Device probe (cached).** `cuda_available()` calls `cudaGetDeviceCount` once and
caches the answer in a `static int`, so we never repeatedly poke the driver:

```cpp
bool cuda_available() {
    static int cached = -1;                       // -1 unknown, 0 no, 1 yes
    if (cached < 0) {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        cached = (e == cudaSuccess && count > 0) ? 1 : 0;
    }
    return cached == 1;
}
```

If there is no GPU, the very first line of `mul_ntt_cuda()` returns
`mul_ntt_cpu(a, b)`.

**Error latch.** Every CUDA call is wrapped in the `CUDA_TRY` macro, which on
failure prints a diagnostic and clears a local `bool ok`:

```cpp
#define CUDA_TRY(expr)                                                          \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "[cuda] %s failed: %s\n", #expr,               \
                         cudaGetErrorString(_e));                               \
            ok = false;                                                         \
        }                                                                       \
    } while (0)
```

Once `ok` is false, the subsequent `if (ok) …` guards skip the rest of the GPU
work, the buffers are freed, and the function ends with:

```cpp
if (!ok) return mul_ntt_cpu(a, b);
return result;
```

So an out-of-memory `cudaMalloc`, a launch failure, or a transfer error all
funnel into the same place: the correct CPU answer. **Correctness never depends
on the GPU being present.** This is exactly the right posture for a numerics
library — the GPU is an accelerator, not a source of truth.

---

## 6. The BBP kernel: embarrassingly parallel

[src/bbp_cuda.cu](../src/bbp_cuda.cu) is the cleanest illustration of CUDA's
data-parallel model in the whole project, because BBP digit extraction has **no
shared state at all**. The hex digit at each position is computed with zero
reference to any other position. So the mapping is the simplest possible: launch
`count` threads, thread `i` computes the digit at `start + i`, writes it to
`out[i]`. No shared memory, no synchronization, no inter-thread communication.

```cpp
__global__ void k_bbp(unsigned char* out, uint64_t start, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = (unsigned char)bbp_digit_value(start + (uint64_t)i);
}
```

The per-digit math is the shared `bbp_digit_value()` from
[bbp_kernel.h](../include/bbp_kernel.h) — *identical* to what the CPU runs (see
§7 for how one source compiles for both). That routine sums the head terms
(`16^(d-k) mod (8k+j)` via exact modular exponentiation `bbp_modpow16`) and the
geometrically shrinking tail, in IEEE-754 double, to recover the leading hex
digit of `{16^d · pi}`.

Host orchestration is minimal — one buffer, one launch, one copy back:

```cpp
const int TPB = 256;                              // threads per block
k_bbp<<<(count + TPB - 1) / TPB, TPB>>>(d_out, start, count);
e = cudaGetLastError();              // catch launch error
…
e = cudaDeviceSynchronize();         // wait, catch async fault
…
e = cudaMemcpy(host.data(), d_out, bytes, cudaMemcpyDeviceToHost);
```

Each digit is stored as a single byte (`unsigned char`) in `out[i]`, so the
device buffer is just `count` bytes; the host converts each `0..15` value to a
hex character with `to_hex_char` at the very end. And exactly like the NTT path,
**any** CUDA error (malloc, launch, sync, or copy) sets `ok = false` and the
function returns `bbp_hex_digits_cpu(start, count)` — "correctness never depends
on the GPU."

> **Why BBP is the textbook GPU case.** Each thread's work is independent and
> compute-bound (modular exponentiation), the memory footprint is one output byte
> per thread, and there is no communication. This is the definition of
> *embarrassingly parallel*, and it is where the GPU's thousands of lanes pay off
> most directly — when there are enough positions to fill them (see §9).

---

## 7. Portability: no `__int128` on Windows, and the HD macro trick

Two portability problems had to be solved to make these files build on Windows
(MSVC host compiler + nvcc) as well as on Linux/GCC. Both are solved in the
headers, not the `.cu` files.

### 7.1 No `__int128` on MSVC → dispatch the high-half multiply

The Goldilocks reduction needs the **high 64 bits** of a 64×64 product. The
clean way to write that is `(unsigned __int128)a * b >> 64` — but **MSVC has no
`__int128` type**. The fix is a single dispatcher, `gl_mulhi`, in
[goldilocks.h](../include/goldilocks.h):

```cpp
GL_HD uint64_t gl_mulhi(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);                 // device intrinsic
#elif defined(_MSC_VER) && defined(_M_X64)
    return __umulh(a, b);                     // MSVC x64 host intrinsic
#else
    return (uint64_t)(((unsigned __int128)a * b) >> 64); // portable fallback
#endif
}
```

Three branches, selected at compile time:

- **On the device** (`__CUDA_ARCH__` defined) → `__umul64hi`, the PTX
  high-multiply intrinsic.
- **On the MSVC x64 host** → `__umulh` from `<intrin.h>` (included via the
  `#if defined(_MSC_VER)` guard near the top of the header).
- **Everywhere else** (GCC/Clang host) → the `__int128` fallback.

Everything downstream — `gl_mul`, `gl_reduce128`, `gl_pow`, the whole field —
calls `gl_mulhi` and stays platform-agnostic. The same `gl_mulhi` is even reused
by the BBP kernel's `bbp_mulmod` for its 128-bit slow path, so the single
intrinsic dispatch covers both algorithms.

### 7.2 One source for host *and* device: the `GL_HD` / `BBP_HD` macros

We want the *exact same* arithmetic to run on the CPU (in a `.cpp`) and the GPU
(in a `.cu`). CUDA requires device functions to carry `__host__ __device__`
qualifiers, but a plain C++ compiler rejects those keywords. The trick is a macro
that expands to the qualifiers only under nvcc:

```cpp
#if defined(__CUDACC__)
#  define GL_HD __host__ __device__ __forceinline__
#else
#  define GL_HD inline
#endif
```

Every field function is declared `GL_HD uint64_t gl_mul(...)`, so:

- compiled by **nvcc** (`__CUDACC__` defined), it becomes a host+device inline
  function callable from kernels, and
- compiled by **plain g++/MSVC**, it becomes an ordinary `inline` function for
  the CPU path.

[bbp_kernel.h](../include/bbp_kernel.h) mirrors this exactly with its own
`BBP_HD` macro:

```cpp
#if defined(__CUDACC__)
#  define BBP_HD __host__ __device__ __forceinline__
#else
#  define BBP_HD inline
#endif
```

So `bbp_digit_value` and friends are written **once** and serve both the CPU
extractor and `k_bbp` on the GPU. This is the single most important structural
decision for maintainability: there is exactly one implementation of the
mathematics, and it cannot drift between host and device.

> **Gotcha worth remembering:** `__CUDACC__` is defined whenever nvcc is
> compiling a `.cu` file — *including its host pass*. That is why the MSVC
> `<intrin.h>` include is gated on `_MSC_VER` (not on the absence of CUDA): the
> host pass of a `.cu` file still needs `__umulh`, and the device-vs-host choice
> inside `gl_mulhi` is made by the *narrower* `__CUDA_ARCH__` macro, which is
> only set during the actual device code-generation pass.

---

## 8. Occupancy, threads-per-block, and memory footprint

### Threads per block: 256

Every kernel launch in both files uses `const int TPB = 256;`. 256 is a
deliberate, conventional choice:

- It is a multiple of the 32-thread warp size (8 warps per block), so no lanes
  are wasted to partial warps.
- It is small enough that many blocks fit on a streaming multiprocessor at once
  (good occupancy: more resident warps to hide memory and arithmetic latency),
  yet large enough to amortize per-block scheduling overhead.
- None of these kernels use shared memory and they use few registers, so
  occupancy is not register- or smem-limited; 256 comfortably saturates the
  device for the grid sizes we launch.

The grid size is always the standard ceiling-division idiom, e.g.
`(n + TPB - 1) / TPB` blocks, with an `if (i >= n) return;` guard in every kernel
so the final partial block does no out-of-bounds work. This is the canonical
"grid-stride-free" 1-D launch pattern: one thread per data element, bounds-checked.

### Memory footprint

**NTT.** Four device buffers per multiply:

| Buffer        | Size                  | Purpose            |
| ------------- | --------------------- | ------------------ |
| `d_a`         | `n · 8` bytes         | operand A / result |
| `d_b`         | `n · 8` bytes         | operand B          |
| `d_Wf`        | `(n/2) · 8` bytes     | forward twiddles   |
| `d_Wi`        | `(n/2) · 8` bytes     | inverse twiddles   |

Total ≈ `n · 8 · (2 + 1) = 24n` bytes (the two half-size tables sum to one
full-size buffer). For `n = 2^20` that is ~24 MB; for `n = 2^24` ~384 MB. The
transform is done **in place** in `d_a`/`d_b` — the butterflies overwrite their
inputs — so there is no separate scratch array, and a too-large transform simply
fails the `cudaMalloc` and falls back to the CPU (§5).

**BBP.** One buffer of `count` bytes (`unsigned char` per digit). Trivial: a
million digits is a megabyte. BBP is compute-bound, not memory-bound.

---

## 9. Measured speedups — and when the GPU does *not* help

From the project's demo runs:

- **NTT multiply:** the GPU transforms complete in **tens of milliseconds**
  where the CPU NTT takes **seconds** for the same large operands. The win grows
  with `N`, because the GPU is doing `N/2` butterflies genuinely in parallel
  while the CPU walks them sequentially within each stage.

- **BBP extraction:** roughly **59×–90×** faster than the CPU for **many** hex
  positions. With thousands of independent, compute-heavy positions, the GPU's
  lanes are saturated and each does a full modular-exponentiation digit
  independently — the ideal case from §6.

### When the GPU does **not** help

The GPU is not free. Three fixed costs must be paid every call: the device
probe, the `cudaMalloc`/`cudaFree` pair, and the host↔device `cudaMemcpy`s,
plus kernel-launch latency. When the *work* is small, those fixed costs dominate
and the CPU wins outright. Concretely:

- **A few BBP positions.** If you ask for only a handful of hex digits, you
  launch a handful of threads — leaving the vast majority of the GPU idle — and
  still pay full allocation, launch, sync, and copy-back overhead. For small
  `count`, `bbp_hex_digits_cpu` finishes before the GPU has even been set up.
  The GPU's advantage is *throughput at scale*, not *latency for one item*.

- **Tiny multiplies.** A small `n` means a transform of a few thousand points,
  which the CPU finishes in microseconds; the host↔device transfers alone cost
  more than that. (In the broader pipeline the multiply dispatcher should — and
  does — keep small products on the CPU.)

The honest rule of thumb: **the GPU pays off when there is enough independent
work to fill thousands of lanes and amortize the transfer/launch overhead.**
Below that threshold, prefer the CPU. The fallback machinery of §5 guarantees
correctness either way; choosing the *fast* path for a given size is a separate,
size-based decision.

---

## 10. Honest limitations

A few caveats, stated plainly so nobody is surprised:

- **BBP double-precision reliability.** Each BBP thread accumulates its head/tail
  sums in IEEE-754 `double` (53-bit mantissa). As the header in
  [bbp_kernel.h](../include/bbp_kernel.h) warns, that is enough for roughly the
  first ~10 hex digits of `{16^d · pi}` to be correct. Because each thread
  extracts only the **single leading** hex digit and keeps ~10 digits of slack,
  the result is reliable across the range we use (and the test suite checks it
  against known digits). Extracting *many* correct digits at one position would
  require extended precision — see [03_bbp.md](03_bbp.md).

- **BBP modulus range.** `bbp_mulmod` has a fast path only while the modulus fits
  in 32 bits; beyond ~500 million positions it switches to the exact 128-bit
  16-bits-at-a-time reduction (valid for `m < 2^48`, i.e. positions up to
  ~3.5×10¹³). That slow path is correct but slower.

- **Single-prime NTT ceiling.** The "one prime, no CRT" argument holds only while
  each convolution coefficient stays below `p`. As [goldilocks.h](../include/goldilocks.h)
  notes, the numbers big enough to break this have ~1.6 billion digits — we run
  out of GPU memory long before that — but it *is* a real bound, not infinity.

- **Division and decimal conversion are still quadratic.** The GPU here
  accelerates *multiplication* (and BBP). The surrounding big-integer division
  and binary→decimal conversion steps in the pipeline remain `O(n^2)`-flavored
  and are not GPU-accelerated; for very large outputs they, not the multiply,
  become the bottleneck.

- **One launch per stage has a floor.** For small `N` the `log2(N)` serialized
  butterfly launches plus transfers cost more than just doing the multiply on the
  CPU (see §9). The design is tuned for *large* transforms.

---

### See also

- [02_binary_splitting.md](02_binary_splitting.md) — how the big multiplies arise
- [03_bbp.md](03_bbp.md) — the BBP formula and its precision analysis
- [05_ntt_goldilocks.md](05_ntt_goldilocks.md) — the NTT math and field theory
- Source: [src/ntt_cuda.cu](../src/ntt_cuda.cu),
  [src/bbp_cuda.cu](../src/bbp_cuda.cu),
  [include/goldilocks.h](../include/goldilocks.h),
  [include/bbp_kernel.h](../include/bbp_kernel.h)
