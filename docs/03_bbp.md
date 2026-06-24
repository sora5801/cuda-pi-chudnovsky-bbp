# 03 — The BBP Formula and Direct Hex-Digit Extraction

> **Source files for this chapter**
> - [include/bbp_kernel.h](../include/bbp_kernel.h) — the shared host/device math (the heart of it)
> - [include/bbp.h](../include/bbp.h) — the public host-facing API
> - [src/bbp_cpu.cpp](../src/bbp_cpu.cpp) — the single-threaded CPU reference
> - [src/bbp_cuda.cu](../src/bbp_cuda.cu) — the GPU kernel (one thread per position)
>
> Companion reading: [02 — binary splitting](02_binary_splitting.md) for the
> *all-digits* Chudnovsky path, and [06 — CUDA design](06_cuda_design.md) for how
> this kernel fits the project's overall GPU strategy.

---

## 1. Why BBP is a different kind of algorithm

Every classical way of computing π — Machin-like arctangents, Gauss–Legendre,
Chudnovsky with [binary splitting](02_binary_splitting.md) — shares one property:
to know the *n*-th digit, you must compute **all** digits up to position *n*. The
work and the memory both grow with *n*. If you want digit one million, you carry
around a million-digit number.

The **Bailey–Borwein–Plouffe (BBP) formula**, discovered in 1995, breaks that
assumption for **hexadecimal** digits. It can compute the *n*-th hex digit of π
**directly**, in `O(n log n)` simple integer operations and — crucially —
**`O(1)` memory**. No big integers. No preceding digits. The comment block at the
top of [bbp_kernel.h](../include/bbp_kernel.h) states the contrast plainly:

```
//  The Chudnovsky algorithm must compute ALL digits up to position n to know the
//  n-th one. The BBP formula (1995) can compute the n-th *hexadecimal* digit of
//  pi DIRECTLY, using O(n log n) simple operations and O(1) memory -- without the
//  preceding digits.
```

That `O(1)`-memory, per-position-independent character is exactly what makes BBP a
*perfect* fit for a GPU, and it is why this is "the cleanest illustration in the
whole project of CUDA's data-parallel model" (comment in
[bbp_cuda.cu](../src/bbp_cuda.cu)).

---

## 2. The BBP formula itself

The formula, exactly as written in the header:

$$
\pi = \sum_{k=0}^{\infty} \frac{1}{16^{k}}
      \left( \frac{4}{8k+1} - \frac{2}{8k+4} - \frac{1}{8k+5} - \frac{1}{8k+6} \right)
$$

The `1/16^k` factor is the whole trick. Because the series is a sum of terms
weighted by powers of `1/16`, multiplying π by `16^d` and taking the fractional
part shifts the hexadecimal point by exactly `d` places — and the leading hex
digit of that fractional part is the digit we want.

To make this tractable, the code splits the formula into four **sub-series**, one
per denominator family. Define:

$$
S_j = \sum_{k \ge 0} \frac{1}{16^{k}\,(8k+j)}
$$

Then π is a simple linear combination:

$$
\pi = 4 S_1 - 2 S_4 - S_5 - S_6
$$

The four `j` values `{1, 4, 5, 6}` and their coefficients `{4, 2, 1, 1}` appear
*verbatim* in the final assembly inside `bbp_digit_value()`:

```cpp
double s = 4.0 * bbp_series(1, d)
         - 2.0 * bbp_series(4, d)
         - 1.0 * bbp_series(5, d)
         - 1.0 * bbp_series(6, d);
```

---

## 3. Reducing to `frac(16^d · π)`

We want the hex digit at 1-based position `pos`. Set `d = pos - 1`. The hex digits
of π **starting just after position `d`** are the leading hex digits of the
fractional part of `16^d · π`. Distributing the `16^d` across the linear
combination and taking everything mod 1:

$$
\{16^{d}\,\pi\} = \big\{\, 4\{16^{d}S_1\} - 2\{16^{d}S_4\} - \{16^{d}S_5\} - \{16^{d}S_6\} \,\big\}
$$

where `{x}` denotes the fractional part of `x`. So the entire problem collapses to:
**compute `{16^d · S_j}` for `j ∈ {1, 4, 5, 6}`.** That single quantity is what
`bbp_series(j, d)` returns, and the assembly above is the literal transcription of
the prompt's `frac(4·S1 − 2·S4 − S5 − S6)`.

The final fractional-part cleanup, again from `bbp_digit_value()`:

```cpp
s -= floor(s);                 // fractional part in [0, 1)
if (s < 0.0) s += 1.0;          // guard against tiny negative rounding
return (int)(16.0 * s);         // leading hex digit
```

The `if (s < 0.0)` guard matters: because the combination subtracts three series
from the first, floating-point rounding can nudge `s` a hair below zero, and a
negative argument to that `(int)(16.0 * s)` truncation would produce a bogus
digit. Adding 1 puts it safely back in `[0, 1)`.

---

## 4. The head/tail split — the engineering core

Computing `{16^d · S_j}` naively would still require summing `1/(16^k (8k+j))` for
`k` from 0 to infinity. The key insight is to split the sum at `k = d`, because the
exponent `d − k` flips sign there:

$$
\{16^{d} S_j\} =
\Bigg\{ \underbrace{\sum_{k=0}^{d} \frac{16^{\,d-k} \bmod (8k+j)}{8k+j}}_{\text{HEAD: exponent} \ge 0}
       + \underbrace{\sum_{k=d+1}^{\infty} \frac{16^{\,d-k}}{8k+j}}_{\text{TAIL: exponent} < 0} \Bigg\}
$$

This is the structure of `bbp_series()`. Let's take the two halves in turn.

### 4.1 The HEAD (`k = 0 … d`): exact modular arithmetic

For `k ≤ d`, the exponent `d − k` is non-negative, so `16^(d-k)` is a (possibly
enormous) integer. But we only ever keep the **fractional part** of the running
sum. The integer part of `16^(d-k) / (8k+j)` contributes nothing to the
fractional accumulator — so we may legally replace the numerator with
`16^(d-k) mod (8k+j)`. That is a *small* number (less than the modulus), computed
**exactly** with modular exponentiation, and the dangerous huge power never
materializes.

```cpp
// HEAD: k = 0 .. d. Each numerator is 16^(d-k) mod (8k+j), computed exactly.
for (uint64_t k = 0; k <= d; ++k) {
    uint64_t m = 8ull * k + j;
    uint64_t r = bbp_modpow16(d - k, m);
    sum += (double)r / (double)m;
    sum -= floor(sum);                   // keep only the fractional part
}
```

The `sum -= floor(sum)` **inside the loop** is not optional bookkeeping — it keeps
`sum` confined to `[0, 1)` on every iteration so the double never loses low-order
bits to a growing integer part. This is what preserves precision across up to *d*
additions.

### 4.2 The TAIL (`k = d+1 … ∞`): the floating geometric remainder

For `k > d`, the exponent `d − k` is negative, so `16^(d-k)` is `16^{-1}`,
`16^{-2}`, … — a geometric sequence shrinking by a factor of 16 each step. After a
dozen or so terms the term is smaller than a double can represent next to the
accumulated sum, so we just add them directly in floating point and stop:

```cpp
double p16 = 1.0 / 16.0;                 // 16^(d-k) for k = d+1
uint64_t k = d + 1;
while (p16 > 1e-18) {
    uint64_t m = 8ull * k + j;
    sum += p16 / (double)m;
    p16 /= 16.0;
    ++k;
}
sum -= floor(sum);
```

The cutoff `1e-18` is below the `~2.2e-16` resolution of a double near 1.0, so once
`p16` drops past it no further term can change `sum`. Typically this loop runs only
~15 iterations regardless of how large `d` is — the tail cost is constant.

---

## 5. Exact modular exponentiation: `16^(d-k) mod (8k+j)`

The head depends entirely on computing `16^e mod m` exactly. Two functions handle
this: `bbp_modpow16` (the square-and-multiply loop) and `bbp_mulmod` (the inner
"multiply two numbers mod *m*" primitive). The subtlety is all in `bbp_mulmod`,
because `a * b` can overflow 64 bits.

### 5.1 Square-and-multiply: `bbp_modpow16`

Standard binary exponentiation — walk the bits of the exponent `e`, squaring the
base each step and multiplying it into the result when the current bit is set:

```cpp
BBP_HD uint64_t bbp_modpow16(uint64_t e, uint64_t m) {
    if (m == 1) return 0;                    // x mod 1 is always 0
    uint64_t result = 1;
    uint64_t base = 16 % m;
    while (e) {
        if (e & 1) result = bbp_mulmod(result, base, m);
        base = bbp_mulmod(base, base, m);
        e >>= 1;
    }
    return result;
}
```

This is `O(log e)` multiplies — and since `e ≤ d ≈ pos`, the whole head is
`O(d log d)`, matching the `O(n log n)` claim. The `if (m == 1) return 0` early-out
guards the `8k+j = 1` corner (it never happens for our `j`, but the modular-power
contract demands it).

### 5.2 `bbp_mulmod`: the fast path and the 128-bit path

```cpp
BBP_HD uint64_t bbp_mulmod(uint64_t a, uint64_t b, uint64_t m) {
    if ((m >> 32) == 0) {
        return (a * b) % m;                  // fast path: no overflow, one divide
    }
    // 128-bit product split into hi:lo, reduced 16 bits at a time, MSB first.
    uint64_t hi = gl_mulhi(a, b);
    uint64_t lo = a * b;
    uint64_t rem = 0;
    for (int sh = 48; sh >= 0; sh -= 16)     // four 16-bit chunks of the high word
        rem = ((rem << 16) | ((hi >> sh) & 0xFFFFu)) % m;
    for (int sh = 48; sh >= 0; sh -= 16)     // four 16-bit chunks of the low word
        rem = ((rem << 16) | ((lo >> sh) & 0xFFFFu)) % m;
    return rem;
}
```

**The fast path (`m < 2^32`).** When the modulus fits in 32 bits, both operands
are already reduced (`a, b < m < 2^32`), so the product `a * b < 2^64` cannot
overflow a `uint64_t`. A single hardware modulo finishes the job. For positions up
to roughly **500 million**, `8k+j` stays under `2^32`, so this branch handles every
realistic query at full speed — one multiply, one divide.

**The 128-bit path (`m ≥ 2^32`).** For larger moduli `a * b` would wrap around
64 bits and lose the high half. Here the code forms the **full 128-bit product**:
`lo = a * b` is the low 64 bits, and `hi = gl_mulhi(a, b)` is the high 64 bits
(`gl_mulhi` is the portable 64×64→high-64 multiply borrowed from
[goldilocks.h](../include/goldilocks.h) — the same primitive the NTT path uses).

It then reduces that 128-bit value mod *m* **16 bits at a time, most-significant
chunk first** — schoolbook long division in base `2^16`. Each step shifts the
running remainder left by 16, ORs in the next 16-bit chunk, and reduces mod *m*.
Because each intermediate `(rem << 16) | chunk` is at most `(m-1)·2^16 + (2^16-1) <
2^48 · 2^16 = 2^64`, the operation never overflows **as long as `m < 2^48`** —
which the header documents covers positions up to about **3.5 × 10¹³**:

```
//  ... reduced 16 bits at a time, which is exact for any m < 2^48
//  (positions up to ~3.5e13).
```

Walking the eight chunks (four from `hi`, four from `lo`) reconstructs the exact
remainder of the 128-bit product. This is slower than the fast path — eight
divides instead of one — but it is only taken at astronomically distant positions,
and it keeps the result **exact** rather than falling back to lossy floating point.

| Modulus range | Branch | Cost | Position reach |
|---|---|---|---|
| `m < 2^32` | `(a*b) % m` | 1 multiply, 1 divide | up to ~5 × 10⁸ |
| `2^32 ≤ m < 2^48` | 128-bit chunked reduction | mulhi + 8 divides | up to ~3.5 × 10¹³ |

---

## 6. The precision caveat — the honest part

This is the single most important thing to understand about this implementation,
and the header is candid about it (the comment is literally titled *"PRECISION
CAVEAT (important, and honest)"*):

> We accumulate the head/tail sums in IEEE-754 **double** (53-bit mantissa). That
> is enough to get roughly the first **~10 hex digits** of `{16^d π}` correct.

Every fractional add into `sum` carries the ~`2.2 × 10⁻¹⁶` rounding error of a
double. Across a head of *d* additions plus the final four-way combination, those
errors accumulate. Empirically this leaves about **10 reliable hex digits** in
`{16^d · π}` before the noise floor — fewer the further out you go.

### Why extracting *one* digit per position is robust

The design sidesteps the caveat completely by a deliberate choice: **each
invocation extracts only the single leading hex digit**, and to get a *block* of
digits the code computes a **separate `{16^d · π}` for every position**.

Look at how a run of digits is produced — there is no "compute once, slice out many
digits." [bbp_cpu.cpp](../src/bbp_cpu.cpp) loops position by position:

```cpp
std::string bbp_hex_digits_cpu(uint64_t start, uint32_t count) {
    std::string out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        out.push_back(to_hex_char(bbp_digit_value(start + i)));
    return out;
}
```

Each call to `bbp_digit_value(start + i)` recomputes the whole head/tail series for
its own `d = start + i − 1` and reads off only the **leading** nibble. Since we
have ~10 digits of accuracy but only consume **1**, there is a comfortable
~9-digit margin against accumulated rounding — at *every* position. That is what
makes the result reliable across the whole range the project uses.

The alternative — extracting many correct digits from a *single* `{16^d π}` — would
demand extended-precision accumulation (more mantissa bits than a double offers).
This implementation deliberately does **not** do that; it trades a little
redundant arithmetic for staying entirely within fast hardware `double`.

### Verification: known prefix and the millionth-position checkpoint

The implementation is not trusted blindly. The comments note the test suite
"verifies it against known digits" and "checks the two [CPU and GPU] against each
other and against the known hex digits of pi." Two anchors matter:

- **The known hex prefix.** π = `3.243F6A88…` in hex, so position 1 is `2`,
  position 2 is `4`, position 3 is `3`, position 4 is `F`, and so on. This is the
  value referenced in [bbp.h](../include/bbp.h): *"position 1 is the first hex
  digit after the radix point (which is 2, since pi = 3.243F6A88...)."* The
  per-digit routine is validated to reproduce this prefix exactly.

- **The position-1,000,000 checkpoint.** The hex digits of π starting at
  position 1,000,000 begin **`26C65E52CB…`**. Hitting this far-out, independently
  published value confirms that the modular-exponentiation head and the geometric
  tail are correct deep into the range — not just near the start where errors are
  smallest. Reproducing a distant checkpoint is the strongest single test of the
  whole pipeline.

The 1-based position convention (position 1 = first hex digit after the point) is
fixed in `bbp_digit_value()` via `d = pos - 1`, so these checkpoints line up
exactly with Bailey's published convention.

---

## 7. The embarrassingly-parallel GPU mapping

Because each position is computed in complete isolation — no shared state, no
ordering, no communication — the GPU mapping is as simple as CUDA gets: **one
thread per hex position.** From [bbp_cuda.cu](../src/bbp_cuda.cu):

```cpp
__global__ void k_bbp(unsigned char* out, uint64_t start, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = (unsigned char)bbp_digit_value(start + (uint64_t)i);
}
```

Thread `i` computes the digit at `start + i` and writes it to `out[i]`. That's the
whole kernel. There is **no shared memory, no `__syncthreads`, no atomics, no
reduction** — the antithesis of the NTT convolution machinery elsewhere in the
project. The launch is the textbook grid-stride-free pattern:

```cpp
const int TPB = 256;                                  // threads per block
k_bbp<<<(count + TPB - 1) / TPB, TPB>>>(d_out, start, count);
```

`256` threads per block, with `(count + TPB - 1) / TPB` blocks rounding up to cover
the tail. The same `bbp_digit_value()` runs on both processors because the routine
is marked `BBP_HD`, which expands to `__host__ __device__ __forceinline__` under
`nvcc` and plain `inline` otherwise:

```cpp
#if defined(__CUDACC__)
#  define BBP_HD __host__ __device__ __forceinline__
#else
#  define BBP_HD inline
#endif
```

This is the project's "write the math once, run it on both" discipline — the CPU
reference and the GPU kernel are guaranteed identical because they compile the
*same source*. See [06 — CUDA design](06_cuda_design.md) for how this
`*_HD` macro pattern recurs across the codebase.

### Graceful CPU fallback

Correctness never depends on a GPU being present. `bbp_hex_digits_cuda` falls back
to the CPU path at every failure point — no device, allocation failure, launch
error, sync error, or copy error:

```cpp
if (!cuda_available()) return bbp_hex_digits_cpu(start, count); // graceful fallback
...
if (!ok) return bbp_hex_digits_cpu(start, count);     // correctness never depends on the GPU
```

The result is bit-identical either way, so the fallback is invisible to the caller.

### Why the GPU wins for many digits — but not always

```
   position d:   start   start+1   start+2   ...   start+count-1
                   |         |         |               |
   thread i:      t0        t1        t2     ...      t(count-1)
                   |         |         |               |
                 [head+tail series, ~O(d log d) each, fully independent]
                   |         |         |               |
   out[i]:        d0        d1        d2     ...      d(count-1)
```

- **Many digits → GPU wins big.** Each position is an independent `O(d log d)`
  computation. With thousands of positions requested, the GPU runs thousands of
  these in parallel across its cores while the CPU grinds them out one at a time.
  This is the ideal workload for data-parallel hardware: lots of identical,
  branch-light, independent work with no synchronization tax.

- **A single far digit can be CPU-fast.** If you only want **one** digit at some
  large position, there is no parallelism to exploit across positions — you have a
  single `O(d log d)` series to evaluate. Launching a kernel then means paying for
  `cudaMalloc`, the kernel launch, `cudaDeviceSynchronize`, and a
  `cudaMemcpy` back — fixed overheads that can dwarf the one computation. A plain
  CPU call (`bbp_hex_digit`, which just wraps `bbp_digit_value`) often returns
  sooner. The GPU's advantage is **throughput across many positions**, not latency
  on a single one.

In short: BBP on the GPU is a width play. Feed it a wide block of positions and it
shines; ask it for one isolated digit and the launch overhead can make the
single-threaded CPU the faster choice.

---

## 8. Summary

| Concept | Where in the code | Note |
|---|---|---|
| BBP formula `π = 4S₁ − 2S₄ − S₅ − S₆` | `bbp_digit_value` | coefficients `{4,2,1,1}`, `j ∈ {1,4,5,6}` |
| Reduce to `frac(16^d·π)` | `bbp_digit_value`, `d = pos-1` | 1-based position convention |
| Head/tail split | `bbp_series` | split at `k = d` where exponent flips sign |
| Exact `16^(d-k) mod m` | `bbp_modpow16` | square-and-multiply, `O(log e)` |
| `mulmod` fast path | `bbp_mulmod`, `m < 2^32` | one multiply, one divide |
| `mulmod` 128-bit path | `bbp_mulmod`, `m < 2^48` | `gl_mulhi` + 16-bit chunked reduction |
| Floating tail | `bbp_series` while-loop | geometric, ~15 terms, cutoff `1e-18` |
| Double-precision caveat | header comment | ~10 reliable hex digits → extract **1** per position |
| Verification | test suite | known prefix `243F6A88…`, position 10⁶ = `26C65E52CB…` |
| GPU mapping | `k_bbp` in `bbp_cuda.cu` | one thread per position, no sync |
| CPU fallback | `bbp_hex_digits_cuda` | bit-identical, always correct |

BBP is the project's answer to "what's the *n*-th digit?" without computing the
first *n* — a constant-memory, embarrassingly-parallel hex-digit faucet. Its honest
limitation is the double-precision floor, neutralized by the one-digit-per-position
design. For the *all-digits-in-decimal* problem, see
[binary splitting](02_binary_splitting.md) and the Chudnovsky pipeline instead.

---

*Next: [04 — division and decimal conversion](04_division.md) ·
Previous: [02 — binary splitting](02_binary_splitting.md) ·
GPU architecture: [06 — CUDA design](06_cuda_design.md)*
