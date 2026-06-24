# Scaling to Billions: An Honest Analysis

This document does two things. First, it works through the **memory arithmetic** of
computing pi to large precision, so you can predict — to within a small constant
factor — how many digits a given GPU can actually hold. Second, it gives an
**honest accounting of the real bottlenecks in this codebase**: which operations
are already fast, and which ones quietly become the wall you hit long before you
run out of memory.

The punchline up front:

- The **multiply** in this project is genuinely fast: a Goldilocks NTT running on
  the GPU, `O(n log n)`. See [src/ntt_cuda.cu](../src/ntt_cuda.cu) and
  [docs/05_ntt_goldilocks.md](05_ntt_goldilocks.md).
- The **division**, **integer square root**, and **decimal conversion** are *not*
  yet sub-quadratic. They lean on Knuth long division (`O(n^2)`) and repeated
  short division. For a few million digits this is fine. Past roughly `10^6`
  digits these `O(n^2)` steps dominate the whole run and the fast multiply stops
  mattering.
- An 8 GB card (the RTX 2080 this project targets) comfortably handles **a few
  million decimal digits**. **Billions of decimal digits** need **tens of GB** of
  working memory and are simply **not feasible on this card**. That is a hardware
  fact, not a code defect.

The code is *architected* for scale — the `div_fast` seam, the single-prime NTT,
the host/device-shared field arithmetic — and *demonstrated* at the millions
scale. The sub-quadratic pieces are the documented next steps, laid out at the end.

---

## Part 1 — The memory math

### Step 1: decimal digits to bits

We want `D` decimal digits of pi. Information-theoretically, one decimal digit
carries `log2(10) ~= 3.3219` bits. So the number itself needs

```
bits = D * log2(10) ~= 3.3219 * D
```

A handy round figure: **10 decimal digits ~= 33.2 bits**, or about **1 GB of bits
per 2.4 billion digits**. But "the number" is the smallest part of the story; the
*transform* that multiplies two such numbers is where the memory actually goes.

### Step 2: bits to 16-bit NTT digits

The NTT multiply does not transform the number in its native base. Look at how
`mul_ntt_cuda` prepares its operands in [src/ntt_cuda.cu](../src/ntt_cuda.cu):

```cpp
// Repack operands into 16-bit transform digits on the host.
std::vector<uint64_t> ha = repack_to16(a, n);
std::vector<uint64_t> hb = repack_to16(b, n);
```

Each operand is sliced into **16-bit digits** before the transform. Why 16 bits and
not 32? Because the convolution must not overflow the single Goldilocks prime. The
header [include/goldilocks.h](../include/goldilocks.h) spells out the bound in its
"no CRT needed" argument:

```
c_k = sum a_i * b_j   with a_i, b_j < 2^16, so each product < 2^32,
and there are at most N/2 such products, giving c_k < (N/2) * 2^32 = N * 2^31.
For any transform length N <= 2^32 this is < 2^63 < p.
```

So 16-bit digits keep every convolution coefficient below

```
p = 2^64 - 2^32 + 1   (GL_P in goldilocks.h)
```

and the field result equals the exact integer result — no wraparound, no second
prime, no CRT. That is the whole reason the digit size is 16 and not larger.

The number of 16-bit digits in one operand is therefore

```
digits16 = ceil(bits / 16) = ceil(3.3219 * D / 16) ~= 0.2076 * D
```

Roughly **one 16-bit NTT digit per 4.8 decimal digits**.

### Step 3: 16-bit digits to transform length N

An NTT convolves two sequences by zero-padding both to a common power-of-two
length `N` that is large enough to hold the full product without the cyclic
convolution wrapping. The product of two `digits16`-digit operands has about
`2 * digits16` digits, so

```
N = next_power_of_two( 2 * digits16 )   (this is what ntt_length / ntt_log2 pick)
```

In round terms, `N` is a little over `2 * 0.2076 * D ~= 0.415 * D`, then rounded
**up** to the next power of two. That rounding is not a rounding error you can
ignore — in the worst case it nearly **doubles** `N`. Budget for it.

### Step 4: transform length N to bytes

This is where the cost lives. Every element of the transform is a Goldilocks field
element, and those are stored as `uint64_t` — **8 bytes each** — throughout
[src/ntt_cuda.cu](../src/ntt_cuda.cu):

```cpp
const size_t bytesN = (size_t)n * sizeof(uint64_t);       // one length-N array
const size_t bytesH = (size_t)(n / 2) * sizeof(uint64_t); // one twiddle table
```

Now count what `mul_ntt_cuda` actually allocates on the device for a single
multiply:

```cpp
CUDA_TRY(cudaMalloc(&d_a, bytesN));   // operand A spectrum        N * 8
CUDA_TRY(cudaMalloc(&d_b, bytesN));   // operand B spectrum        N * 8
CUDA_TRY(cudaMalloc(&d_Wf, bytesH));  // forward twiddle table   N/2 * 8
CUDA_TRY(cudaMalloc(&d_Wi, bytesH));  // inverse twiddle table   N/2 * 8
```

So one GPU multiply needs

```
device_bytes = 2 * (N * 8) + 2 * (N/2 * 8) = 24 * N bytes
```

Plus the host-side `ha`, `hb`, and the `hc` result vector — each `N * 8` bytes of
pinned-or-pageable host memory — but those live in system RAM, not on the card.
The binding constraint is the **24 * N bytes of device memory** (in this allocation
pattern) for the multiply at the root of the binary-splitting tree.

### Step 5: putting it together — a worked table

Combine the chain `D -> bits -> digits16 -> N -> bytes`. The transform length
dominates because it sits inside two power-of-two roundings, so the honest way to
read this table is "order of magnitude, expect a factor of up to 2 either way."

| Decimal digits `D` | Operand bits | 16-bit digits | Approx. `N` (pow2) | GPU bytes (`24N`) |
|--------------------|--------------|---------------|--------------------|-------------------|
| 10^5 (100 K)       | ~0.33 Mbit   | ~21 K         | 2^16 = 65 536      | ~1.5 MB           |
| 10^6 (1 M)         | ~3.3 Mbit    | ~208 K        | 2^19 = 524 288     | ~12 MB            |
| 10^7 (10 M)        | ~33 Mbit     | ~2.08 M       | 2^22 = 4.19 M      | ~101 MB           |
| 10^8 (100 M)       | ~332 Mbit    | ~20.8 M       | 2^26 = 67.1 M      | ~1.6 GB           |
| 10^9 (1 B)         | ~3.32 Gbit   | ~208 M        | 2^29 = 537 M       | ~12.9 GB          |
| 10^10 (10 B)       | ~33.2 Gbit   | ~2.08 B       | 2^32 = 4.29 B      | ~103 GB           |

Read off the consequences:

- **A few million digits is comfortable on 8 GB.** At 10 million decimal digits the
  *root* multiply's device footprint is on the order of **100 MB**. Even allowing
  for several such buffers live at once, plus the host-side copies, you are using a
  small fraction of an RTX 2080's 8 GB. This is the regime the project is
  demonstrated in.

- **One billion decimal digits needs ~13 GB for the single root multiply alone** —
  before counting the BigInt operands themselves, the binary-splitting tree's other
  live nodes, the `isqrt` scratch, or the division's normalized copies. That does
  not fit in 8 GB. It would be tight even on a 24 GB card.

- **Ten billion decimal digits is ~100 GB** and also bumps into the NTT's own
  ceiling: `N` reaches `2^32`, the maximum transform length the single Goldilocks
  prime supports (its two-adicity is exactly 32; see property 2 in
  [include/goldilocks.h](../include/goldilocks.h)). Past that you must pack wider
  digits and bring in multi-prime CRT (discussed below), which only *adds* memory.

So: **billions of decimal digits is a tens-of-GB problem and is not feasible on an
8 GB RTX 2080.** No amount of code cleverness changes the byte count of a length-`N`
transform. This is the honest hardware ceiling.

> A useful sanity check from [include/goldilocks.h](../include/goldilocks.h): the
> single prime only *mathematically* breaks at numbers with ~1.6 billion digits.
> The comment itself notes "we run out of GPU memory long before that." The memory
> wall, not the arithmetic, is what stops you first.

---

## Part 2 — The real bottlenecks in THIS codebase

Suppose you *do* have the memory (a bigger card, or you stay in the few-million
range). Is the multiply the thing that takes the time? Below a few hundred thousand
digits, the multiply dominates and everything is great. Above roughly `10^6`
digits, three other operations — all currently **super-linear and not yet
accelerated** — take over the wall-clock time. Here they are, honestly.

### The fast part: NTT multiply, `O(n log n)` on the GPU

Credit where due. [src/ntt_cuda.cu](../src/ntt_cuda.cu) implements the whole
multiply pipeline on the device: bit-reversal, `log2(N)` butterfly stages, a
pointwise product, and the inverse transform. The butterfly stage is one kernel
launch per stage with `N/2` independent threads:

```cpp
for (uint32_t len = 2; len <= n; len <<= 1) {
    uint32_t stride = n / len;
    k_butterfly<<<(halfN + TPB - 1) / TPB, TPB>>>(d_a, n, len, W, stride);
    CUDA_TRY(cudaGetLastError());
}
```

Total work `O(N log N)`, and all the heavy field arithmetic (`gl_mul`, `gl_add`,
`gl_sub`, `gl_reduce128` from [include/goldilocks.h](../include/goldilocks.h)) is
division-free thanks to the Solinas reduction. This is the part of the system that
is genuinely built for scale. The Chudnovsky assembly in
[src/chudnovsky.cpp](../src/chudnovsky.cpp) routes its big multiplies through it
automatically, because every `BigInt::operator*` dispatches by size and the large
products near the root of the [binary splitting](02_binary_splitting.md) tree land
on the GPU.

So the multiply is not the problem. The next three operations are.

### Bottleneck (a): final division is Knuth `O(n^2)`

The last step of `compute_pi_chudnovsky` is one enormous division —
`floor(pi * 10^prec) = numerator / T` — in [src/chudnovsky.cpp](../src/chudnovsky.cpp):

```cpp
BigInt pi_scaled = div_fast(numerator, r.T);        // floor(pi * 10^prec)
```

The name `div_fast` is aspirational. Open [src/bignum_div.cpp](../src/bignum_div.cpp)
and read what it actually does today:

```cpp
BigInt div_fast(const BigInt& a, const BigInt& b) {
    // Placeholder for a future sub-quadratic divider. Today it forwards to the
    // exact Knuth division so every caller already gets a correct result; only
    // the asymptotic speed of very large divisions will improve when a Newton or
    // Burnikel-Ziegler implementation replaces this body.
    BigInt q, r;
    divmod_knuth(a, b, q, r);
    return q;
}
```

`div_fast` **forwards to `divmod_knuth`**, which is Knuth's Algorithm D
(`divmod_mag`). Algorithm D produces one quotient limb per iteration; each
iteration multiplies the whole divisor by the estimated quotient digit and
subtracts it across all `n` limbs:

```cpp
for (size_t jj = m - n + 1; jj-- > 0;) {        // ~m - n quotient limbs
    ...
    for (size_t i = 0; i < n; ++i) {            // O(n) work per limb
        uint64_t p = qhat * vn[i];
        ...
    }
}
```

That is an outer loop of length `~(m - n)` wrapping an inner loop of length `n`:
**`O(n^2)`** in the limb count, and it runs entirely on the CPU. For a few million
digits it is tolerable. At tens of millions of digits this single division can
dominate the whole run, because the multiply it competes with is only `O(n log n)`.

The seam is deliberately placed: callers already say `div_fast`, so when a
sub-quadratic divider is written, *nothing in [src/chudnovsky.cpp](../src/chudnovsky.cpp)
changes*. But as of today, the asymptotics are quadratic.

### Bottleneck (b): isqrt does ~log(bits) of those quadratic divisions

`compute_pi_chudnovsky` also needs `sqrt(10005)` to full precision, computed as an
integer square root of a `2 * prec`-digit radicand:

```cpp
BigInt radicand = BigInt::from_u64(R_CONST) * one * one; // 10005 * 10^(2 prec)
BigInt sqrtC = isqrt(radicand);
```

`isqrt` in [src/bignum_div.cpp](../src/bignum_div.cpp) is Newton's method:

```cpp
BigInt isqrt(const BigInt& s) {
    ...
    BigInt x = BigInt((int64_t)1).shl(bits / 2 + 1);
    while (true) {
        BigInt q, r;
        divmod_knuth(s, x, q, r);                // q = floor(s / x)
        BigInt y = (x + q).shr(1);               // y = (x + s/x) / 2
        if (y.cmp(x) >= 0) break;
        x = y;
    }
    return x;
}
```

Newton's method converges quadratically, so it needs only **`~log2(bits)`
iterations** — a small number, maybe 30-something even for enormous inputs. The
catch is the body of the loop: each iteration calls `divmod_knuth`, the very same
`O(n^2)` long division. So `isqrt` costs roughly

```
log2(bits) * O(n^2)
```

The `log2(bits)` factor is gentle; the `O(n^2)` factor is not. At scale, the square
root is a second quadratic millstone hanging next to the final division. A
sub-quadratic divider fixes **both** at once — which is exactly why fixing division
is the highest-leverage change in the codebase.

### Bottleneck (c): to_decimal is repeated division by 10^9, also ~`O(n^2)`

Even after pi is computed as a scaled integer, you still have to print it in base
10. `BigInt::to_decimal` in [src/bignum_div.cpp](../src/bignum_div.cpp) peels off
nine decimal digits at a time by short-dividing the whole number by `10^9`:

```cpp
while (!cur.empty()) {
    uint64_t rem = 0;
    for (size_t i = cur.size(); i-- > 0;) {  // short division of `cur` by 10^9
        uint64_t v = (rem << 32) | cur[i];
        cur[i] = (uint32_t)(v / TEN9);
        rem = v % TEN9;
    }
    while (!cur.empty() && cur.back() == 0) cur.pop_back();
    groups.push_back((uint32_t)rem);         // this 9-digit group is done
}
```

The inner short-division loop is `O(n)` (it touches every limb). It runs once per
9-digit group, and there are `~D/9` groups. The number does shrink by one limb every
~few groups, but the total work is the classic arithmetic series and comes out to
**`O(n^2)`**. The file's own comment is candid about it:

```
This is O(d^2) in the digit count d, but uses only the very fast
single-limb division inner loop.
```

So for a result with millions of digits, the *output formatting alone* is a
quadratic pass. It uses a cheap inner loop, so its constant factor is small — but
quadratic is quadratic, and at `10^7`-`10^8` digits it becomes a noticeable slice
of the run.

### Summary: where the time actually goes

| Operation                         | Where                                   | Today's complexity | On GPU? |
|-----------------------------------|-----------------------------------------|--------------------|---------|
| Big multiply (root of BS tree)    | `mul_ntt_cuda` ([src/ntt_cuda.cu](../src/ntt_cuda.cu)) | `O(n log n)` | yes |
| Binary splitting merges           | `chudnovsky_bs` ([src/chudnovsky.cpp](../src/chudnovsky.cpp)) | dominated by its multiplies | via multiply |
| Final division                    | `div_fast` -> `divmod_knuth` ([src/bignum_div.cpp](../src/bignum_div.cpp)) | **`O(n^2)`** | no (CPU) |
| Integer square root               | `isqrt` ([src/bignum_div.cpp](../src/bignum_div.cpp)) | **`log(bits) * O(n^2)`** | no (CPU) |
| Decimal output                    | `BigInt::to_decimal` ([src/bignum_div.cpp](../src/bignum_div.cpp)) | **`O(n^2)`** | no (CPU) |

Below ~`10^6` digits the first two rows dominate and the GPU multiply earns its
keep. Above ~`10^6` digits the last three rows take over. That is the honest state
of the codebase: **a fast multiply attached to three quadratic tails.**

---

## Part 3 — The roadmap to billions

Here is the ordered list of work that would move this project from "millions,
demonstrated" toward "billions, in principle" — assuming you also bring the
hardware. Each item names the exact code it would touch.

### 1. Sub-quadratic division built on the fast multiply

This is the single highest-leverage change, because it fixes bottlenecks **(a)** and
**(b)** simultaneously. Two standard approaches, both reducing division to
multiplication so they inherit the `O(n log n)` NTT:

- **Newton-Raphson reciprocal.** Compute an approximate reciprocal `1/T` to the
  needed precision by iterating `x <- x * (2 - T * x)`, which *doubles* the number
  of correct digits each step and uses only multiplications and subtractions. Then
  `numerator / T ~= numerator * (1/T)`, with a final correction step to nail the
  exact floor. Total cost `O(M(n))` where `M(n)` is the multiply cost — i.e.
  `O(n log n)` here. Each multiply in the iteration would route straight through
  `mul_ntt_cuda`.

- **Burnikel-Ziegler recursive division.** A divide-and-conquer long division that
  splits operands into halves and recurses, replacing the inner schoolbook
  multiply-subtract with NTT multiplies. Also `O(M(n) log n)`-ish, and it produces
  an exact quotient and remainder directly (no separate correction), which suits
  the self-checking `q*b + r == a` invariant the test suite already relies on.

The drop-in point is unambiguous: replace the body of `div_fast` in
[src/bignum_div.cpp](../src/bignum_div.cpp). Keep `divmod_knuth` as the
ground-truth reference for tests. Because `isqrt` and `compute_pi_chudnovsky`
already call into this layer (Newton's `isqrt` would call the new divider; the final
step already says `div_fast`), both the square root and the final division get faster
the moment the body changes — no caller edits.

### 2. Sub-quadratic (divide-and-conquer) base conversion

To fix bottleneck **(c)**, replace the repeated-divide-by-`10^9` loop in
`BigInt::to_decimal` with a **divide-and-conquer** scheme:

- Precompute `10^(9 * 2^k)` powers by repeated squaring.
- To print an `n`-limb number, split it once by dividing by the power of 10 that
  sits near its middle, giving a high half and a low half of roughly equal size,
  then recurse on each half and concatenate.

With a sub-quadratic divider underneath (item 1), each split is a fast division and
the recursion gives `O(M(n) log n)` base conversion instead of `O(n^2)`. Note the
dependency: this is only worth doing *after* division is fast, since each split is a
division. The file's comment already points here: "A sub-quadratic
divide-and-conquer variant is described in docs/08_scaling_to_billions.md."

### 3. Persistent GPU buffers and cached twiddles

Today every call to `mul_ntt_cuda` in [src/ntt_cuda.cu](../src/ntt_cuda.cu)
allocates four device buffers, **rebuilds both twiddle tables from scratch**, and
frees everything before returning:

```cpp
CUDA_TRY(cudaMalloc(&d_a, bytesN));   ...
k_build_twiddles<<<...>>>(d_Wf, half, omega);
k_build_twiddles<<<...>>>(d_Wi, half, gl_inv(omega));
...
cudaFree(d_a); cudaFree(d_b); cudaFree(d_Wf); cudaFree(d_Wi);
```

For a single multiply that is fine. But binary splitting performs *many* multiplies,
and a Newton division performs *many more*, often at the same or related transform
lengths. The per-call overhead — allocation, host-to-device copies, and rebuilding
`O(N/2 log N)` worth of twiddles every time — becomes pure waste. The improvements:

- **Cache twiddle tables** keyed by transform length, so a length-`N` table is built
  once and reused across every multiply at that length.
- **Pool / persist device buffers** so repeated multiplies at the same size reuse
  allocations instead of round-tripping the allocator.
- Optionally keep operands resident on the device across a chain of operations to
  cut `cudaMemcpy` traffic.

None of this changes the asymptotics — it attacks the constant factor — but at scale
the constant factor is real wall-clock time.

### 4. Multi-prime CRT — only if you pack wider digits

This project deliberately uses **one** prime and **16-bit** digits, and
[include/goldilocks.h](../include/goldilocks.h) explains exactly why that suffices:
every convolution coefficient stays below `GL_P`, so "one prime recovers the exact
product" and "we sidestep all of that" — the Chinese Remainder Theorem machinery.

You would only need multi-prime CRT if you packed **wider digits** (say 24- or
32-bit) to push more data through each transform element, at which point a single
64-bit prime can no longer hold the convolution sums and you must run the NTT modulo
several primes and recombine with CRT. That is a real technique used by record
setters, but it is a trade, not a free win: it multiplies the number of transforms
(and the memory) by the number of primes. Within the single-prime design here it is
**not needed** until you change the digit packing, and the `N <= 2^32` ceiling
(two-adicity 32) is the more pressing limit anyway.

### 5. Out-of-core and multi-GPU — for true records

Part 1 showed that billions of decimal digits is a tens-of-GB problem. Real record
computations do not fit the transform in one GPU's memory at all. They:

- **Go out-of-core**, streaming pieces of the transform between host RAM (or NVMe
  SSD) and device memory, so the working set on the GPU is a window, not the whole
  array.
- **Span multiple GPUs (or nodes)**, partitioning the transform and exchanging the
  cross-partition data each stage needs.

This is exactly the territory of [**y-cruncher**](http://www.numberworld.org/y-cruncher/),
the program used for essentially every modern pi record. y-cruncher's swap-mode and
multi-threaded, multi-node disk-backed FFTs are the reference design for what
"billions and beyond" actually requires in engineering terms. It is a useful north
star: this codebase shares the same *shape* (fast transform-based multiply, binary
splitting, Newton-style division as the goal) at a far smaller scale.

---

## Where this leaves us — honestly

- The architecture is right: a single-prime Goldilocks NTT multiply on the GPU, a
  binary-splitting Chudnovsky driver, and a `div_fast` seam waiting for a
  sub-quadratic divider. The big multiplications already run on the device.
- The demonstrated regime is **millions of decimal digits**, which an 8 GB RTX 2080
  handles with memory to spare.
- The honest limitations are concrete and named: `div_fast` forwards to Knuth
  `O(n^2)` division, `isqrt` pays that cost `~log(bits)` times, and `to_decimal` is
  a quadratic base conversion. Past ~`10^6` digits these dominate.
- **Billions of decimal digits is a tens-of-GB problem** and is not feasible on this
  card regardless of software — that ceiling is set by the `24 * N`-byte transform,
  not by the code.

The next steps — sub-quadratic division (Newton/Burnikel-Ziegler), divide-and-conquer
base conversion, cached twiddles and persistent buffers, and eventually out-of-core
and multi-GPU — are the documented path from here to there.

See also: [Chudnovsky series](01_chudnovsky.md) ·
[binary splitting](02_binary_splitting.md) ·
[NTT and the Goldilocks field](05_ntt_goldilocks.md) ·
[src/bignum_div.cpp](../src/bignum_div.cpp) ·
[src/ntt_cuda.cu](../src/ntt_cuda.cu) ·
[src/chudnovsky.cpp](../src/chudnovsky.cpp) ·
[include/goldilocks.h](../include/goldilocks.h)
