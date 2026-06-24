# 05 — The NTT and the Goldilocks Field (the mathematical heart)

This is the engine room. Everything else in the pipeline — [binary splitting](02_binary_splitting.md) that builds a rational with millions-of-digit numerator and denominator, the final division, the decimal conversion — ultimately bottlenecks on one operation: **multiplying two enormous integers**. Schoolbook multiplication is `O(n^2)`; at a few million digits that is hopeless. This document explains how we get to `O(n log n)` *exactly* (no rounding), using a Number Theoretic Transform over the **Goldilocks prime** `p = 2^64 - 2^32 + 1`.

The relevant source files:

- [include/goldilocks.h](../include/goldilocks.h) — the finite field `GF(p)`: add, sub, mul, the Solinas `reduce128`, pow, inverse, roots of unity.
- [include/ntt.h](../include/ntt.h) — the glue both CPU and GPU share: 16-bit packing, carry propagation, bit reversal, transform-length selection.
- [src/ntt_cpu.cpp](../src/ntt_cpu.cpp) — the readable CPU reference transform (`ntt_transform`) and the public `mul_ntt_cpu`.

The GPU port of this same algorithm is described in [06_cuda_design.md](06_cuda_design.md).

---

## 1. Multiplication *is* convolution

Write a big integer `A` in some base `B` as a sequence of digits `a_0, a_1, a_2, ...` so that

```
A = a_0 + a_1*B + a_2*B^2 + ...
```

Multiply two such numbers `A` and `B` (digits `b_j`) and collect powers of the base. The coefficient sitting on `B^k` in the product — *before* you carry — is

```
c_k = sum over (i + j = k) of a_i * b_j
```

That sum-over-`i+j=k` is exactly the definition of the **discrete linear convolution** of the digit sequences `a` and `b`. So:

> Multiplying two integers = convolving their digit sequences, then propagating carries.

The CPU file states this in its header comment ([src/ntt_cpu.cpp](../src/ntt_cpu.cpp)):

```
//  if A has digits a_i and B has digits b_j (in some base), then the product's
//  digit-weights are c_k = sum_{i+j=k} a_i*b_j  (before carrying).
```

Computing that convolution directly is the `O(n^2)` schoolbook method — every `a_i` meets every `b_j`. The whole trick of fast multiplication is to compute the *same* convolution in `O(n log n)`.

The `c_k` here can be much larger than a single digit (they are sums of products), which is fine and intended: we let them grow, then fix them up at the very end with **carry propagation** (`carry_propagate` in [include/ntt.h](../include/ntt.h)).

---

## 2. The convolution theorem — and why a *transform* helps

The convolution theorem says: convolution in the "time domain" becomes **pointwise multiplication** in the "frequency domain". If `F` is the discrete Fourier transform (DFT), then

```
A (*) B  =  F^{-1}( F(A) .* F(B) )
```

where `(*)` is convolution and `.*` is element-wise (pointwise) product. The CPU header writes it as:

```
//      convolution(A, B) = InverseTransform( Transform(A) .* Transform(B) )
```

Why is this a win? A pointwise product of two length-`N` arrays is `O(N)`. So if we can transform, multiply pointwise, and inverse-transform all in `O(N log N)`, the total cost is `O(N log N)` instead of `O(N^2)`. The Fast Fourier Transform gives us exactly that `O(N log N)` for `F` and `F^{-1}`. That is the entire reason the transform exists in this pipeline: it converts an expensive convolution into a cheap pointwise multiply sandwiched between two fast transforms.

### Why NTT instead of FFT

A classical FFT works with **complex floating-point** roots of unity `e^{2*pi*i/N}`. Those are irrational; the moment you store them in `double` you have rounding error, and that error accumulates across `log N` butterfly stages. For *display* math that is tolerable. For **provably correct digits of pi** it is not — a single wrong bit in a high coefficient corrupts the carry chain and silently poisons the answer.

The fix is to replace the complex field `C` with a **finite field of integers** `GF(p)`, and replace `e^{2*pi*i/N}` with an integer `omega` that behaves identically: a *primitive N-th root of unity*, meaning `omega^N = 1` and no smaller power equals 1. Every value in the transform is then an exact integer in `[0, p)`. No rounding, ever. This is the **Number Theoretic Transform (NTT)**. The CPU header says it plainly:

```
//  ... the NUMBER THEORETIC
//  TRANSFORM (NTT), which is the exact same Cooley-Tukey butterfly structure but
//  carried out in a finite field GF(p) using a primitive root of unity omega in
//  place of the complex number e^{2*pi*i/N}. Working in a field of integers means
//  every value is exact -- no floating-point rounding ...
```

For an NTT to exist at length `N` (a power of two), we need a prime `p` such that `N | (p - 1)`. That divisibility is what guarantees a primitive `N`-th root of unity lives in `GF(p)`. Choosing the right `p` is the next section.

---

## 3. The Goldilocks prime `p = 2^64 - 2^32 + 1`

The field constants live at the top of [include/goldilocks.h](../include/goldilocks.h):

```cpp
constexpr uint64_t GL_P   = 0xFFFFFFFF00000001ULL; // 2^64 - 2^32 + 1
constexpr uint64_t GL_EPS = 0xFFFFFFFFULL;         // 2^32 - 1  (== 2^64 mod p)
constexpr uint64_t GL_G   = 7ULL;                  // smallest primitive root of GF(p)
```

The prime is

```
p = 2^64 - 2^32 + 1 = 18446744069414584321 = 0xFFFFFFFF00000001
```

It is famous in zero-knowledge-proof systems (Plonky2, RISC Zero) and is the right tool here for three reasons the header spells out.

### Property 1 — it fits in a `uint64_t`

`p` is just under `2^64`, so any field element fits in a single 64-bit word. No big-integer bookkeeping inside the inner loop.

### Property 2 — huge two-adicity

```
p - 1 = 2^64 - 2^32 = 2^32 * (2^32 - 1) = 2^32 * 3 * 5 * 17 * 257 * 65537
```

The factor `2^32` is the **two-adicity**: the largest power of two dividing `p - 1` is `2^32`. Since the NTT needs `N | (p - 1)` with `N` a power of two, we can pick *any* transform length up to `N = 2^32` — about 4 billion points. We will never come close to needing that; we run out of GPU memory first. The header:

```
//    2. p - 1 = 2^32 * (2^32 - 1) = 2^32 * 3 * 5 * 17 * 257 * 65537.
//       The factor 2^32 means transform lengths up to 2^32 are available
//       (its "two-adicity" is 32) -- astronomically more than we will ever use.
```

`GL_G = 7` is the smallest primitive root of `GF(p)`: a generator whose powers run through every nonzero element. Because `7` generates the whole multiplicative group of order `p-1`, the element

```
omega = 7^((p-1)/N)
```

has multiplicative order *exactly* `N` — it is our primitive `N`-th root of unity. That is computed by `gl_root_of_unity` (Section 6).

### Property 3 — division-free modular reduction

This is the property that makes Goldilocks *fast*, not just *correct*. The special shape of `p` gives two congruences:

```
2^64 == 2^32 - 1   (mod p)        [because p = 2^64 - (2^32 - 1)]
2^96 == -1         (mod p)
```

The constant `2^32 - 1` is exactly `GL_EPS` (call it EPSILON). These two facts let us reduce a full 128-bit product back into `[0, p)` with only shifts, a couple of multiplies by the small constant `EPS`, adds, and subtracts — **no 64-bit integer division**. On a GPU, 64-bit division is *emulated* in software and painfully slow, so eliminating it is a large win. This style of reduction (exploiting a prime of special form) is a **Solinas reduction**. The header:

```
//    3. ... a 128-bit product can be folded back into [0, p) with a few shifts,
//       multiplies by the small constant EPSILON = 2^32 - 1, adds, and subtracts.
//       This is a "Solinas reduction". On a GPU, avoiding 64-bit integer
//       division (which is emulated and slow) is a big win.
```

---

## 4. The field arithmetic, formula by formula

Every function is marked `GL_HD`, which expands to `__host__ __device__ __forceinline__` under `nvcc` and to plain `inline` otherwise — **the same source compiles for CPU and GPU**:

```cpp
#if defined(__CUDACC__)
#  define GL_HD __host__ __device__ __forceinline__
#else
#  define GL_HD inline
#endif
```

### 4.1 The one platform-specific primitive: `gl_mulhi`

To get the full 128-bit product of two 64-bit numbers without a portable 128-bit type, we need the *high* 64 bits separately. The low 64 bits are just `a * b` (C++ wraps mod `2^64`, which is what we want). The high half dispatches per platform:

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

This is the single place where host and device differ — there is no portable 128-bit integer on MSVC/Windows, so we reach for the right intrinsic (`__umul64hi` on the GPU, `__umulh` on the MSVC host).

### 4.2 Modular addition `gl_add`

```cpp
GL_HD uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a) s += GL_EPS;       // carry out of 64 bits -> add 2^64 == EPS (mod p)
    if (s >= GL_P) s -= GL_P;      // canonicalize down into [0, p)
    return s;
}
```

If `a + b` overflows 64 bits, the true sum is `(wrapped) + 2^64`. Since `2^64 == EPS (mod p)`, we recover correctness by adding `EPS`. Then one conditional subtract brings the result into `[0, p)`. The detection `s < a` is the standard unsigned-overflow test.

### 4.3 Modular subtraction `gl_sub`

```cpp
GL_HD uint64_t gl_sub(uint64_t a, uint64_t b) {
    uint64_t d = a - b;
    if (a < b) d -= GL_EPS;        // borrow -> result is short by EPS, correct it
    return d;
}
```

If `a < b` the bare subtraction borrows (wraps by `+2^64`). The mathematically correct result is `a - b + p`; since `p = 2^64 - EPS`, that equals `(wrapped value) - EPS`. So on borrow we subtract `EPS`.

### 4.4 The Solinas reduction `gl_reduce128` (the heart)

This is the most important function in the whole field. It takes the 128-bit product as two halves `(lo, hi)` and folds it into `[0, p)`. It is the Plonky2 `reduce128` algorithm. The derivation, straight from the comment:

```
x = lo + 2^64 * hi
  = lo + 2^64 * (hi_lo + 2^32 * hi_hi)        [split hi into two 32-bit halves]
  = lo + hi_lo * 2^64 + hi_hi * 2^96
  = lo + hi_lo * EPS   - hi_hi                (mod p)
```

The last line applies the two congruences: `2^64 == EPS` and `2^96 == -1`. Now we just compute `lo + hi_lo*EPS - hi_hi` carefully:

```cpp
GL_HD uint64_t gl_reduce128(uint64_t lo, uint64_t hi) {
    uint32_t hi_hi = (uint32_t)(hi >> 32);          // bits [96,128), weight 2^96 == -1
    uint32_t hi_lo = (uint32_t)(hi & 0xFFFFFFFFu);  // bits [64,96),  weight 2^64 == EPS

    // t0 = lo - hi_hi  (the "2^96 == -1" contribution); borrow -> subtract EPS.
    uint64_t t0 = lo - (uint64_t)hi_hi;
    if (lo < (uint64_t)hi_hi) t0 -= GL_EPS;

    // t1 = hi_lo * EPS  (the "2^64 == EPS" contribution). hi_lo < 2^32 and
    // EPS < 2^32, so the product is < 2^64 and cannot overflow.
    uint64_t t1 = (uint64_t)hi_lo * GL_EPS;

    // t2 = t0 + t1, folding a possible carry (again 2^64 == EPS).
    uint64_t t2 = t0 + t1;
    if (t2 < t0) {                 // carry out of 64 bits
        uint64_t before = t2;
        t2 += GL_EPS;
        if (t2 < before) t2 += GL_EPS; // extremely rare second carry
    }
    while (t2 >= GL_P) t2 -= GL_P;  // canonicalize into [0, p)
    return t2;
}
```

Step by step:

1. Split the high word `hi` into two 32-bit halves. `hi_hi` carries weight `2^96` (which is `-1` mod `p`); `hi_lo` carries weight `2^64` (which is `EPS` mod `p`).
2. `t0 = lo - hi_hi`. Because `hi_hi` enters with weight `-1`, we *subtract* it. On borrow, correct by `-EPS` (same borrow logic as `gl_sub`).
3. `t1 = hi_lo * EPS`. Both factors are `< 2^32`, so this product is `< 2^64` and never overflows a `uint64_t` — a crucial property of the chosen split.
4. `t2 = t0 + t1`, folding any carry by adding `EPS` (with a rare double-fold guard for pathological inputs).
5. Finally a short `while` loop canonicalizes into `[0, p)`. It runs at most a couple of times.

No division anywhere — only shifts, one `32x32 -> 64` multiply, adds, subtracts, and conditional folds.

### 4.5 Modular multiply `gl_mul`

```cpp
GL_HD uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b;            // low 64 bits (wraps mod 2^64 -- exactly right)
    uint64_t hi = gl_mulhi(a, b);   // high 64 bits
    return gl_reduce128(lo, hi);
}
```

Full 128-bit product split into `(lo, hi)`, then `gl_reduce128`. This is the single most-called function in the entire program: every butterfly does one `gl_mul`.

### 4.6 Pow, inverse, and roots of unity

```cpp
GL_HD uint64_t gl_pow(uint64_t base, uint64_t exp) {     // square-and-multiply
    uint64_t result = 1;
    while (exp) {
        if (exp & 1) result = gl_mul(result, base);
        base = gl_mul(base, base);
        exp >>= 1;
    }
    return result;
}

GL_HD uint64_t gl_inv(uint64_t a) {                      // Fermat: a^(p-2) == a^{-1}
    return gl_pow(a, GL_P - 2);
}

GL_HD uint64_t gl_root_of_unity(uint64_t N) {            // omega = g^((p-1)/N)
    return gl_pow(GL_G, (GL_P - 1) / N);
}
```

- `gl_pow` is standard binary exponentiation: `O(log exp)` multiplies.
- `gl_inv` uses **Fermat's little theorem**: for nonzero `a`, `a^(p-2) = a^{-1} (mod p)`. We only ever invert the transform length `N` and roots of unity, all nonzero, so Fermat is the simplest correct choice (no extended-Euclid needed).
- `gl_root_of_unity(N)` returns `7^((p-1)/N)`, the primitive `N`-th root. For `N <= 2^32` the division `(p-1)/N` is **exact** because `2^32 | (p-1)` — and that exactness is precisely what makes `omega` have order exactly `N`.

---

## 5. The 16-bit packing argument — why ONE prime is enough (no CRT)

This is the most subtle and most beautiful engineering decision in the project, so it deserves its own section.

When you convolve digit sequences in `GF(p)`, the field happily computes `c_k mod p`. But we want the *true integer* `c_k`, not its residue. These two coincide **only if the true `c_k` is already smaller than `p`** — otherwise it wraps around and the answer is garbage. So the question is: how big can a convolution coefficient get, and can we keep it under `p`?

We control coefficient size by choosing how many bits each transform digit holds. The code packs the big integer into **16-bit** digits. From [include/ntt.h](../include/ntt.h):

```cpp
inline std::vector<uint64_t> repack_to16(const std::vector<uint32_t>& a, size_t length) {
    std::vector<uint64_t> out(length, 0);
    size_t k = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        out[k++] = a[i] & 0xFFFFu;          // low 16 bits  (digit weight 2^(32i))
        out[k++] = (a[i] >> 16) & 0xFFFFu;  // high 16 bits (digit weight 2^(32i+16))
    }
    return out;
}
```

Each 32-bit limb becomes two 16-bit digits. Now the bound. A convolution coefficient is

```
c_k = sum a_i * b_j   with  a_i, b_j < 2^16,
```

so each product `a_i * b_j < 2^32`. There are at most `N/2` such products contributing to any single `c_k` (the operands together occupy at most `N` slots), giving

```
c_k < (N/2) * 2^32 = N * 2^31.
```

For **any** transform length `N <= 2^32`:

```
c_k < 2^32 * 2^31 = 2^63 < p.
```

Every true coefficient is already below `p`. So the field result *equals* the integer result — nothing wraps — and **a single Goldilocks prime recovers the exact product**. From [include/goldilocks.h](../include/goldilocks.h):

```
//  For any transform length N <= 2^32 this is < 2^63 < p. Because every true
//  coefficient is already smaller than p, the field result EQUALS the integer
//  result -- nothing wraps around -- so one prime recovers the exact product.
//  (Numbers big enough to break this have ~1.6 billion digits; we run out of GPU
//  memory long before that.)
```

### The trade-off, stated honestly

If we had packed **32-bit** digits instead, products would be `< 2^64` and a single `c_k` could reach roughly `N * 2^63`, blowing past `p` almost immediately. To recover the true coefficient you would then need to run the NTT modulo *several* primes and reassemble with the **Chinese Remainder Theorem (CRT)** — more transforms, more code, more chances for bugs. The 16-bit choice sidesteps all of that. The cost, stated in [include/ntt.h](../include/ntt.h):

```
//  The price is 2x as many transform points as a 32-bit packing would use; the
//  payoff is dramatically simpler, obviously-correct code.
```

So: 2x the points (a constant factor), in exchange for a single-prime, no-CRT, obviously-correct multiply that works up to ~1.6 billion digits — far beyond where GPU memory gives out. A very good deal.

### Choosing the transform length

```cpp
inline size_t ntt_length(size_t na, size_t nb) {
    size_t conv_len = 2 * na + 2 * nb;     // (2na + 2nb - 1) rounded a touch high
    return ntt_next_pow2(conv_len);
}
```

Each operand of `na` limbs becomes `2*na` digits, so the linear convolution has up to `2*na + 2*nb - 1` output digits. We round that up to the next power of two (`ntt_next_pow2`), as required for a radix-2 transform. The slightly-high `conv_len` is harmless padding.

---

## 6. The iterative Cooley–Tukey radix-2 NTT

Now the transform itself, `ntt_transform` in [src/ntt_cpu.cpp](../src/ntt_cpu.cpp). It is **in place** — the same array `a` is input and output — and a single `bool inverse` flag selects forward vs. inverse (the inverse also applies the `1/N` scaling at the end).

### Step 1 — bit-reversal permutation

```cpp
for (uint32_t i = 0; i < n; ++i) {
    uint32_t j = ntt_bit_reverse(i, logn);
    if (i < j) std::swap(a[i], a[j]);
}
```

The iterative radix-2 algorithm consumes its inputs in **bit-reversed index order**. Permuting up front (swapping `a[i]` with `a[bitreverse(i)]`, guarded by `i < j` so each pair swaps exactly once) is what lets the butterflies run in place afterward. `ntt_bit_reverse` reverses the low `log2n` bits of an index ([include/ntt.h](../include/ntt.h)):

```cpp
inline uint32_t ntt_bit_reverse(uint32_t i, int log2n) {
    uint32_t r = 0;
    for (int b = 0; b < log2n; ++b) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}
```

### Step 2 — butterfly stages

```cpp
for (size_t len = 2; len <= n; len <<= 1) {
    uint64_t w_len = gl_root_of_unity((uint64_t)len);
    if (inverse) w_len = gl_inv(w_len);

    const size_t half = len >> 1;
    for (size_t i = 0; i < n; i += len) {      // each block of size len
        uint64_t w = 1;                         // twiddle starts at w_len^0 = 1
        for (size_t j = 0; j < half; ++j) {
            uint64_t u = a[i + j];
            uint64_t v = gl_mul(a[i + j + half], w);
            a[i + j]        = gl_add(u, v);     // even output
            a[i + j + half] = gl_sub(u, v);     // odd output
            w = gl_mul(w, w_len);               // advance twiddle: w *= w_len
        }
    }
}
```

We merge size-2 blocks into size-4, size-8, ..., up to size-`N`. There are `log2(N)` stages. At each stage of block size `len`:

- `w_len = gl_root_of_unity(len)` is a primitive `len`-th root (`w_len^len == 1`). For the **inverse** transform we use its inverse, `gl_inv(w_len)`.
- Within each block we walk a **twiddle factor** `w = w_len^j` across the lower half and do the butterfly:

```
a[i+j]        = u + w*v        (even output)
a[i+j+half]   = u - w*v        (odd output)
```

with `u = a[i+j]` and `v = a[i+j+half]`, *all arithmetic in `GF(p)`* via `gl_add` / `gl_sub` / `gl_mul`. The twiddle advances by one multiply per step (`w = gl_mul(w, w_len)`), so each butterfly costs one field multiply. Total work: `O(N log N)` field multiplies.

This is identical in shape to a complex FFT butterfly — the *only* difference is that `+`, `-`, `*` are the Goldilocks field operations and `omega` is an integer root of unity rather than `e^{2*pi*i/N}`.

### Step 3 — inverse normalization (`1/N`)

```cpp
if (inverse) {
    uint64_t ninv = gl_inv((uint64_t)n);
    for (size_t i = 0; i < n; ++i) a[i] = gl_mul(a[i], ninv);
}
```

A forward-then-inverse round trip scales every element by `N`. So the inverse transform must divide by `N` — i.e. multiply by `N^{-1}` in the field. We compute `gl_inv(N)` once and scale every element. (This is the field analogue of the `1/N` in an inverse DFT.)

---

## 7. Putting it together — `mul_ntt_cpu`

The public entry point ties the pipeline together exactly as the header promised:

```cpp
std::vector<uint32_t> mul_ntt_cpu(const std::vector<uint32_t>& a,
                                  const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};            // x * 0 = 0

    const size_t n = ntt_length(a.size(), b.size());  // power-of-two transform size

    std::vector<uint64_t> fa = repack_to16(a, n);     // A as padded 16-bit digits
    std::vector<uint64_t> fb = repack_to16(b, n);     // B as padded 16-bit digits

    ntt_transform(fa, false);                         // A -> spectrum
    ntt_transform(fb, false);                         // B -> spectrum

    for (size_t i = 0; i < n; ++i)                    // pointwise product of spectra
        fa[i] = gl_mul(fa[i], fb[i]);

    ntt_transform(fa, true);                          // spectrum -> convolution coefficients

    return carry_propagate(fa);                       // coefficients -> base-2^32 magnitude
}
```

The six steps, matching the file's `PIPELINE` comment:

1. **Repack** each operand's 32-bit limbs into 16-bit digits (`repack_to16`) — keeps coefficients below `p`.
2. **Zero-pad** both to the common power-of-two length `N` (done inside `repack_to16` via the `length` argument).
3. **Forward-NTT** each operand (`ntt_transform(.., false)`).
4. **Pointwise multiply** the two spectra in `GF(p)` (`gl_mul` in a loop) — this is the convolution theorem at work.
5. **Inverse-NTT** the product spectrum (`ntt_transform(.., true)`), yielding the exact convolution coefficients `c_k`.
6. **Carry-propagate** back into base-`2^32` limbs (`carry_propagate`).

### Carry propagation

The inverse transform hands back exact non-negative integers `c_k < p`, each a coefficient in base `2^16`. `carry_propagate` ([include/ntt.h](../include/ntt.h)) walks low-to-high, adding the running carry, peeling 16 bits per digit, and gluing digit pairs back into 32-bit limbs:

```cpp
inline std::vector<uint32_t> carry_propagate(const std::vector<uint64_t>& conv) {
    std::vector<uint32_t> out;
    out.reserve(conv.size() / 2 + 2);
    uint64_t carry = 0;
    size_t i = 0;
    while (i < conv.size() || carry) {
        uint64_t cur = carry + (i < conv.size() ? conv[i] : 0);
        uint32_t lo16 = (uint32_t)(cur & 0xFFFFu);
        carry = cur >> 16;
        ++i;

        uint64_t cur2 = carry + (i < conv.size() ? conv[i] : 0);
        uint32_t hi16 = (uint32_t)(cur2 & 0xFFFFu);
        carry = cur2 >> 16;
        ++i;

        out.push_back(lo16 | (hi16 << 16));   // assemble one 32-bit limb
    }
    while (!out.empty() && out.back() == 0) out.pop_back(); // normalize
    return out;
}
```

Two 16-bit digits combine into one 32-bit output limb. The carry can be large (each `c_k` is up to ~`2^63`), so it is kept in a 64-bit accumulator and threaded through the whole sweep. A trailing-zero trim normalizes the result. Note this carry pass is inherently **sequential** `O(N)` — cheap compared to the transforms, but worth knowing it does not parallelize like the butterflies do.

---

## 8. Why this is all *exact* — the correctness chain

It is worth restating the full argument that the answer is provably correct, end to end:

1. The convolution coefficients `c_k` are honest non-negative integers, each `< 2^63 < p` (Section 5).
2. The NTT computes the convolution `mod p`. Because every `c_k` is already `< p`, `c_k mod p == c_k` — the field result *is* the integer result.
3. Field arithmetic (`gl_add`, `gl_sub`, `gl_mul` via `gl_reduce128`) is exact integer arithmetic mod `p` — no floating point appears anywhere.
4. `omega = 7^((p-1)/N)` is a genuine order-`N` root of unity because `2^32 | (p-1)` makes the exponent exact; this makes the forward/inverse pair a true invertible transform.
5. Therefore `carry_propagate` receives the exact `c_k` and reconstructs the exact product.

This is why the project can claim *provably correct* digits of pi, and why the CPU and GPU implementations are cross-checked: the test suite multiplies CPU-NTT, GPU-NTT, and schoolbook against each other and demands bit-for-bit agreement (see the CPU header note, and [06_cuda_design.md](06_cuda_design.md)).

---

## 9. From CPU reference to GPU

Everything above is the *reference* implementation. It is correct but single-threaded; the butterfly stages and the pointwise multiply are embarrassingly parallel, which is exactly what a GPU eats for breakfast. The CUDA port keeps the identical math — the very same `goldilocks.h` (that is what `GL_HD` is for) and the same 16-bit packing / carry logic from `ntt.h` — and maps the stages onto thousands of threads.

How the butterflies become CUDA kernels, how memory is laid out and coalesced, how twiddles are handled per stage, and how the host orchestrates it all is the subject of the next document:

- **Next:** [06_cuda_design.md](06_cuda_design.md) — the CUDA design.
- **Back:** [04_pi_algorithms.md](04_pi_algorithms.md) · [02_binary_splitting.md](02_binary_splitting.md)

---

### Quick reference — the field at a glance

| Symbol | Value | Meaning |
| --- | --- | --- |
| `GL_P` | `2^64 - 2^32 + 1` = `0xFFFFFFFF00000001` | the Goldilocks prime `p` |
| `GL_EPS` | `2^32 - 1` = `0xFFFFFFFF` | `== 2^64 mod p`; the Solinas fold constant |
| `GL_G` | `7` | smallest primitive root of `GF(p)` |
| two-adicity | `32` | max power-of-two transform length is `2^32` |
| packing | 16-bit digits | keeps every `c_k < 2^63 < p` → single prime, no CRT |
| `omega` | `7^((p-1)/N)` | primitive `N`-th root of unity (`gl_root_of_unity`) |
| `2^64 mod p` | `EPS` | first Solinas congruence |
| `2^96 mod p` | `-1` | second Solinas congruence |
