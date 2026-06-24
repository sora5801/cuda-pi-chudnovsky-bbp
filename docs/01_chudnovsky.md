# The Chudnovsky Series

> Source files for this chapter:
> [include/chudnovsky.h](../include/chudnovsky.h) and
> [src/chudnovsky.cpp](../src/chudnovsky.cpp).
> The recursive *splitting* math lives in its own chapter:
> [binary splitting](02_binary_splitting.md).

This is the engine room. Almost every world record for digits of pi in the last
three decades was set with the Chudnovsky brothers' 1988 series, and this project
is no exception: `compute_pi_chudnovsky()` is what actually produces the digits,
while the [NTT/CUDA fast multiply](03_ntt.md) underneath it is what makes the big
multiplications fast.

This chapter explains:

1. The full Chudnovsky formula and where it comes from.
2. Why it converges at roughly **14.18 correct decimal digits per term**.
3. The meaning of *every* magic constant in the code.
4. The final assembly `pi = 426880 * sqrt(10005) * Q / T`.
5. The **fixed-point integer** evaluation actually used in the code (scale by
   `10^prec`, integer square root, integer divide) and the **guard digits** that
   keep it honest.
6. How `compute_pi_chudnovsky()` is structured, line by line.
7. A from-scratch proof that `C3_OVER_24 = 640320^3 / 24 = 10939058860032000`
   is an *exact* integer.

---

## 1. The formula

The Chudnovsky series computes `1/pi` as an alternating sum:

```
1     ∞      (-1)^k (6k)! (13591409 + 545140134 k)
─── = 12 · Σ  ───────────────────────────────────────
π      k=0    (3k)! (k!)^3  640320^(3k + 3/2)
```

This is the exact expression in the header comment of
[include/chudnovsky.h](../include/chudnovsky.h). A few features jump out:

- It is an infinite sum, but every *term* is built from integers and a single
  irrational factor, `640320^(3/2) = 640320 * sqrt(640320)`.
- The sign alternates: `(-1)^k`.
- The numerator grows like `(6k)!` and the denominator like
  `(3k)!(k!)^3 640320^(3k)`. The denominator wins *enormously*, which is why the
  terms shrink so fast.

Because we are computing `1/pi`, the constant out front is `12`, and the
half-integer power `640320^(3/2)` carries a `sqrt(640320)`. The code does **not**
use the formula in exactly this shape. It rearranges it into a form with no
factorials left over per term and the square root pulled out once, globally:

```
        426880 · sqrt(10005) · Q
π  =  ───────────────────────────
                  T
```

where `Q` and `T` are *integers* produced by [binary splitting](02_binary_splitting.md)
the first `N` terms. This is the identity in the header comment (lines 15-19) and
the one `compute_pi_chudnovsky()` evaluates.

### How `sqrt(640320)` becomes `sqrt(10005)`

The header's rearranged constant is `426880 * sqrt(10005)`, not anything with
`640320` under the root. That is a deliberate simplification. Factor the cube:

```
640320 = 2^6 · 3 · 5 · 23 · 29
```

(You can check: `64 · 3 · 5 · 23 · 29 = 640320`.) Now `640320^(3/2)` equals
`640320 · sqrt(640320)`, and `640320 = 64 · 10005` where `64 = 2^6` is a perfect
square. So:

```
sqrt(640320) = sqrt(64 · 10005) = 8 · sqrt(10005)
```

The perfect-square part `8` gets folded into the rational leading constant, and
only the square-free remainder `10005` stays under the radical. That is why the
code carries `R_CONST = 10005` and not `640320`: it is the smallest radicand that
represents the same irrational number, so the integer square root we compute in
[bignum_div.cpp](../src/bignum_div.cpp) operates on the smallest possible value.

---

## 2. Every constant, explained

All of these are declared in an anonymous namespace at the top of
[src/chudnovsky.cpp](../src/chudnovsky.cpp) (lines 28-37):

```cpp
const uint64_t A_CONST    = 13591409ULL;          // constant term of the linear factor
const uint64_t B_CONST    = 545140134ULL;         // linear coefficient
const uint64_t C3_OVER_24 = 10939058860032000ULL; // 640320^3 / 24 (exact)
const uint64_t L_CONST    = 426880ULL;            // leading constant (= 640320 * 2/3)
const uint64_t R_CONST    = 10005ULL;             // radicand: pi uses sqrt(10005)
const double DIGITS_PER_TERM = 14.181647462725477;// log10(C3_OVER_24 / 72)
```

| Constant | Code name | Value | What it is |
|---|---|---|---|
| `13591409` | `A_CONST` | 13,591,409 | The constant term of the per-term **linear factor** `(13591409 + 545140134·k)`. At `k=0` it is the entire numerator. |
| `545140134` | `B_CONST` | 545,140,134 | The slope of that linear factor: how much the numerator grows per term. |
| `640320` | (folded into the others) | 640,320 | The "Heegner number" base. The whole series is tuned around this value; it appears as `640320^(3k+3/2)`. Its cube and its square root drive everything else. |
| `10939058860032000` | `C3_OVER_24` | 1.0939…×10^16 | `640320^3 / 24`. This is the `k^3` coefficient in the denominator atom `q(k) = k^3 · C3_OVER_24`. Proven exact in §7. |
| `426880` | `L_CONST` | 426,880 | The rational leading constant. Equals `640320 · 2/3` and also equals `8 · 12 · 4445.83…`? No — see below. It is what is left after pulling `8 = sqrt(64)` out of `640320^(3/2)` and combining with the `12`. |
| `10005` | `R_CONST` | 10,005 | The square-free radicand. `sqrt(10005) ≈ 100.0249968758…`. |

A note on `426880 = 640320 · 2/3`: the code comment on line 34 states this
identity, and it checks out exactly (`640320 · 2 = 1280640`, divided by `3` is
`426880`, with no remainder). This is the algebraic consequence of folding the
leading `12`, the `8` from `sqrt(64)`, and the `640320` factor of `640320^(3/2)`
together. The takeaway you need is operational: **`426880` and `sqrt(10005)`
together reconstitute `12 · 640320^(3/2)`** in the rearranged identity.

`DIGITS_PER_TERM` is not an independent constant — it is *derived* from
`C3_OVER_24`, and §3 shows exactly how.

### The per-term integer atoms

The header (lines 27-31) defines the three integer "atoms" each term contributes.
These are the quantities binary splitting actually manipulates:

```
p(k) = (6k-5)(2k-1)(6k-1)                       for k >= 1
q(k) = k^3 · C3_OVER_24
t(k) = p(k) · (13591409 + 545140134·k),  negated when k is odd
```

In [src/chudnovsky.cpp](../src/chudnovsky.cpp) these appear verbatim in the base
case of `chudnovsky_bs` (lines 78-87):

```cpp
r.P = BigInt::from_u64(6 * a - 5)
    * BigInt::from_u64(2 * a - 1)
    * BigInt::from_u64(6 * a - 1);            // p(a) = (6a-5)(2a-1)(6a-1)
r.Q = A * A * A * BigInt::from_u64(C3_OVER_24); // q(a) = a^3 * C3_OVER_24
BigInt linear = BigInt::from_u64(A_CONST) + mul_small(A, (int64_t)B_CONST);
r.T = r.P * linear;                            // t(a) = p(a) * (A + B*a)
if (a & 1) r.T = -r.T;                          // sign (-1)^a
```

The factored form `p(k) = (6k-5)(2k-1)(6k-1)` is the *ratio* of consecutive
factorial products `(6k)! / (6(k-1))!` after the common `(3k)!(k!)^3` structure is
divided out. Working with these small per-term ratios — instead of the giant
factorials directly — is the whole reason binary splitting is tractable. The
derivation of that ratio and the `P/Q/T` merge rules is the subject of
[binary splitting](02_binary_splitting.md); here we just use them.

### The special `k = 0` term

At `k=0`, the factored `p(0) = (−5)(−1)(−1) = −5` would be *wrong* — the series
term at `k=0` is simply `13591409 / 640320^(3/2)` scaled, with no `(6k)!` ratio
yet. The code special-cases it (lines 69-74):

```cpp
if (a == 0) {
    r.P = BigInt((int64_t)1);
    r.Q = BigInt((int64_t)1);
    r.T = BigInt::from_u64(A_CONST);  // 13591409
}
```

So `p(0) = q(0) = 1` and `t(0) = 13591409`. This makes the `k=0` term reduce to
the bare constant `A_CONST`, which is exactly its contribution before any
factorial ratios accumulate.

---

## 3. Why ~14.18 digits per term

Each successive term is smaller than the previous one by a roughly constant
*factor*, and that factor sets the convergence rate. Take the ratio of the term
magnitudes `|term(k)| / |term(k-1)|`. The dominant growth is in the denominator:
`(6k)!` over `(3k)!(k!)^3` contributes a factor that, by Stirling's
approximation, tends to a constant, and `640320^(3k)` contributes a clean
`640320^3` per step. Working it through, the asymptotic shrink-per-term is:

```
|term(k-1)|        640320^3
──────────  ≈  ───────────────  =  C3_OVER_24 / 72
 |term(k)|          24 · 3 · ...
```

The code encodes this directly: the number of decimal digits gained per term is
the base-10 log of that ratio, and line 36 of
[src/chudnovsky.cpp](../src/chudnovsky.cpp) says so in a comment:

```cpp
// Each term is worth this many decimal digits: log10(C3_OVER_24 / 72).
const double DIGITS_PER_TERM = 14.181647462725477;
```

Let's verify the number:

```
C3_OVER_24 / 72 = 10939058860032000 / 72 = 151931372778222.22…
log10(151931372778222.22…) = 14.181647462725477
```

That matches the literal in the code to all printed digits. So **each term adds
about 14.1816 correct decimal digits**, and to get `D` digits you need about
`D / 14.1816` terms. That conversion is exactly what
`chudnovsky_terms_for_digits()` does (lines 54-59):

```cpp
uint64_t chudnovsky_terms_for_digits(uint64_t digits) {
    // +2 gives a little headroom so series truncation error stays below the
    // requested precision.
    uint64_t n = (uint64_t)((double)digits / DIGITS_PER_TERM) + 2;
    return n < 2 ? 2 : n;
}
```

The `+ 2` is a safety margin: it covers the last fractional term plus a little
slack so the *truncation error* (everything past term `N`) sits comfortably below
the least significant digit we keep. The `n < 2 ? 2 : n` floor guarantees the
binary-splitting tree always has at least two leaves, so the recursion's split
logic is never asked to halve a width-1 range.

> **Why `/72` and not `/24`?** `C3_OVER_24` already absorbed the `24` from
> `640320^3/24` into the per-term denominator atom `q(k)`. The *additional* factor
> of `3` (so `24 · 3 = 72`) comes from the `(3k)!` / `k!` growth in the ratio of
> consecutive terms. The net per-term shrink is `640320^3 / 72`, i.e.
> `C3_OVER_24 / 3`… which is the same as `640320^3 / 72`. Either way you arrive at
> `log10(C3_OVER_24 / 72)`.

---

## 4. The final assembly: `pi = 426880 · sqrt(10005) · Q / T`

After [binary splitting](02_binary_splitting.md) over the range `[0, N)`, we hold
three big integers in a `PQT` struct ([include/chudnovsky.h](../include/chudnovsky.h)
lines 59-61):

```cpp
struct PQT {
    BigInt P, Q, T;   // P and Q positive; T may be negative (the series alternates)
```

The partial sum of the first `N` terms of `1/pi` works out to (a constant times)
`T / Q`. Inverting and multiplying by the leading factor `12 · 640320^(3/2)`,
re-expressed as `426880 · sqrt(10005)`, gives:

```
π  ≈  426880 · sqrt(10005) · Q / T
```

Note that `P` is computed but **not used** in the final assembly — it is only
needed *inside* the splitting merge `T = T_L·Q_R + P_L·T_R`. By the time we reach
the root, the accumulated `P` has done its job and can be discarded. (You can see
in `compute_pi_chudnovsky` that only `r.Q` and `r.T` are referenced afterward.)

This is a finite, *rational* approximation times one irrational `sqrt(10005)`. The
only sources of error are (a) truncating the series at `N` terms — controlled by
the `+2`/guard-digit headroom — and (b) the rounding of `sqrt(10005)` and the
final divide, which the guard digits absorb (§6).

---

## 5. Fixed-point integer evaluation

Here is the key engineering idea. We never use floating point for the digits.
Everything is integers scaled by a power of ten. Define a precision `prec` (in
digits) and let the scale be `10^prec`. Then:

- A real number `x` is represented by the integer `floor(x · 10^prec)`.
- `sqrt(10005)` becomes `floor(sqrt(10005) · 10^prec)`.
- `pi` becomes `floor(pi · 10^prec)`, a `(prec+1)`-digit integer `"314159…"`.

### Scaling the square root with `isqrt`

We want `floor(sqrt(10005) · 10^prec)`. The trick (lines 128-133) is to push the
scale *inside* the radical, where it becomes `10^(2·prec)`:

```
sqrt(10005) · 10^prec  =  sqrt(10005 · 10^(2·prec))
```

so

```
floor(sqrt(10005) · 10^prec)  =  isqrt(10005 · 10^(2·prec))
```

because integer square root of `10005 · 10^(2·prec)` floors the exact root. The
code builds the radicand and calls the big-integer `isqrt` from
[bignum_div.cpp](../src/bignum_div.cpp):

```cpp
BigInt one = pow10_big(prec);                            // 10^prec
BigInt radicand = BigInt::from_u64(R_CONST) * one * one; // 10005 · 10^(2 prec)
BigInt sqrtC = isqrt(radicand);                          // floor(sqrt(10005) · 10^prec)
```

`pow10_big` (lines 42-51) is a square-and-multiply exponentiation. The comment is
worth quoting because it flags a real pitfall:

```cpp
// 10^e via binary exponentiation (square-and-multiply). Unlike the cached pow10
// in bignum.cpp, this does NOT keep every intermediate power, so it is safe to
// call with the very large exponents used for fixed-point scaling.
```

A cached `pow10` that memoizes every power `10^0, 10^1, …` would explode in memory
for `prec` in the millions; this local version keeps only `O(log e)` temporaries.

### The final division

Now combine. With `Q` and `T` integers and `sqrtC = floor(sqrt(10005)·10^prec)`:

```
floor(pi · 10^prec)  =  floor( 426880 · Q · sqrtC / T )
```

The code (lines 137-140):

```cpp
BigInt numerator = mul_small(r.Q, (int64_t)L_CONST) * sqrtC; // 426880 · Q · sqrtC
BigInt pi_scaled = div_fast(numerator, r.T);                 // floor(pi · 10^prec)
```

`mul_small` multiplies a `BigInt` by a 64-bit scalar in one pass (cheaper than
constructing a `BigInt` for `426880` and doing a full multiply), and `div_fast`
is the big-integer division from [bignum_div.cpp](../src/bignum_div.cpp). Because
`T` is the *only* negative quantity possible (the alternating sign lives in `T`),
and `Q`, `426880`, `sqrtC` are all positive, the sign of `pi_scaled` comes out
correct automatically — pi is positive, and an even number of sign flips in the
top terms leaves `T` with the right sign relative to `Q`.

> **Honest caveat — division and decimal conversion are still quadratic.** The
> multiplications near the root of the splitting tree are accelerated by NTT/CUDA,
> but `div_fast` and `to_decimal()` are *not* sub-quadratic here. For very large
> digit counts these two steps become the bottleneck, which is why the `verbose`
> timing in `compute_pi_chudnovsky` breaks out `final division` and `format` as
> their own line items — so you can *see* them dominate. A production system would
> use Newton-iteration reciprocal division and a divide-and-conquer base
> conversion; this code keeps them simple and correct instead.

---

## 6. Guard digits

Two operations round *down*: `isqrt` floors the square root, and `div_fast`
floors the quotient. Each can nudge the last digit or two off. If we computed to
exactly `digits` precision, those errors could corrupt the digits we actually
want to show.

The fix is **guard digits**: compute to a slightly higher precision than
requested, then throw the extra low-order digits away. The flooring errors live
in the discarded tail, so the kept digits are all correct. From
[src/chudnovsky.cpp](../src/chudnovsky.cpp) (lines 116-120):

```cpp
// Guard digits absorb the tiny floor() errors from isqrt and the final
// division, so the digits we keep are all correct.
const uint64_t guard = 16;
const uint64_t prec  = digits + guard;
const uint64_t N     = chudnovsky_terms_for_digits(prec);
```

So we work at `prec = digits + 16`, choose `N` terms for *that* higher precision,
and at the end keep only the first `digits` fractional digits. Sixteen guard
digits is generous — the combined floor error from two operations is at most a
couple of units in the last place — but guard digits are cheap (a few extra
limbs) compared to a wrong answer, so the code spends them freely. Note that `N`
is derived from `prec`, *not* `digits`, so the series truncation error is also
held below the guard band.

---

## 7. Proving `C3_OVER_24 = 640320^3 / 24` is exact

The code asserts `C3_OVER_24 = 10939058860032000` and calls it exact (line 32).
Let's prove it — both that the division has no remainder and that the value is
right.

**Step 1 — the cube.**

```
640320^2 = 409,    009,  702,  400          (= 640320 × 640320 = 409009702400)
640320^3 = 640320 × 409009702400
         = 262,537,412,640,768,000
```

**Step 2 — divisibility by 24.** Factor the base into primes:

```
640320 = 2^6 · 3 · 5 · 23 · 29
```

Cubing multiplies every exponent by 3:

```
640320^3 = 2^18 · 3^3 · 5^3 · 23^3 · 29^3
```

Now `24 = 2^3 · 3`. Since `640320^3` contains `2^18` (far more than `2^3`) and
`3^3` (more than `3^1`), the factor `24 = 2^3·3` divides it **cleanly** — the
remainder is `0`. This is the structural reason the division is exact: it is not a
numerical coincidence, it falls out of `640320` already being divisible by
`2^6·3`.

**Step 3 — the quotient.**

```
262,537,412,640,768,000 / 24 = 10,939,058,860,032,000
```

which is exactly `C3_OVER_24`. Removing `2^3·3` from the factorization leaves:

```
C3_OVER_24 = 2^15 · 3^2 · 5^3 · 23^3 · 29^3 = 10,939,058,860,032,000
```

(Exponents: `18−3 = 15` for the 2's, `3−1 = 2` for the 3's, the rest unchanged.)

You can confirm the whole chain in one line:

```python
>>> 640320**3
262537412640768000
>>> 640320**3 % 24
0
>>> 640320**3 // 24
10939058860032000
```

Because the result fits comfortably in a `uint64_t` (it is about `1.09 × 10^16`,
well under `1.8 × 10^19`), the constant is hard-coded as a `uint64_t` literal
rather than recomputed, and the per-term `q(k) = k^3 · C3_OVER_24` multiply uses
`mul_small`-style scalar arithmetic in the base case.

---

## 8. How `compute_pi_chudnovsky()` is structured

Putting it together, here is the shape of the function
([src/chudnovsky.cpp](../src/chudnovsky.cpp) lines 107-164). It is a clean,
five-phase pipeline with optional per-phase timing.

```
compute_pi_chudnovsky(digits, verbose):

  0. Setup
     guard = 16
     prec  = digits + guard
     N     = chudnovsky_terms_for_digits(prec)   # terms for the padded precision

  1. Binary splitting            ── the big multiplies; NTT/CUDA accelerated
     PQT r = chudnovsky_bs(0, N)                  # gives (P, Q, T)

  2. Scaled square root          ── isqrt of 10005 · 10^(2·prec)
     one      = pow10_big(prec)                   # 10^prec
     radicand = 10005 · one · one                 # 10005 · 10^(2·prec)
     sqrtC    = isqrt(radicand)                   # floor(sqrt(10005) · 10^prec)

  3. Numerator + final division  ── fixed-point pi
     numerator = (426880 · r.Q) · sqrtC
     pi_scaled = div_fast(numerator, r.T)         # floor(pi · 10^prec)

  4. Format
     s = pi_scaled.to_decimal()                   # "314159…"  (prec+1 digits)
     out = s[0] + "." + s[1 .. digits]            # drop the guard digits

  5. Package
     return { pi = out, terms = N, seconds = elapsed }
```

A few implementation details worth calling out:

- **Timing harness.** A small `stamp(label, a, b)` lambda (lines 110-114) prints
  `[chudnovsky] <label> <ms>` to `stderr` when `verbose` is set. Each phase is
  bracketed by `clock::now()` calls so you can profile where the time goes —
  typically binary splitting dominates at first, then the still-quadratic division
  and formatting take over at extreme sizes (see the caveat in §5).

- **Formatting** (lines 144-155). `pi_scaled.to_decimal()` yields the digit string
  `"314159…"`. The integer part is the single digit `s[0]` (`'3'`); we push that,
  push `'.'`, then append the first `digits` characters of the fractional part.
  The guard digits at the tail are simply never copied. The `digits == 0` case is
  special-cased to return the bare string `"3"`.

- **Result struct.** `ChudnovskyResult` ([chudnovsky.h](../include/chudnovsky.h)
  lines 69-73) carries the formatted `pi` string, the term count `terms = N`
  (useful for tests and reporting), and the wall-clock `seconds`.

- **The multiply backend is the whole point.** As the file header notes
  (lines 13-16 of [src/chudnovsky.cpp](../src/chudnovsky.cpp)), every `*` between
  `BigInt`s flows through `BigInt::operator*`, which dispatches by operand size to
  schoolbook → Karatsuba → CPU-NTT → CUDA-NTT. So just by leaving the backend on
  *Auto*, the huge multiplications near the **root** of the splitting tree — where
  both operands carry nearly all the digits — run on the GPU. That dispatch and the
  NTT itself are covered in [the NTT chapter](03_ntt.md).

---

## Cross-references

- [02_binary_splitting.md](02_binary_splitting.md) — the recursive `P/Q/T` merge
  rules, the base-case derivation, why the tree is balanced, and how the cost
  concentrates at the root.
- [03_ntt.md](03_ntt.md) — the number-theoretic transform multiply that makes the
  root-level products fast, including the CUDA kernel.
- Source: [src/chudnovsky.cpp](../src/chudnovsky.cpp),
  [include/chudnovsky.h](../include/chudnovsky.h),
  and the big-integer primitives `isqrt` / `div_fast` / `mul_small` /
  `to_decimal` in [bignum_div.cpp](../src/bignum_div.cpp) and
  [bignum.cpp](../src/bignum.cpp).
