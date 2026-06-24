# 04 — The Big-Integer Library (`BigInt`)

To compute pi to millions of digits we must do **exact** integer arithmetic on
numbers that are millions of bits long. A 64-bit `unsigned long long` holds only
about 19 decimal digits; a one-million-digit number needs ~3.3 million bits, or
~52,000 of our 32-bit "limbs." No built-in C++ type can hold that, so this
project ships its own big integer, `pidigits::BigInt`.

Everything sits on top of this type: the [binary splitting](02_binary_splitting.md)
recursion is nothing but a tree of `BigInt` multiplies and adds, and the
[Chudnovsky](03_chudnovsky.md) final assembly needs a square root, a division,
and a base-10 conversion. The performance of the *whole program* is dominated by
how fast we can **multiply** two enormous integers — which is exactly why the
fast multiply backends ([NTT on the GPU](05_ntt_goldilocks.md)) get their own
files.

This document covers the type as implemented in:

- [`include/bignum.h`](../include/bignum.h) — the declarations and the long design comments.
- [`src/bignum.cpp`](../src/bignum.cpp) — construction, add/sub, shifts, schoolbook & Karatsuba multiply, the dispatcher, decimal parsing.
- [`src/bignum_div.cpp`](../src/bignum_div.cpp) — Knuth division, integer square root, decimal output.

---

## 1. Representation: sign-magnitude, base 2³², little-endian

The type is small and blunt:

```cpp
class BigInt {
public:
    std::vector<uint32_t> mag; // little-endian base-2^32 magnitude, normalized
    int sign = 0;              // -1, 0, or +1; 0 iff mag is empty
    ...
};
```

### Magnitude

The magnitude is stored as a `std::vector<uint32_t>` called `mag`. Each element
is a **limb** — one "digit" in a positional number system whose base is
`B = 2^32 = 4294967296`. The vector is **little-endian**: `mag[0]` is the *least*
significant limb, `mag[1]` the next 32 bits up, and so on. The mathematical value
of the magnitude is

```
|value| = sum over i of  mag[i] * (2^32)^i
```

Memory-wise, a `BigInt` is just a 24-byte `std::vector` header (pointer, size,
capacity) plus the heap buffer of limbs, plus a 4-byte `sign`. The limbs are
contiguous, which is what lets the multiply backends slice the buffer, reinterpret
it as 16-bit words for the NTT, or hand a raw pointer to CUDA.

**Why base 2³²?** The header spells out the reasoning ([`bignum.h`](../include/bignum.h)):

- A product of two 32-bit limbs fits in a 64-bit register (`32 + 32 = 64` bits),
  so schoolbook multiplication never overflows a `uint64_t` accumulator — even
  after adding a column's worth of carries (`(2^32-1)^2 + 2*(2^32-1) < 2^64`).
- Bit shifts (needed by Newton's method for square root) are natural in a
  power-of-two base.
- Repacking into the 16-bit words the NTT wants is a trivial split, not a
  base conversion.

### Sign

`sign ∈ {-1, 0, +1}`, and it is `0` **if and only if** the magnitude is empty
(the number is zero). This **sign-magnitude** scheme deliberately keeps the
magnitude code — the hard, performance-critical part — completely sign-agnostic.
The handful of places that care about sign (add, sub, compare, multiply) handle
it explicitly and briefly, e.g. the multiply sign rule is the one-liner
`r.sign = a.sign * b.sign;`.

---

## 2. The normalization invariant

There is exactly one shape rule, enforced everywhere:

> **No leading zero limbs.** The most-significant limb `mag.back()` is always
> non-zero, *except* for zero itself, whose `mag` is empty. And `sign == 0` iff
> `mag` is empty.

The private helper `trim` does the magnitude half of the job:

```cpp
void trim(std::vector<uint32_t>& v) {
    while (!v.empty() && v.back() == 0) v.pop_back();
}
```

and `BigInt::normalize()` ties the magnitude and sign together:

```cpp
void BigInt::normalize() {
    trim(mag);
    if (mag.empty()) sign = 0;             // magnitude zero forces sign zero
    else if (sign == 0) sign = 1;          // non-empty magnitude must have a sign
}
```

Every operation that *builds* a `BigInt` ends with a call to `normalize()` (or,
for the magnitude-only helpers, a call to `trim`). Why bother? Two reasons:

1. **Comparison can be O(1) on length.** Because trimmed vectors have no leading
   zeros, `cmp_mag_vec` compares lengths first and only walks limbs when the
   lengths tie:

   ```cpp
   int cmp_mag_vec(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
       if (a.size() != b.size()) return a.size() < b.size() ? -1 : +1;
       for (size_t i = a.size(); i-- > 0;)            // i goes high..low
           if (a[i] != b[i]) return a[i] < b[i] ? -1 : +1;
       return 0;
   }
   ```

2. **There is a unique representation of every value.** No `+0`/`-0` ambiguity,
   no "is this `[5,0,0]` equal to `[5]`?" surprises. Tests can compare results
   limb-for-limb.

---

## 3. Addition and subtraction (carry / borrow)

The signed operators reduce to two magnitude-only helpers. All the genuinely
tricky bit-twiddling lives in the magnitude layer.

### Magnitude add — ripple carry

```cpp
std::vector<uint32_t> add_mag(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b) {
    const std::vector<uint32_t>& big   = a.size() >= b.size() ? a : b;
    const std::vector<uint32_t>& small = a.size() >= b.size() ? b : a;
    std::vector<uint32_t> out;
    out.reserve(big.size() + 1);                     // +1 for a possible final carry
    uint64_t carry = 0;
    for (size_t i = 0; i < big.size(); ++i) {
        uint64_t sum = (uint64_t)big[i] + carry + (i < small.size() ? small[i] : 0);
        out.push_back((uint32_t)sum);                // low 32 bits stay here
        carry = sum >> 32;                           // high bits carry to next column
    }
    if (carry) out.push_back((uint32_t)carry);       // final carry -> new top limb
    return out;
}
```

The whole algorithm is grade-school column addition. The one engineering point is
the `uint64_t sum` accumulator: a 32-bit limb plus a 32-bit limb plus a 1-bit
carry is at most `2^33 - 1`, which fits comfortably in 64 bits, so the low 32 bits
are the column digit and `sum >> 32` (always 0 or 1) is the carry. No leading zero
can appear, so `add_mag` needs no `trim`.

### Magnitude subtract — ripple borrow, requires `a >= b`

```cpp
std::vector<uint32_t> sub_mag(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b) {
    std::vector<uint32_t> out;
    out.reserve(a.size());
    uint64_t borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t bi = (i < b.size() ? b[i] : 0);
        // Add 2^32 before subtracting so the result is always a valid
        // non-negative 64-bit number; the carry-out tells us if we borrowed.
        uint64_t diff = (uint64_t)a[i] + 0x100000000ULL - bi - borrow;
        out.push_back((uint32_t)diff);               // low 32 bits = column result
        borrow = (diff >> 32) ? 0 : 1;               // no overflow happened => we borrowed
    }
    trim(out);                                       // 0x...1 - 0x...1 can leave high zeros
    return out;
}
```

`sub_mag` is the one helper with a **precondition**: the caller guarantees
`a >= b`. The trick is to pre-add the base `0x100000000` (= 2³²) to each column
before subtracting `bi` and the incoming `borrow`. That keeps `diff` in valid
unsigned range no matter what. If the column genuinely did *not* need to borrow,
`diff` overflowed past 2³² and `diff >> 32` is 1; if it *did* need to borrow,
`diff < 2^32` and the shifted value is 0 — hence the inverted-looking
`borrow = (diff >> 32) ? 0 : 1`. Because `a >= b`, the final borrow is provably 0.
Unlike addition, subtraction *can* create leading zeros (e.g. when the top limbs
cancel), so `sub_mag` ends in `trim`.

### Putting the sign back on

`operator+` does a short case analysis and delegates:

```cpp
BigInt operator+(const BigInt& a, const BigInt& b) {
    if (a.sign == 0) return b;
    if (b.sign == 0) return a;
    BigInt r;
    if (a.sign == b.sign) {                  // same sign: magnitudes add
        r.mag = add_mag(a.mag, b.mag);
        r.sign = a.sign;
    } else {                                 // opposite signs: magnitudes subtract
        int c = a.cmp_mag(b);
        if (c == 0) return BigInt();         // exact cancellation -> zero
        if (c > 0) { r.mag = sub_mag(a.mag, b.mag); r.sign = a.sign; }
        else       { r.mag = sub_mag(b.mag, a.mag); r.sign = b.sign; }
    }
    r.normalize();
    return r;
}
```

The rule is the one you learned by hand: equal signs add magnitudes and keep the
sign; opposite signs subtract the smaller magnitude from the larger and take the
sign of the larger. Note how the `cmp_mag` check guarantees `sub_mag`'s
`a >= b` precondition before either call. Subtraction is then trivially
`a - b == a + (-b)`:

```cpp
BigInt operator-(const BigInt& a, const BigInt& b) { return a + (-b); }
```

---

## 4. Schoolbook multiplication — O(n·m)

This is the literal definition of multiplication and the project's **golden
reference**: every faster algorithm is tested against it.

```cpp
std::vector<uint32_t> mul_schoolbook(const std::vector<uint32_t>& a,
                                     const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};           // anything * 0 = 0
    std::vector<uint32_t> out(a.size() + b.size(), 0); // product fits in na+nb limbs
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t carry = 0;
        uint64_t ai = a[i];
        for (size_t j = 0; j < b.size(); ++j) {
            uint64_t cur = (uint64_t)out[i + j] + ai * (uint64_t)b[j] + carry;
            out[i + j] = (uint32_t)cur;               // keep low 32 bits in place
            carry = cur >> 32;                        // carry the rest upward
        }
        out[i + b.size()] += (uint32_t)carry;         // deposit the final carry
    }
    trim(out);
    return out;
}
```

The product of two limbs lands in column `i + j`. The key safety argument is the
one from §1: the running accumulator `out[i+j] + ai*b[j] + carry` can be as large
as `(2^32-1) + (2^32-1)^2 + (2^32-1) = 2^64 - 2^32`, which still fits in a
`uint64_t` — so the inner loop never overflows. The cost is one 64-bit multiply
per pair of limbs, i.e. **O(n·m)**. For the million-limb numbers Chudnovsky
produces, that quadratic blow-up is fatal, which is the entire motivation for the
next two sections and for the NTT.

---

## 5. Karatsuba multiplication — O(n^1.585)

Karatsuba is the classic sub-quadratic trick. Split each operand at limb position
`k` into a low half and a high half:

```
a = a1 * B^k + a0
b = b1 * B^k + b0
```

The naive expansion needs **four** half-size products:

```
a*b = a1*b1 * B^2k + (a1*b0 + a0*b1) * B^k + a0*b0
```

Karatsuba's insight is that the awkward middle term can be recovered from the
other two with just **one** extra multiply, using the 3-multiply identity:

```
z0 = a0*b0
z2 = a1*b1
z1 = (a0+a1)*(b0+b1) - z0 - z2   ==  a1*b0 + a0*b1
```

So **three** half-size multiplies replace four. Recursing, the cost exponent
drops from 2 to `log2(3) ≈ 1.585`. The code mirrors the math exactly:

```cpp
std::vector<uint32_t> mul_karatsuba(const std::vector<uint32_t>& a,
                                    const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};
    size_t na = a.size(), nb = b.size();
    if (std::min(na, nb) < g_mul.karatsuba_threshold)
        return mul_schoolbook(a, b);                  // small inputs: schoolbook wins

    size_t k = std::max(na, nb) / 2;                  // split point (in limbs)
    // low(v) / high(v) slice each operand at k; high is empty if v is short.
    ...
    std::vector<uint32_t> z0 = mul_karatsuba(a0, b0);          // a0*b0
    std::vector<uint32_t> z2 = mul_karatsuba(a1, b1);          // a1*b1
    std::vector<uint32_t> z1 = mul_karatsuba(add_mag(a0, a1),  // (a0+a1)*(b0+b1)
                                             add_mag(b0, b1));
    z1 = sub_mag(z1, z0);                                       // z1 -= z0
    z1 = sub_mag(z1, z2);                                       // z1 -= z2

    // Reassemble: result = z2*B^(2k) + z1*B^k + z0.
    std::vector<uint32_t> out = z0;
    out = add_mag(out, shift_limbs(z1, k));
    out = add_mag(out, shift_limbs(z2, 2 * k));
    trim(out);
    return out;
}
```

A few details worth noticing:

- **Multiplying by `B^k` is free.** `shift_limbs(v, n)` just prepends `n` zero
  limbs — no arithmetic, since the base is a power of the limb width:

  ```cpp
  std::vector<uint32_t> shift_limbs(const std::vector<uint32_t>& a, size_t n) {
      if (a.empty()) return {};
      std::vector<uint32_t> out(a.size() + n, 0);
      std::copy(a.begin(), a.end(), out.begin() + n);
      return out;
  }
  ```

- **Both subtractions are safe.** `z1` (the product of two *sums*) is always at
  least `z0 + z2`, so the two `sub_mag` calls never violate the `a >= b`
  precondition.
- **The base case is schoolbook.** Below `g_mul.karatsuba_threshold` limbs (32 by
  default) the recursion stops and calls `mul_schoolbook`, whose tiny constant
  factor beats Karatsuba's bookkeeping on small inputs.
- The recursion is on the *magnitude* helpers (`add_mag`, `sub_mag`), so it never
  touches sign — Karatsuba is pure unsigned arithmetic.

---

## 6. The multiply dispatcher and its thresholds

User code calls `operator*`, which strips the sign and forwards the magnitudes
into `mul_dispatch`. The dispatcher is the single place that decides *which*
algorithm runs:

```cpp
std::vector<uint32_t> mul_dispatch(const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};
    size_t n = std::min(a.size(), b.size());          // size of the SMALLER operand

    MulBackend backend = g_mul.backend;
    if (backend == MulBackend::Auto) {
        if (n < g_mul.karatsuba_threshold)      backend = MulBackend::Schoolbook;
        else if (n < g_mul.ntt_threshold)       backend = MulBackend::Karatsuba;
        else backend = g_mul.prefer_cuda && cuda_available()
                          ? MulBackend::NttCuda : MulBackend::NttCpu;
    }
    ...
    switch (backend) {
        case MulBackend::Schoolbook: return mul_schoolbook(a, b);
        case MulBackend::Karatsuba:  return mul_karatsuba(a, b);
        case MulBackend::NttCpu:     return mul_ntt_cpu(a, b);
        case MulBackend::NttCuda:    return mul_ntt_cuda(a, b);
        default:                     return mul_schoolbook(a, b);
    }
}
```

The decision is driven by the global `MulConfig g_mul` (declared in
[`bignum.h`](../include/bignum.h)). In **`Auto`** mode the choice is by the
*smaller* operand's limb count — that bounds the inner work — against two
thresholds:

| Smaller operand size (limbs) | Backend chosen      | Asymptotic cost |
| ---------------------------- | ------------------- | --------------- |
| `n < 32` (`karatsuba_threshold`) | `Schoolbook`    | O(n²)           |
| `32 ≤ n < 256` (`ntt_threshold`) | `Karatsuba`     | O(n^1.585)      |
| `n ≥ 256`                        | `NttCuda` if a GPU is present (`prefer_cuda && cuda_available()`), else `NttCpu` | O(n log n) |

The defaults (`karatsuba_threshold = 32`, `ntt_threshold = 256`,
`prefer_cuda = true`) come straight from the `MulConfig` struct. Setting
`g_mul.backend` to anything other than `Auto` **forces** one backend regardless
of size — this is how the test suite runs every algorithm against schoolbook on
the same inputs to prove the fast ones are correct. With `g_mul.verbose = true`
the dispatcher prints a `[mul] AxB limbs via <name>` line per big multiply, which
is handy for confirming the GPU path actually fires on large runs. The NTT
backends themselves are documented in [05 — NTT over Goldilocks](05_ntt_goldilocks.md).

There is also a fast path for "multiply by a machine word," `mul_small`, used
heavily by the binary-splitting leaves (which multiply by small factors like
`(6k-5)(2k-1)(6k-1)`). It calls the internal `mul_word`, a single carry-propagating
pass — far cheaper than building a one-limb `BigInt` and going through the
dispatcher.

---

## 7. Division — Knuth's Algorithm D

Division is the hard one. The project implements the exact, reference long-division
algorithm — **Knuth's Algorithm D** (TAOCP Vol. 2, §4.3.1; the well-known "divmnu"
formulation from *Hacker's Delight*) — in [`src/bignum_div.cpp`](../src/bignum_div.cpp).
The public entry point assumes a non-negative dividend and a positive divisor:

```cpp
void divmod_knuth(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r) {
    assert(a.sign >= 0 && b.sign > 0 && "divmod_knuth expects a >= 0, b > 0");
    ...
}
```

and produces `q = floor(a/b)` and `r = a - q*b` with `0 <= r < b`. The real work
is in the magnitude routine `divmod_mag`. Walking it in order:

### 7.1 Quick exits and single-limb short division

```cpp
if (cmp_mag_vec(u_in, v_in) < 0) { q.clear(); r = u_in; trim(r); return; } // a < b
```

If the divisor is a single limb, full Algorithm D is overkill; a simple
left-to-right short division suffices, bringing down one limb at a time:

```cpp
if (n == 1) {
    uint32_t d = v_in[0];
    q.assign(m, 0);
    uint64_t rem = 0;
    for (size_t i = m; i-- > 0;) {
        uint64_t cur = (rem << 32) | u_in[i];   // bring down next limb
        q[i] = (uint32_t)(cur / d);
        rem = cur % d;
    }
    ...
}
```

This same single-limb loop is the engine behind decimal output (§9).

### 7.2 Normalization — make the top divisor bit set

The accuracy of Algorithm D's quotient-digit estimate depends on the divisor's
leading limb being large. So we shift *both* operands left by `s = nlz32(v[n-1])`
bits — just enough that the divisor's top bit (bit 31) is set:

```cpp
const int s = nlz32(v_in[n - 1]);   // leading-zero count of the divisor's top limb
std::vector<uint32_t> vn(n);
for (size_t i = n - 1; i > 0; --i)
    vn[i] = (v_in[i] << s) | (s ? (uint32_t)((uint64_t)v_in[i - 1] >> (32 - s)) : 0);
vn[0] = v_in[0] << s;
```

The dividend gets the same shift **plus one extra high limb of headroom** (`un`
has `m + 1` limbs). Shifting both `u` and `v` by the same amount does not change
the quotient (`(u·2^s)/(v·2^s) = u/v`), and the remainder is simply un-shifted
back down at the end. With the top bit set, Knuth's theorem guarantees the
estimate `qhat` is at most **2** too large — a bound the next step relies on.

### 7.3 The main loop — estimate, correct, subtract

One quotient limb is produced per iteration, from the most significant down:

```cpp
for (size_t jj = m - n + 1; jj-- > 0;) {
    const size_t j = jj;
    // Estimate this quotient digit from the top two dividend limbs.
    uint64_t num  = ((uint64_t)un[j + n] << 32) | un[j + n - 1];
    uint64_t qhat = num / vn[n - 1];
    uint64_t rhat = num % vn[n - 1];
    // Refine qhat downward until qhat*vn[n-2] <= rhat*B + un[j+n-2].
    while (qhat >= B ||
           qhat * vn[n - 2] > ((rhat << 32) | un[j + n - 2])) {
        --qhat;
        rhat += vn[n - 1];
        if (rhat >= B) break;       // rhat overflowed the base; estimate is fine now
    }
    ...
}
```

**The `qhat` estimate.** Dividing the top *two* dividend limbs by the top *one*
divisor limb gives a first guess for the quotient digit. By itself this guess can
be too big; the `while` loop tests it against the *next* divisor limb
(`vn[n-2]`) and walks `qhat` down. After normalization this loop runs at most a
couple of times, so the estimate is essentially O(1) per digit.

**Multiply-and-subtract.** With `qhat` in hand we subtract `qhat * divisor` from
the current dividend window, tracking a combined borrow/carry `k`:

```cpp
int64_t k = 0;
for (size_t i = 0; i < n; ++i) {
    uint64_t p = qhat * vn[i];
    int64_t t = (int64_t)un[i + j] - k - (int64_t)(p & 0xFFFFFFFFu);
    un[i + j] = (uint32_t)t;
    k = (int64_t)(p >> 32) - (t >> 32); // (t>>32) sign-extends the borrow
}
int64_t t = (int64_t)un[j + n] - k;
un[j + n] = (uint32_t)t;
q[j] = (uint32_t)qhat;
```

The signed `t >> 32` cleverly sign-extends to recover the borrow into the next
column.

**Add-back correction.** Even after refinement, `qhat` can occasionally be exactly
1 too large — detected when the final `t` went negative. The fix is to decrement
the quotient digit and add the divisor back **once**:

```cpp
if (t < 0) {
    --q[j];
    uint64_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t tt = (uint64_t)un[i + j] + vn[i] + carry;
        un[i + j] = (uint32_t)tt;
        carry = tt >> 32;
    }
    un[j + n] = (uint32_t)((uint64_t)un[j + n] + carry);
}
```

This "estimate, occasionally over-shoot, add back" structure is the heart of
Algorithm D. The add-back is rare (it happens with probability ~2/B per digit),
so it costs essentially nothing on average but is *essential* for exactness.

### 7.4 Denormalize the remainder

Finally the remainder living in `un` is shifted back down by the same `s` bits to
undo §7.2's normalization:

```cpp
r.assign(n, 0);
for (size_t i = 0; i < n - 1; ++i)
    r[i] = (un[i] >> s) | (s ? (uint32_t)((uint64_t)un[i + 1] << (32 - s)) : 0);
r[n - 1] = un[n - 1] >> s;
trim(r);
```

### 7.5 The self-checking invariant

Division is its own oracle. For `q = floor(a/b)`, `r = a - q*b`, two facts must
hold:

```
q * b + r == a       AND       0 <= r < b
```

The test suite checks **exactly** this identity, which is why the project trusts
`divmod_knuth` without needing a second, independent divider — any bug in the
quotient or remainder would break one of the two conditions. (This is the same
philosophy used for the square root below, which is cross-checked two ways.)

### 7.6 `div_fast` is a placeholder — and division is still ~quadratic

```cpp
BigInt div_fast(const BigInt& a, const BigInt& b) {
    // Placeholder for a future sub-quadratic divider. Today it forwards to the
    // exact Knuth division so every caller already gets a correct result...
    BigInt q, r;
    divmod_knuth(a, b, q, r);
    return q;
}
```

Be honest about this: **division is still O(n²).** Algorithm D does O(n) work per
quotient limb and there are O(n) limbs, so a full-precision division costs the
same order as schoolbook multiply. `div_fast` exists only so the API is in place
for a future Newton-reciprocal or Burnikel–Ziegler divider to be slotted in
*without touching any callers* — but today it simply calls Knuth. The same is
true of decimal conversion (§9). For the algorithmic plan to make these
sub-quadratic, see [08 — Scaling to billions](08_scaling_to_billions.md).

This matters because the Chudnovsky final assembly does exactly *one* giant
division (and `isqrt` does a handful), so at the multi-million-digit scale the
fast NTT multiply can carry the binary-splitting tree while these quadratic tails
start to dominate the *final* assembly. That trade-off is the subject of doc 08.

---

## 8. Integer square root — Newton, with a bit-by-bit reference

Chudnovsky needs `sqrt(10005)` evaluated at full precision (really
`isqrt(10005 * 10^(2·digits))`). `isqrt(s)` returns `floor(sqrt(s))` via Newton's
method on `x ← (x + s/x) / 2`:

```cpp
BigInt isqrt(const BigInt& s) {
    if (s.sign <= 0) return BigInt();            // sqrt(0)=0; negatives -> 0 by convention
    size_t bits = s.bit_length();
    if (bits <= 2) return BigInt((int64_t)1);    // s in {1,2,3} -> floor sqrt = 1

    // Initial overestimate x0 = 2^(bits/2 + 1) >= sqrt(s) (since s < 2^bits).
    BigInt x = BigInt((int64_t)1).shl(bits / 2 + 1);
    while (true) {
        BigInt q, r;
        divmod_knuth(s, x, q, r);                // q = floor(s / x)
        BigInt y = (x + q).shr(1);               // y = (x + s/x) / 2
        if (y.cmp(x) >= 0) break;                // stopped decreasing -> x is the answer
        x = y;
    }
    return x;
}
```

The subtle correctness facts:

- **Start high.** `x0 = 2^(bits/2 + 1)` is provably `>= sqrt(s)`. From any starting
  point at or above the true root, the Newton iterate (with *floor* division
  throughout) **monotonically decreases** and lands exactly on `floor(sqrt(s))`.
- **Stop when it stops shrinking.** `if (y.cmp(x) >= 0) break;` is the standard
  termination: once the sequence would no longer decrease, the previous `x` is the
  floor of the square root. This avoids oscillating around the answer.
- **Fast convergence, but each step is a division.** Newton doubles the number of
  correct bits per step, so only ~`log2(bits)` iterations are needed — but every
  iteration pays for one `divmod_knuth`, which (per §7.6) is quadratic. So `isqrt`
  inherits division's asymptotics.

Note `shr(1)` is the "divide by 2" here; the project's right-shift asserts a
non-negative operand because flooring semantics for negatives are never needed.

### The bit-by-bit reference

For testing, [`bignum_div.cpp`](../src/bignum_div.cpp) keeps the classic
digit-by-digit base-2 square root, `isqrt_bitwise`, which is slow but *obviously*
correct:

```cpp
BigInt isqrt_bitwise(const BigInt& s) {
    if (s.sign <= 0) return BigInt();
    size_t bits = s.bit_length();
    size_t start = (bits - 1) & ~size_t(1);      // largest even index <= bits-1
    BigInt bit = BigInt((int64_t)1).shl(start);  // highest power-of-four <= s
    BigInt result;                               // 0
    BigInt n = s;                                // running remainder
    while (bit.sign != 0) {
        BigInt rb = result + bit;
        if (n.cmp(rb) >= 0) {
            n = n - rb;
            result = result.shr(1) + bit;
        } else {
            result = result.shr(1);
        }
        bit = bit.shr(2);                        // next lower power of four
    }
    return result;
}
```

The test suite validates the fast `isqrt` two independent ways: against
`isqrt_bitwise`, and against the defining inequality `x² <= s < (x+1)²`. Having a
trivially-correct reference next to the clever implementation is the same
belt-and-suspenders discipline as the schoolbook multiply reference and the
division invariant.

---

## 9. Decimal output — repeated division by 10⁹

The number lives in base 2³²; humans want base 10. `BigInt::to_decimal()`
converts by repeatedly peeling off **9 decimal digits at a time** — `10^9` fits
in a single 32-bit limb (`10^9 < 2^30`), so each peel is the cheap single-limb
short division from §7.1:

```cpp
std::string BigInt::to_decimal() const {
    if (sign == 0) return "0";

    std::vector<uint32_t> cur = mag;             // working copy we destroy as we go
    std::vector<uint32_t> groups;                // base-10^9 "limbs", least significant first
    const uint32_t TEN9 = 1000000000u;           // 10^9

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

    // Assemble most-significant group first (no padding), then 9-digit groups.
    std::string out;
    if (sign < 0) out.push_back('-');
    out += std::to_string(groups.back());
    char buf[16];
    for (size_t i = groups.size() - 1; i-- > 0;) {
        std::snprintf(buf, sizeof(buf), "%09u", groups[i]);
        out += buf;
    }
    return out;
}
```

Each remainder is one group of up to nine decimal digits, produced least-significant
first. Re-assembly is the only place padding matters: the **most-significant**
group is printed bare with `std::to_string` (no leading zeros), while every other
group is zero-padded to exactly nine digits with `%09u` — otherwise a value like
`5` sitting in a middle group would lose its leading zeros and corrupt the number.

> Note the symmetry with input: `BigInt::from_decimal` in
> [`bignum.cpp`](../src/bignum.cpp) does the reverse, consuming up to 9 digits per
> chunk (since `999,999,999 < 2^30` fits a `uint32_t`) via one `mul_small` and one
> add. Parsing is only ever used for short constants and test inputs, never for
> the multi-million-digit results, which are *generated*, not parsed.

**Honesty about cost.** Like division, `to_decimal` is **O(d²)** in the digit
count `d`: each of the `~d/9` short divisions touches the whole (shrinking)
magnitude. The inner loop is the fastest possible single-limb division, so the
constant factor is small, but the asymptotics are still quadratic. A
divide-and-conquer base conversion (recursively split the number by an
appropriate power of ten built with the fast multiply) would make this
`O(M(d) log d)`; that plan, along with the matching sub-quadratic divider, is
described in [08 — Scaling to billions](08_scaling_to_billions.md).

---

## 10. Summary

| Operation                | Implementation                          | Cost            | Where |
| ------------------------ | --------------------------------------- | --------------- | ----- |
| Representation           | sign + little-endian base-2³² `mag`     | —               | [`bignum.h`](../include/bignum.h) |
| Normalize                | `trim` + sign fix-up                    | O(n)            | [`bignum.cpp`](../src/bignum.cpp) |
| Add / Sub                | ripple carry / borrow, 64-bit accum     | O(n)            | [`bignum.cpp`](../src/bignum.cpp) |
| Multiply (small)         | schoolbook reference                    | O(n·m)          | [`bignum.cpp`](../src/bignum.cpp) |
| Multiply (medium)        | Karatsuba, 3-multiply identity          | O(n^1.585)      | [`bignum.cpp`](../src/bignum.cpp) |
| Multiply (large)         | NTT, CPU or CUDA, via `mul_dispatch`    | O(n log n)      | [05_ntt_goldilocks](05_ntt_goldilocks.md) |
| Divide / mod             | Knuth Algorithm D (self-checking)       | **O(n²)**       | [`bignum_div.cpp`](../src/bignum_div.cpp) |
| Integer sqrt             | Newton + bit-by-bit reference           | O(div · log n)  | [`bignum_div.cpp`](../src/bignum_div.cpp) |
| Decimal output           | repeated ÷10⁹                           | **O(d²)**       | [`bignum_div.cpp`](../src/bignum_div.cpp) |

The multiply side of the library is asymptotically modern (Karatsuba → NTT), and
that is what lets the binary-splitting tree scale. Division and decimal conversion
remain honestly **quadratic** today — correct and well-tested, but the two pieces
that [08 — Scaling to billions](08_scaling_to_billions.md) targets next.

**Related docs:** [02 — Binary splitting](02_binary_splitting.md) ·
[03 — Chudnovsky](03_chudnovsky.md) ·
[05 — NTT over Goldilocks](05_ntt_goldilocks.md) ·
[08 — Scaling to billions](08_scaling_to_billions.md)
