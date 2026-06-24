# Binary Splitting

> Source for this chapter: [`include/chudnovsky.h`](../include/chudnovsky.h) and the
> `chudnovsky_bs` function in [`src/chudnovsky.cpp`](../src/chudnovsky.cpp).
> Prerequisite: [01_chudnovsky.md](01_chudnovsky.md) (the series itself).
> Sequels: [04_bignum.md](04_bignum.md) (the `BigInt` type every multiply flows
> through) and [05_ntt_goldilocks.md](05_ntt_goldilocks.md) (the fast multiply that
> makes the whole thing tractable).

This chapter explains the algorithm that turns the Chudnovsky series into a *small
number of enormous integer multiplications* instead of a *huge number of
full-precision floating-point operations*. That re-shaping is the single most
important idea in the whole pipeline, because it is what lets a fast multiply
(NTT on the CPU, or NTT on a CUDA GPU) do essentially all of the work.

---

## 1. The problem: naive summation works at full precision the whole way

Recall the series we are evaluating (from [`chudnovsky.h`](../include/chudnovsky.h)):

```
1/pi = 12 * sum_{k=0}^{inf} (-1)^k (6k)! (13591409 + 545140134 k)
                             -----------------------------------------
                             (3k)! (k!)^3 640320^(3k + 3/2)
```

Each term adds about **14.18 correct decimal digits** — the code stores this as
`DIGITS_PER_TERM = 14.181647462725477` and uses it in
`chudnovsky_terms_for_digits`:

```cpp
uint64_t chudnovsky_terms_for_digits(uint64_t digits) {
    uint64_t n = (uint64_t)((double)digits / DIGITS_PER_TERM) + 2;
    return n < 2 ? 2 : n;
}
```

So for `D` digits we need `N ≈ D / 14.18` terms. For one million digits that is
about 70,500 terms; for one billion digits, about 70.5 million.

Now imagine the **obvious** way to sum the series: keep a running fixed-point
accumulator that holds `D` digits, compute each term as a fraction, and add it in.
Two things make this catastrophic:

1. **Every operation runs at the full target precision.** The accumulator is `D`
   digits wide from term 0 to term `N`. Even term 5 — which only contributes
   ~70 digits of *new* information — gets divided and added across the entire
   million-digit accumulator. You pay full width for terms that deserve almost
   nothing.
2. **Division is everywhere.** Each term is a ratio. To add a ratio into a
   fixed-point accumulator you must divide by the denominator, and bignum division
   is far more expensive than multiplication (see the honest caveat in
   [04_bignum.md](04_bignum.md): `div_fast` is still effectively quadratic-ish at
   the sizes that matter, and we want to do exactly **one** of them, at the end).

The total cost of naive summation is roughly `N` operations each of width `D`,
i.e. `O(N * M(D))` where `M(D)` is the cost of a `D`-digit multiply. Since
`N` is proportional to `D`, that is `O(D * M(D))` — a full factor of `D` worse than
what binary splitting achieves.

**Binary splitting's promise:** never divide mid-stream, never carry full precision
for the early terms, and arrange the work so the only giant multiplications happen
a handful of times near the end. The total cost drops to roughly `O(M(D) * log D)`.

---

## 2. The per-term integer atoms: p, q, t

The trick is to refuse to compute any *fraction* until the very end. Instead, each
term `k` is described by three **exact integers**. From the header comment in
[`chudnovsky.h`](../include/chudnovsky.h):

```
p(k) = (6k-5)(2k-1)(6k-1)          (for k >= 1)
q(k) = k^3 * C3_OVER_24            where C3_OVER_24 = 640320^3 / 24
t(k) = p(k) * (13591409 + 545140134 k),   negated when k is odd
```

Reading these off the code in `chudnovsky_bs`:

| atom | meaning | code |
|------|---------|------|
| `p(k)` | numerator piece that links term `k` to term `k-1` | `BigInt::from_u64(6*a-5) * BigInt::from_u64(2*a-1) * BigInt::from_u64(6*a-1)` |
| `q(k)` | denominator piece (the `(3k)!(k!)^3 640320^{3k}` growth) | `A*A*A * BigInt::from_u64(C3_OVER_24)` |
| `t(k)` | the linear numerator `13591409 + 545140134 k`, times `p(k)`, with sign | `r.T = r.P * linear; if (a & 1) r.T = -r.T;` |

The relevant constants, exactly as declared in [`chudnovsky.cpp`](../src/chudnovsky.cpp):

```cpp
const uint64_t A_CONST    = 13591409ULL;          // constant term of the linear factor
const uint64_t B_CONST    = 545140134ULL;         // linear coefficient
const uint64_t C3_OVER_24 = 10939058860032000ULL; // = 640320^3 / 24 (exact)
```

The key property: `p(k)` and `q(k)` are designed so that the **ratio of consecutive
terms** factors cleanly:

```
term(k) / term(k-1) = - p(k) / q(k)
```

That is *why* these specific cubic/cubic expressions appear — they are the rational
recurrence of the hypergeometric series. Because the ratio is rational with small
integer building blocks (each `6a-5`, `2a-1`, `6a-1`, `a`, and `C3_OVER_24` fits in
64 bits), every per-term atom can be built with `BigInt::from_u64` and a couple of
multiplies. No big numbers exist at the leaves at all.

A subtle but important detail about how `t(k)` is built. The linear factor
`13591409 + 545140134 * a` can overflow 64 bits once `a` is large, so the code does
**not** form it with plain `uint64_t` arithmetic. It uses a `BigInt` add and the
`mul_small` helper (a `BigInt × int64_t` fast path described in
[04_bignum.md](04_bignum.md)):

```cpp
BigInt linear = BigInt::from_u64(A_CONST) + mul_small(A, (int64_t)B_CONST);
r.T = r.P * linear;
if (a & 1) r.T = -r.T;          // sign (-1)^a
```

---

## 3. The base case, including the `a == 0` special case

The recursion bottoms out at a **single term**, `b - a == 1`. Here is the exact
code:

```cpp
if (b - a == 1) {
    if (a == 0) {
        // The k=0 term: p(0) and q(0) are special-cased to 1 so that the term
        // reduces to the constant 13591409.
        r.P = BigInt((int64_t)1);
        r.Q = BigInt((int64_t)1);
        r.T = BigInt::from_u64(A_CONST);
    } else {
        // p(a) = (6a-5)(2a-1)(6a-1).
        BigInt A = BigInt::from_u64(a);
        r.P = BigInt::from_u64(6 * a - 5)
            * BigInt::from_u64(2 * a - 1)
            * BigInt::from_u64(6 * a - 1);
        // q(a) = a^3 * C3_OVER_24.
        r.Q = A * A * A * BigInt::from_u64(C3_OVER_24);
        // linear factor (13591409 + 545140134*a); mul_small handles the big a.
        BigInt linear = BigInt::from_u64(A_CONST) + mul_small(A, (int64_t)B_CONST);
        r.T = r.P * linear;
        if (a & 1) r.T = -r.T;          // sign (-1)^a
    }
    return r;
}
```

### Why `a == 0` is special

If you blindly plugged `k = 0` into the general formulas you would get nonsense for
this representation:

- `q(0) = 0^3 * C3_OVER_24 = 0` — a zero denominator atom. That would poison every
  merge above it (`Q = Q_L * Q_R` would zero out whole subtrees).
- `p(0) = (−5)(−1)(−1) = −5` — not the value we want as the "linking" numerator for
  the very first term either.

The `k = 0` term of the actual series is simply the constant **13591409** (the whole
fraction collapses because `(6·0)! = (3·0)! = (0!)^3 = 640320^0 = 1`). So the code
*defines* the atoms for that leaf to make the bookkeeping work out:

- `P = 1`, `Q = 1` — the identity elements, so this leaf never scales its siblings.
- `T = 13591409` — the literal value of the first term, with **no sign flip**
  (`(-1)^0 = +1`).

This is exactly what `r.T = BigInt::from_u64(A_CONST)` produces. Think of it as: the
left edge of the whole tree contributes a clean additive constant and a multiplicative
identity, which is precisely the role term 0 plays.

### The `a >= 1` leaf

For every other single term, the three atoms are built directly from their
definitions. Note `r.P` is a product of three sub-64-bit factors, `r.Q` is `a^3`
times the big-but-still-64-bit `C3_OVER_24`, and `r.T = p(a) * linear` with the
alternating sign applied via `if (a & 1) r.T = -r.T;`. These leaf integers are tiny
(tens of digits), so leaf work is essentially free — all the cost lives higher up.

---

## 4. The merge rule, and why T pairs left-T with right-Q

When we have the triples for a left subrange `L = [a, m)` and a right subrange
`R = [m, b)`, we combine them into the triple for `[a, b)`:

```cpp
uint64_t m = (a + b) / 2;
PQT L = chudnovsky_bs(a, m);
PQT R = chudnovsky_bs(m, b);

r.P = L.P * R.P;
r.Q = L.Q * R.Q;
r.T = L.T * R.Q + L.P * R.T;
```

The `P` and `Q` merges are obvious — products of numerator/denominator pieces are
just products. The interesting one is **`T`**:

```
T = T_L * Q_R + P_L * T_R
```

### Why this exact pairing?

Hold the invariant in your head. For a range `[a, b)`, the partial sum of the series
over those terms equals `T / Q` *after the shared structure has been factored out*,
and `P` is the accumulated numerator factor that the **next** range to the right will
need in order to be expressed on a common footing with this one.

Concretely, the running sum splits as:

```
sum over [a, b)  =  (sum over left)  +  (sum over right, but scaled to the left's basis)
```

- `T_L / Q_L` is the left block's contribution.
- The right block's contribution `T_R / Q_R` is expressed *relative to where the
  right block starts*. To splice it onto the left block, every right term must be
  multiplied by the cumulative numerator factor that got us from `a` to `m`, which is
  exactly `P_L`. That is the `P_L * T_R` piece.
- Meanwhile the left block's `T_L` must be put over the **common denominator** of the
  whole range. The combined denominator is `Q_L * Q_R`, so `T_L` (currently over
  `Q_L`) gets multiplied by `Q_R`. That is the `T_L * Q_R` piece.

Put together over the common denominator `Q = Q_L * Q_R`:

```
T_L     P_L * T_R       T_L * Q_R + P_L * T_R       T
----  + ----------  =   ---------------------   =   ---
Q_L        Q_R               Q_L * Q_R              Q
```

So **`T` pairs left-T with right-Q** because `T_L` lives over `Q_L` and must be
re-based onto the wider denominator `Q_L * Q_R` — multiplying by `Q_R` does exactly
that — while the right contribution must be lifted into the left's numerator basis by
`P_L`. The asymmetry (`T_L` meets `Q_R`, but `T_R` meets `P_L`) is the whole reason
binary splitting is correct. Swap them and you get garbage.

At the very top of the recursion, `chudnovsky_bs(0, N)` returns `(P, Q, T)` for the
entire sum, and the final assembly only needs `Q` and `T`:

```cpp
// pi = 426880 * sqrt(10005) * Q / T   (the P at the root is discarded)
BigInt numerator = mul_small(r.Q, (int64_t)L_CONST) * sqrtC;
BigInt pi_scaled = div_fast(numerator, r.T);        // floor(pi * 10^prec)
```

Note that **exactly one division** happens in the entire computation (`div_fast`),
right at the end — never per-term. That is the payoff of carrying exact integers.
`L_CONST = 426880` and the `sqrt(10005)` factor (`sqrtC`) come straight from the
closed form `pi = (426880 * sqrt(10005) * Q) / T`.

---

## 5. The balanced binary tree, and where the work concentrates

`chudnovsky_bs` splits at the midpoint `m = (a + b) / 2`, so the recursion forms a
**balanced binary tree**: each internal node has two children of (nearly) equal width,
and the depth is about `log2(N)`.

Here is the tree for `N = 8` terms, `chudnovsky_bs(0, 8)`:

```
                         [0,8)              <- ROOT: operands ~all D digits
                        /     \                (the giant multiply)
                  [0,4)         [4,8)        <- each ~D/2 digits
                 /     \       /     \
            [0,2)   [2,4)  [4,6)   [6,8)     <- each ~D/4 digits
            /  \    /  \    /  \    /  \
          0    1  2    3  4    5  6    7     <- leaves: tiny (tens of digits)
```

The crucial observation is about **operand size as a function of height**:

- At a **leaf**, the integers are a few dozen digits. Multiplying them is free.
- One level up, the products are a hundred-ish digits.
- The sizes roughly **double every level you climb**, because a merge multiplies two
  child integers whose digit counts add.
- At the **root**, `Q_L` and `Q_R` each carry about `D/2` digits, so `Q = Q_L * Q_R`
  is a `D/2 × D/2 → D` digit multiply. The `T = T_L * Q_R + P_L * T_R` merge at the
  root involves two more multiplies of similar size.

So the total work is dominated by the **top few levels**. Quantitatively, with a
near-linear multiply of cost `M(n)`:

| level (from root) | # nodes | size per multiply | cost of level |
|-------------------|---------|-------------------|---------------|
| 0 (root)          | 1       | `~D`              | `~M(D)`       |
| 1                 | 2       | `~D/2`            | `~2 M(D/2) ≈ M(D)` |
| 2                 | 4       | `~D/4`            | `~4 M(D/4) ≈ M(D)` |
| ...               | ...     | ...               | ...           |
| bottom            | `~N`    | tiny              | negligible    |

Each level costs roughly the same (`~M(D)`), and there are `~log N ≈ log D` levels, so
the grand total is `O(M(D) log D)`. But that sum is **front-loaded**: the single root
multiply alone is `M(D)`, the same order as *every other multiply in the tree
combined*. The header says it plainly:

> *"the only truly enormous multiplications happen near the ROOT, where both operands
> have ~all the digits -- and those are exactly the multiplications our NTT/CUDA fast
> multiply accelerates."*

---

## 6. Therefore: fast multiplication is the whole game

Put the two facts together:

1. Almost all the time is spent in a handful of multiplications near the root.
2. Those multiplications are between `~D/2`-digit and `~D`-digit integers.

It follows that **the speed of pi is the speed of one big multiply.** Everything
else — the millions of leaf constructions, the lower-tree merges, the bookkeeping —
is rounding error in the total runtime. If you want pi faster, you make the big
multiply faster. Full stop.

That is why this project funnels *every* `BigInt::operator*` through a size-dispatched
backend. From the file header in [`chudnovsky.cpp`](../src/chudnovsky.cpp):

> *"Every multiplication here ultimately flows through `BigInt::operator*`, which
> dispatches to schoolbook / Karatsuba / CPU-NTT / CUDA-NTT by size. So simply by
> leaving the multiply backend on Auto, the big multiplications near the root of the
> splitting tree run on the GPU."*

- **Leaf and low-tree multiplies** are small, so the dispatcher picks schoolbook or
  Karatsuba — see [04_bignum.md](04_bignum.md).
- **The root-region multiplies** are huge, so the dispatcher picks the
  number-theoretic transform: CPU-NTT, or the CUDA-NTT over the Goldilocks prime —
  see [05_ntt_goldilocks.md](05_ntt_goldilocks.md).

The binary splitting code itself contains **no** mention of NTT, GPUs, or transform
sizes. It just writes `L.P * R.P`, `L.Q * R.Q`, `L.T * R.Q + L.P * R.T` and trusts
the `BigInt` layer to choose the right engine by operand size. That separation is the
elegant part: the *math* (this file) decides *what* to multiply; the *bignum* layer
decides *how*.

---

## 7. A worked tiny example: N = 4

Let us actually run `chudnovsky_bs(0, 4)` by hand and watch the atoms flow up the
tree. We need the leaves for `k = 0, 1, 2, 3`.

### Leaves

**`k = 0`** (the special case): `P = 1`, `Q = 1`, `T = 13591409`.

**`k = 1`**:
- `p(1) = (6−5)(2−1)(6−1) = 1 · 1 · 5 = 5`
- `q(1) = 1^3 · C3_OVER_24 = 10939058860032000`
- `linear = 13591409 + 545140134 · 1 = 558731543`
- `t(1) = p(1) · linear = 5 · 558731543 = 2793657715`, then negated (`k` odd):
  `T = −2793657715`

**`k = 2`**:
- `p(2) = (12−5)(4−1)(12−1) = 7 · 3 · 11 = 231`
- `q(2) = 2^3 · C3_OVER_24 = 8 · 10939058860032000 = 87512470880256000`
- `linear = 13591409 + 545140134 · 2 = 1103871677`
- `t(2) = 231 · 1103871677 = 254994357387`; `k` even, so `T = +254994357387`

**`k = 3`**:
- `p(3) = (18−5)(6−1)(18−1) = 13 · 5 · 17 = 1105`
- `q(3) = 27 · 10939058860032000 = 295354589220864000`
- `linear = 13591409 + 545140134 · 3 = 1649011811`
- `t(3) = 1105 · 1649011811 = 1822158051155`, negated (`k` odd):
  `T = −1822158051155`

### Merge the left pair `[0, 2)` from leaves 0 and 1

```
P = P0 * P1 = 1 * 5 = 5
Q = Q0 * Q1 = 1 * 10939058860032000 = 10939058860032000
T = T0 * Q1 + P0 * T1
  = 13591409 * 10939058860032000 + 1 * (-2793657715)
  = 148678435198935838688000 - 2793657715
  = 148678435198935835894285
```

### Merge the right pair `[2, 4)` from leaves 2 and 3

```
P = P2 * P3 = 231 * 1105 = 255255
Q = Q2 * Q3 = 87512470880256000 * 295354589220864000
            = 25847751699443765365112426496000000
T = T2 * Q3 + P2 * T3
  = 254994357387 * 295354589220864000
    + 231 * (-1822158051155)
  = 75313078285616027718243168000 - 420918509816805
  = 75313078285616027297324658195
```

### Merge the root `[0, 4)`

```
P = P_left * P_right = 5 * 255255 = 1276275
Q = Q_left * Q_right
  = 10939058860032000 * 25847751699443765365112426496000000
T = T_left * Q_right + P_left * T_right
  = 148678435198935835894285 * 25847751699443765365112426496000000
    + 5 * 75313078285616027297324658195
```

Notice what just happened at the root: the two `T`/`Q` products are by far the
largest multiplications in the example — `Q_left` (17 digits) times `Q_right`
(35 digits), and a 24-digit `T_left` times that same 35-digit `Q_right`. At `N = 4`
these are still trivial, but scale `N` to 70 million and those root operands become
*hundreds of millions of digits each* — that is the multiply you must accelerate.

To sanity-check the math you would finish with the closed form
`pi ≈ 426880 · sqrt(10005) · Q / T` (this is what `compute_pi_chudnovsky` does with
`prec`-digit fixed point); with only 4 terms you would already get pi correct to
~57 digits, since `4 × 14.18 ≈ 57`.

---

## 8. Recursion depth and memory

### Depth

The recursion in `chudnovsky_bs` is balanced (`m = (a + b) / 2`), so the call-stack
depth is `O(log2(N))`. Even for a billion digits (`N ≈ 7.05e7`), that is only
`log2(7.05e7) ≈ 27` stack frames deep. Stack overflow is a non-issue; this is a
**shallow** recursion, not a deep one. Each frame is small — it holds two `PQT`
locals (`L` and `R`) plus the result `r`.

### Memory — the real cost is the big integers, not the stack

What actually consumes memory is the **size of the `BigInt`s near the root**. A `PQT`
holds three `BigInt`s (`P`, `Q`, `T`), and near the top of the tree each of `Q` and
`T` is a `~D`-digit number. At a billion decimal digits that is on the order of
hundreds of megabytes *per* big integer, and the merge

```cpp
r.T = L.T * R.Q + L.P * R.T;
```

transiently holds `L`, `R`, the two products, and the sum simultaneously. So peak
memory is a small multiple of the output size, dominated entirely by the top one or
two levels of the tree. The leaves and lower levels are negligible by comparison.

A practical consequence: because the recursion descends the **left** child fully
before the right (`PQT L = chudnovsky_bs(a, m); PQT R = chudnovsky_bs(m, b);`), the
left subtree's result `L` is held live in the frame while the entire right subtree is
computed. At the root this means one half-sized result sits in memory while the other
half is built — another reason peak memory tracks the output size. (More
sophisticated implementations stream or free aggressively to shave the constant
factor; this implementation keeps it simple and readable, matching the comment-heavy
tone of the source.)

### Why the `P` at the root is wasted work (a minor honest caveat)

The final formula needs only `Q` and `T`:

```cpp
BigInt numerator = mul_small(r.Q, (int64_t)L_CONST) * sqrtC;
BigInt pi_scaled = div_fast(numerator, r.T);
```

`r.P` is never read. Yet `chudnovsky_bs` still computes `r.P = L.P * R.P` at **every**
node, including the root, where it is a large multiply whose result is immediately
discarded. This is a deliberate simplicity-for-speed trade: special-casing "don't
compute `P` at the root" would save one big multiply (a few percent), at the cost of a
messier interface. Some highly tuned pi programs do exactly that elision; this one
keeps the merge uniform. Worth knowing if you profile and wonder why there is one more
giant multiply than strictly necessary.

---

## 9. Summary

- Naive summation pays **full precision for every term** and does a division per
  term — `O(D · M(D))`. Binary splitting keeps **exact integer atoms** and divides
  **once**, at the end — `O(M(D) · log D)`.
- Each term is three integers: `p(k)`, `q(k)`, `t(k)`, built from sub-64-bit factors
  at the leaves. The `k = 0` leaf is special-cased to `P = Q = 1`, `T = 13591409`.
- The merge `P = P_L·P_R`, `Q = Q_L·Q_R`, `T = T_L·Q_R + P_L·T_R` combines ranges;
  `T` pairs **left-T with right-Q** because `T_L` must be re-based onto the wider
  denominator `Q_L·Q_R`, while the right contribution is lifted into the left's
  numerator basis by `P_L`.
- The tree is **balanced** and **shallow** (`~log2 N` deep), but the work is
  **front-loaded** at the root, where operands carry ~all `D` digits.
- Therefore **fast multiplication is the whole game** — the root-region multiplies
  dominate, and they are handed to NTT/CUDA by the size-dispatching `BigInt`.

Next: how those integers are stored and multiplied in
[04_bignum.md](04_bignum.md), and how the giant multiplies are accelerated in
[05_ntt_goldilocks.md](05_ntt_goldilocks.md).
