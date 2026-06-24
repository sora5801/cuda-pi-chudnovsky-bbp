// ============================================================================
//  bignum.cpp  --  Core big-integer arithmetic
// ============================================================================
//
//  This file implements the "easy but essential" half of BigInt:
//    * construction / normalization / comparison
//    * addition and subtraction (with sign handling)
//    * bit shifts
//    * the two classic multiplication algorithms: schoolbook and Karatsuba
//    * the multiply *dispatcher* that chooses a backend by size
//    * small helpers (multiply by a machine word, powers of ten, decimal parse)
//
//  Division, square root, and decimal *output* live in bignum_div.cpp because
//  they are conceptually a level up (they are built ON TOP of multiplication).
//
//  Everything here works on the little-endian, base-2^32 limb representation
//  described at length in bignum.h. Read that header first if you have not.
// ============================================================================

#include "bignum.h"

#include <algorithm>  // std::max, std::min, std::reverse
#include <cassert>
#include <stdexcept>

namespace pidigits {

// The one global multiply configuration object declared in the header.
MulConfig g_mul;

// B = 2^32 is the base of our number system. Each limb is a digit in base B.
// We never name the literal 4294967296 in code; we work with uint32_t limbs and
// uint64_t accumulators so the base is implicit in the types.

// ============================================================================
//  Section 1: magnitude-only helpers (no sign).
//
//  These operate directly on std::vector<uint32_t> limb arrays. Keeping them
//  separate from the signed BigInt logic means the performance-critical code
//  never has to think about signs.
// ============================================================================
namespace {

// Strip most-significant zero limbs so the top limb is non-zero (or the vector
// is empty, representing zero). All magnitude producers call this at the end.
void trim(std::vector<uint32_t>& v) {
    while (!v.empty() && v.back() == 0) v.pop_back();
}

// Compare two magnitudes. Returns -1 if a<b, 0 if a==b, +1 if a>b.
// Assumes both are already trimmed (no leading zeros), so we can compare lengths
// first and only then compare limbs from the most significant downward.
int cmp_mag_vec(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : +1;
    for (size_t i = a.size(); i-- > 0;) {            // i goes high..low
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : +1;
    }
    return 0;
}

// out = a + b  (magnitudes). Classic ripple-carry addition in base 2^32.
// We use a 64-bit accumulator so a 32-bit + 32-bit + carry sum cannot overflow.
std::vector<uint32_t> add_mag(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b) {
    const std::vector<uint32_t>& big   = a.size() >= b.size() ? a : b;
    const std::vector<uint32_t>& small = a.size() >= b.size() ? b : a;
    std::vector<uint32_t> out;
    out.reserve(big.size() + 1);                     // +1 for a possible final carry
    uint64_t carry = 0;
    for (size_t i = 0; i < big.size(); ++i) {
        // sum of this column: limb of big, limb of small (0 past its end), carry.
        uint64_t sum = (uint64_t)big[i] + carry + (i < small.size() ? small[i] : 0);
        out.push_back((uint32_t)sum);                // low 32 bits stay here
        carry = sum >> 32;                           // high bits carry to next column
    }
    if (carry) out.push_back((uint32_t)carry);       // final carry becomes a new top limb
    // (no trim needed: the construction above never leaves a leading zero)
    return out;
}

// out = a - b  (magnitudes), REQUIRES a >= b. Ripple-borrow subtraction.
// The trick `(uint64_t)a[i] - b[i] - borrow` is computed in 64-bit signed-ish
// space; if it "went negative" the high 32 bits are all 1s, which we detect by
// shifting and masking to recover the borrow.
std::vector<uint32_t> sub_mag(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b) {
    std::vector<uint32_t> out;
    out.reserve(a.size());
    uint64_t borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t bi = (i < b.size() ? b[i] : 0);
        // Add 2^32 ("0x100000000") before subtracting so the result is always a
        // valid non-negative 64-bit number; the carry-out tells us whether we had
        // to "borrow" from the next column.
        uint64_t diff = (uint64_t)a[i] + 0x100000000ULL - bi - borrow;
        out.push_back((uint32_t)diff);               // low 32 bits = column result
        borrow = (diff >> 32) ? 0 : 1;               // if no overflow happened, we borrowed
    }
    // Since a >= b, the final borrow must be 0. Trim possible leading zeros (e.g.
    // 0x...1 - 0x...1 produces high zero limbs).
    trim(out);
    return out;
}

// Shift a magnitude left by `n` whole limbs (i.e. multiply by (2^32)^n). This is
// just prepending n zero limbs at the bottom. Used by Karatsuba to place the
// high and middle partial products.
std::vector<uint32_t> shift_limbs(const std::vector<uint32_t>& a, size_t n) {
    if (a.empty()) return {};
    std::vector<uint32_t> out(a.size() + n, 0);
    std::copy(a.begin(), a.end(), out.begin() + n);
    return out;
}

} // anonymous namespace

// ============================================================================
//  Section 2: schoolbook and Karatsuba multiplication (magnitude only).
// ============================================================================

// O(n*m) "long multiplication". For every pair of limbs we form a 64-bit partial
// product and accumulate it into the right column, propagating carries. This is
// the literal definition of multiplication and is our golden reference that all
// faster algorithms are tested against.
std::vector<uint32_t> mul_schoolbook(const std::vector<uint32_t>& a,
                                     const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};           // anything * 0 = 0
    std::vector<uint32_t> out(a.size() + b.size(), 0); // product fits in na+nb limbs
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t carry = 0;
        uint64_t ai = a[i];
        for (size_t j = 0; j < b.size(); ++j) {
            // Accumulate: existing column value + ai*bj + carry. All four terms
            // together fit in 64 bits because (2^32-1)^2 + 2*(2^32-1) < 2^64.
            uint64_t cur = (uint64_t)out[i + j] + ai * (uint64_t)b[j] + carry;
            out[i + j] = (uint32_t)cur;               // keep low 32 bits in place
            carry = cur >> 32;                        // carry the rest upward
        }
        out[i + b.size()] += (uint32_t)carry;         // deposit the final carry
    }
    trim(out);
    return out;
}

// O(n^1.585) Karatsuba multiplication.
//
//  Idea: write a = a1*B^k + a0 and b = b1*B^k + b0 (splitting each number into a
//  low half a0/b0 and a high half a1/b1 at limb position k). The naive product
//      a*b = a1*b1*B^2k + (a1*b0 + a0*b1)*B^k + a0*b0
//  needs FOUR half-size multiplications. Karatsuba's insight is that the middle
//  term can be recovered from the other two with just ONE more multiplication:
//      z0 = a0*b0
//      z2 = a1*b1
//      z1 = (a0+a1)*(b0+b1) - z0 - z2   ==  a1*b0 + a0*b1
//  So THREE half-size multiplications suffice. Recursing, the cost exponent
//  drops from 2 (schoolbook) to log2(3) ~= 1.585. Below a threshold we hand off
//  to schoolbook, whose tiny constant factor wins for small inputs.
std::vector<uint32_t> mul_karatsuba(const std::vector<uint32_t>& a,
                                    const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};
    size_t na = a.size(), nb = b.size();
    if (std::min(na, nb) < g_mul.karatsuba_threshold)
        return mul_schoolbook(a, b);

    size_t k = std::max(na, nb) / 2;                  // split point (in limbs)

    // Build the four halves as separate limb vectors. If a number is shorter than
    // k limbs, its high half is empty (zero).
    auto low  = [&](const std::vector<uint32_t>& v) {
        std::vector<uint32_t> r(v.begin(), v.begin() + std::min(k, v.size()));
        trim(r); return r;
    };
    auto high = [&](const std::vector<uint32_t>& v) {
        if (v.size() <= k) return std::vector<uint32_t>{};
        std::vector<uint32_t> r(v.begin() + k, v.end());
        trim(r); return r;
    };
    std::vector<uint32_t> a0 = low(a), a1 = high(a);
    std::vector<uint32_t> b0 = low(b), b1 = high(b);

    std::vector<uint32_t> z0 = mul_karatsuba(a0, b0);          // a0*b0
    std::vector<uint32_t> z2 = mul_karatsuba(a1, b1);          // a1*b1
    std::vector<uint32_t> z1 = mul_karatsuba(add_mag(a0, a1),  // (a0+a1)*(b0+b1)
                                             add_mag(b0, b1));
    // z1 -= z0; z1 -= z2  (both subtractions are guaranteed non-negative)
    z1 = sub_mag(z1, z0);
    z1 = sub_mag(z1, z2);

    // Reassemble: result = z2*B^(2k) + z1*B^k + z0.
    std::vector<uint32_t> out = z0;
    out = add_mag(out, shift_limbs(z1, k));
    out = add_mag(out, shift_limbs(z2, 2 * k));
    trim(out);
    return out;
}

// ============================================================================
//  Section 3: the multiply dispatcher.
//
//  Given two magnitudes, pick an algorithm. In Auto mode we choose by the size
//  of the SMALLER operand (that bounds the inner work). Explicit modes force one
//  backend, which the test suite uses to compare them all.
// ============================================================================
std::vector<uint32_t> mul_dispatch(const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};
    size_t n = std::min(a.size(), b.size());

    MulBackend backend = g_mul.backend;
    if (backend == MulBackend::Auto) {
        if (n < g_mul.karatsuba_threshold)      backend = MulBackend::Schoolbook;
        else if (n < g_mul.ntt_threshold)       backend = MulBackend::Karatsuba;
        else backend = g_mul.prefer_cuda && cuda_available()
                          ? MulBackend::NttCuda : MulBackend::NttCpu;
    }

    if (g_mul.verbose) {
        const char* name =
            backend == MulBackend::Schoolbook ? "schoolbook" :
            backend == MulBackend::Karatsuba  ? "karatsuba"  :
            backend == MulBackend::NttCpu     ? "ntt-cpu"    : "ntt-cuda";
        std::fprintf(stderr, "[mul] %zux%zu limbs via %s\n", a.size(), b.size(), name);
    }

    switch (backend) {
        case MulBackend::Schoolbook: return mul_schoolbook(a, b);
        case MulBackend::Karatsuba:  return mul_karatsuba(a, b);
        case MulBackend::NttCpu:     return mul_ntt_cpu(a, b);
        case MulBackend::NttCuda:    return mul_ntt_cuda(a, b);
        default:                     return mul_schoolbook(a, b);
    }
}

// ============================================================================
//  Section 4: BigInt construction, normalization, comparison.
// ============================================================================

void BigInt::normalize() {
    trim(mag);
    if (mag.empty()) sign = 0;             // magnitude zero forces sign zero
    else if (sign == 0) sign = 1;          // non-empty magnitude must have a sign
}

BigInt::BigInt(int64_t v) {
    if (v == 0) { sign = 0; return; }
    sign = v < 0 ? -1 : +1;
    // Take the absolute value in unsigned space. Writing it as
    // 0 - (uint64_t)v avoids undefined behavior when v == INT64_MIN (whose
    // positive does not fit in int64_t).
    uint64_t u = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
    mag.push_back((uint32_t)u);
    if (u >> 32) mag.push_back((uint32_t)(u >> 32));
}

BigInt BigInt::from_u64(uint64_t u) {
    BigInt r;
    if (u == 0) return r;
    r.sign = 1;
    r.mag.push_back((uint32_t)u);
    if (u >> 32) r.mag.push_back((uint32_t)(u >> 32));
    return r;
}

int BigInt::cmp_mag(const BigInt& o) const { return cmp_mag_vec(mag, o.mag); }

int BigInt::cmp(const BigInt& o) const {
    if (sign != o.sign) return sign < o.sign ? -1 : +1; // different signs: order by sign
    if (sign == 0) return 0;                            // both zero
    int m = cmp_mag(o);                                 // same sign: compare magnitudes
    return sign > 0 ? m : -m;                            // ...flipped if both negative
}

BigInt BigInt::operator-() const {
    BigInt r = *this;
    r.sign = -r.sign;     // negating zero leaves sign 0, which is correct
    return r;
}

BigInt BigInt::abs() const {
    BigInt r = *this;
    if (r.sign < 0) r.sign = 1;
    return r;
}

uint64_t BigInt::low64() const {
    uint64_t lo = mag.size() > 0 ? mag[0] : 0;
    uint64_t hi = mag.size() > 1 ? mag[1] : 0;
    return lo | (hi << 32);
}

size_t BigInt::bit_length() const {
    if (mag.empty()) return 0;
    size_t top = mag.size() - 1;
    uint32_t hi = mag[top];
    size_t bits = 0;
    while (hi) { ++bits; hi >>= 1; }          // bits = position of highest set bit in top limb
    return top * 32 + bits;
}

// ============================================================================
//  Section 5: signed addition and subtraction.
//
//  These reduce signed +/- to magnitude add/sub by reasoning about the signs:
//    a + b with equal signs    -> add magnitudes, keep the sign
//    a + b with opposite signs -> subtract the smaller magnitude from the larger,
//                                 take the sign of the larger magnitude
//  Subtraction is just addition of the negation.
// ============================================================================

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

BigInt operator-(const BigInt& a, const BigInt& b) {
    return a + (-b);                          // a - b == a + (-b)
}

// ============================================================================
//  Section 6: multiplication and small-scalar multiplication.
// ============================================================================

BigInt operator*(const BigInt& a, const BigInt& b) {
    BigInt r;
    if (a.sign == 0 || b.sign == 0) return r; // 0 * anything = 0
    r.mag = mul_dispatch(a.mag, b.mag);
    r.sign = a.sign * b.sign;                 // sign-magnitude rule
    r.normalize();
    return r;
}

// Multiply a magnitude by a single 32-bit word (carry-propagating). Internal.
static std::vector<uint32_t> mul_word(const std::vector<uint32_t>& a, uint32_t m) {
    if (a.empty() || m == 0) return {};
    std::vector<uint32_t> out;
    out.reserve(a.size() + 1);
    uint64_t carry = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t cur = (uint64_t)a[i] * m + carry;
        out.push_back((uint32_t)cur);
        carry = cur >> 32;
    }
    if (carry) out.push_back((uint32_t)carry);
    return out;
}

BigInt mul_small(const BigInt& a, uint32_t m) {
    BigInt r;
    if (a.sign == 0 || m == 0) return r;
    r.mag = mul_word(a.mag, m);
    r.sign = a.sign;
    r.normalize();
    return r;
}

BigInt mul_small(const BigInt& a, int64_t m) {
    if (m == 0 || a.sign == 0) return BigInt();
    int msign = m < 0 ? -1 : 1;
    uint64_t um = m < 0 ? (uint64_t)0 - (uint64_t)m : (uint64_t)m;
    // If the scalar fits in 32 bits, use the fast single-word path; otherwise
    // fall back to a full BigInt multiply (rare; e.g. 545140134 fits in 32 bits
    // but a few constants might not).
    BigInt r;
    if ((um >> 32) == 0) {
        r = mul_small(a, (uint32_t)um);
    } else {
        r = a * BigInt::from_u64(um);
    }
    r.sign = a.sign * msign;
    r.normalize();
    return r;
}

// ============================================================================
//  Section 7: bit shifts.
// ============================================================================

// value * 2^bits. We split the shift into whole-limb and sub-limb parts.
BigInt BigInt::shl(size_t bits) const {
    if (sign == 0 || bits == 0) return *this;
    size_t limb_shift = bits / 32;            // whole 32-bit limbs to move up
    unsigned bit_shift = (unsigned)(bits % 32); // leftover bits within a limb
    BigInt r;
    r.sign = sign;
    r.mag.assign(mag.size() + limb_shift + 1, 0);
    if (bit_shift == 0) {
        for (size_t i = 0; i < mag.size(); ++i) r.mag[i + limb_shift] = mag[i];
    } else {
        uint32_t carry = 0;                   // bits spilling into the next limb
        for (size_t i = 0; i < mag.size(); ++i) {
            uint64_t v = ((uint64_t)mag[i] << bit_shift) | carry;
            r.mag[i + limb_shift] = (uint32_t)v;
            carry = (uint32_t)(v >> 32);
        }
        r.mag[mag.size() + limb_shift] = carry;
    }
    r.normalize();
    return r;
}

// floor(value / 2^bits). For our use (always non-negative operands) this is a
// plain logical right shift of the magnitude. (We assert non-negativity because
// flooring semantics for negatives are not needed anywhere in this project.)
BigInt BigInt::shr(size_t bits) const {
    if (sign == 0 || bits == 0) return *this;
    assert(sign > 0 && "shr is only used on non-negative values in this project");
    size_t limb_shift = bits / 32;
    unsigned bit_shift = (unsigned)(bits % 32);
    if (limb_shift >= mag.size()) return BigInt(); // shifted everything away -> 0
    BigInt r;
    r.sign = sign;
    size_t out_len = mag.size() - limb_shift;
    r.mag.assign(out_len, 0);
    if (bit_shift == 0) {
        for (size_t i = 0; i < out_len; ++i) r.mag[i] = mag[i + limb_shift];
    } else {
        for (size_t i = 0; i < out_len; ++i) {
            uint32_t lo = mag[i + limb_shift] >> bit_shift;
            uint32_t hi = (i + limb_shift + 1 < mag.size())
                              ? mag[i + limb_shift + 1] << (32 - bit_shift) : 0;
            r.mag[i] = lo | hi;
        }
    }
    r.normalize();
    return r;
}

// ============================================================================
//  Section 8: powers of ten, and decimal parsing.
// ============================================================================

// pow10(e) returns 10^e as a BigInt, memoized in a growing cache. Powers of ten
// are needed to (a) scale pi into a fixed-point integer and (b) convert between
// binary and decimal. Building them incrementally (10^e = 10^(e-1) * 10) keeps it
// simple; the cache makes repeated calls free.
const BigInt& pow10(size_t exp) {
    static std::vector<BigInt> cache = { BigInt((int64_t)1) }; // cache[0] = 1
    while (cache.size() <= exp) {
        cache.push_back(mul_small(cache.back(), (uint32_t)10));
    }
    return cache[exp];
}

// Parse a (possibly signed, arbitrarily long) decimal string into a BigInt.
// We consume the digits 9 at a time: 9 decimal digits always fit in a uint32_t
// (max 999,999,999 < 2^30), so each chunk is one multiply-by-10^k plus one add.
// This is O(n^2) in the number of digits but we only ever parse short constants
// and test inputs, never the multi-million-digit results (those are GENERATED,
// not parsed), so the simplicity is worth it.
BigInt BigInt::from_decimal(const std::string& s) {
    size_t i = 0;
    int sgn = 1;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) { if (s[i] == '-') sgn = -1; ++i; }
    BigInt result;                                  // starts at 0
    while (i < s.size()) {
        // Grab up to 9 digits for this chunk.
        size_t take = std::min<size_t>(9, s.size() - i);
        uint32_t chunk = 0, scale = 1;
        for (size_t j = 0; j < take; ++j) {
            char c = s[i + j];
            if (c < '0' || c > '9') throw std::invalid_argument("from_decimal: bad digit");
            chunk = chunk * 10 + (uint32_t)(c - '0');
            scale *= 10;
        }
        i += take;
        result = mul_small(result, scale);          // shift existing value up
        result = result + BigInt::from_u64(chunk);  // add this chunk
    }
    result.sign = result.mag.empty() ? 0 : sgn;
    result.normalize();
    return result;
}

} // namespace pidigits
