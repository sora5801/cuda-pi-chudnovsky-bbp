// ============================================================================
//  bignum_div.cpp  --  Division, integer square root, and decimal output
// ============================================================================
//
//  These three operations sit one level above multiplication. The Chudnovsky
//  final assembly (chudnovsky.cpp) needs all of them:
//
//      pi ~= ( Q * 426880 * isqrt(10005 * 10^(2*digits)) ) / T
//                              \_______ isqrt _______/      \_ division _/
//      and then the result is printed in base 10           \_ to_decimal _/
//
//  We implement:
//    * divmod_knuth -- Knuth's Algorithm D, the classic exact long division.
//      It is our correctness ground truth and the workhorse used everywhere.
//    * div_fast     -- a thin wrapper today (forwards to Knuth); the API exists
//      so a sub-quadratic Newton/Burnikel-Ziegler divider can be slotted in
//      later without touching callers. (See docs/08_scaling_to_billions.md.)
//    * isqrt        -- integer floor(sqrt(s)) by Newton's method, with a slow
//      but obviously-correct bit-by-bit reference (isqrt_bitwise) for testing.
//    * BigInt::to_decimal -- base-2^32 -> base-10 string by repeated division by
//      10^9 (one 32-bit limb), collecting nine decimal digits at a time.
//
//  CORRECTNESS PHILOSOPHY
//  ----------------------
//  Division has a self-checking invariant: for q = floor(a/b), r = a - q*b we must
//  have  q*b + r == a  AND  0 <= r < b. The test suite checks exactly that, so we
//  can trust divmod without needing a second independent divider. isqrt is checked
//  against isqrt_bitwise and against  x^2 <= s < (x+1)^2.
// ============================================================================

#include "bignum.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <cstdio>

namespace pidigits {

// Local copies of the tiny magnitude helpers (the ones in bignum.cpp are file-
// local). Kept here so this translation unit is self-contained.
namespace {
void trim(std::vector<uint32_t>& v) { while (!v.empty() && v.back() == 0) v.pop_back(); }

int cmp_mag_vec(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : +1;
    for (size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : +1;
    return 0;
}

// Number of leading zero bits in a 32-bit word (0..32). Used by the Knuth
// normalization step, which shifts the divisor so its top bit is set.
int nlz32(uint32_t x) {
    if (x == 0) return 32;
    int n = 0;
    while (!(x & 0x80000000u)) { x <<= 1; ++n; }
    return n;
}

// ----------------------------------------------------------------------------
//  The core: Knuth's Algorithm D on base-2^32 limb vectors.
//  Computes q = floor(u / v) and r = u - q*v, with 0 <= r < v.
//  Preconditions: v is non-empty and normalized (top limb non-zero), u
//  normalized. This is the well-known "divmnu" formulation (Hacker's Delight /
//  Knuth TAOCP vol. 2, sec. 4.3.1).
// ----------------------------------------------------------------------------
void divmod_mag(const std::vector<uint32_t>& u_in, const std::vector<uint32_t>& v_in,
                std::vector<uint32_t>& q, std::vector<uint32_t>& r) {
    // Quick exits.
    if (cmp_mag_vec(u_in, v_in) < 0) { q.clear(); r = u_in; trim(r); return; }
    const size_t n = v_in.size();
    const size_t m = u_in.size();

    // --- Special case: single-limb divisor -> simple short division ---------
    if (n == 1) {
        uint32_t d = v_in[0];
        q.assign(m, 0);
        uint64_t rem = 0;
        for (size_t i = m; i-- > 0;) {
            uint64_t cur = (rem << 32) | u_in[i];   // bring down next limb
            q[i] = (uint32_t)(cur / d);
            rem = cur % d;
        }
        trim(q);
        r.clear();
        if (rem) r.push_back((uint32_t)rem);
        return;
    }

    // --- Normalize: shift divisor left so its top limb has bit 31 set -------
    // This makes the quotient-digit estimate qhat accurate to within +2.
    const int s = nlz32(v_in[n - 1]);
    std::vector<uint32_t> vn(n);
    for (size_t i = n - 1; i > 0; --i)
        vn[i] = (v_in[i] << s) | (s ? (uint32_t)((uint64_t)v_in[i - 1] >> (32 - s)) : 0);
    vn[0] = v_in[0] << s;

    // The dividend gets the same shift and ONE extra high limb of headroom.
    std::vector<uint32_t> un(m + 1);
    un[m] = s ? (uint32_t)((uint64_t)u_in[m - 1] >> (32 - s)) : 0;
    for (size_t i = m - 1; i > 0; --i)
        un[i] = (u_in[i] << s) | (s ? (uint32_t)((uint64_t)u_in[i - 1] >> (32 - s)) : 0);
    un[0] = u_in[0] << s;

    q.assign(m - n + 1, 0);
    const uint64_t B = 0x100000000ULL; // the base, 2^32

    // --- Main loop: produce one quotient limb per iteration, top-down -------
    for (size_t jj = m - n + 1; jj-- > 0;) {
        const size_t j = jj;
        // Estimate this quotient digit from the top two dividend limbs.
        uint64_t num = ((uint64_t)un[j + n] << 32) | un[j + n - 1];
        uint64_t qhat = num / vn[n - 1];
        uint64_t rhat = num % vn[n - 1];
        // Refine qhat downward until qhat*vn[n-2] <= rhat*B + un[j+n-2].
        while (qhat >= B ||
               qhat * vn[n - 2] > ((rhat << 32) | un[j + n - 2])) {
            --qhat;
            rhat += vn[n - 1];
            if (rhat >= B) break;       // rhat overflowed the base; estimate is fine now
        }

        // Multiply the divisor by qhat and subtract from the current window.
        int64_t k = 0;                   // running borrow/carry across limbs
        for (size_t i = 0; i < n; ++i) {
            uint64_t p = qhat * vn[i];
            int64_t t = (int64_t)un[i + j] - k - (int64_t)(p & 0xFFFFFFFFu);
            un[i + j] = (uint32_t)t;
            k = (int64_t)(p >> 32) - (t >> 32); // (t>>32) sign-extends the borrow
        }
        int64_t t = (int64_t)un[j + n] - k;
        un[j + n] = (uint32_t)t;
        q[j] = (uint32_t)qhat;

        // If we over-subtracted (qhat was 1 too big), add the divisor back once.
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
    }
    trim(q);

    // --- Denormalize the remainder: r = un >> s -----------------------------
    r.assign(n, 0);
    for (size_t i = 0; i < n - 1; ++i)
        r[i] = (un[i] >> s) | (s ? (uint32_t)((uint64_t)un[i + 1] << (32 - s)) : 0);
    r[n - 1] = un[n - 1] >> s;
    trim(r);
}
} // anonymous namespace

// ----------------------------------------------------------------------------
//  Public BigInt wrappers.
// ----------------------------------------------------------------------------
void divmod_knuth(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r) {
    assert(a.sign >= 0 && b.sign > 0 && "divmod_knuth expects a >= 0, b > 0");
    std::vector<uint32_t> qm, rm;
    divmod_mag(a.mag, b.mag, qm, rm);
    q.mag = std::move(qm); q.sign = q.mag.empty() ? 0 : 1; q.normalize();
    r.mag = std::move(rm); r.sign = r.mag.empty() ? 0 : 1; r.normalize();
}

BigInt div_fast(const BigInt& a, const BigInt& b) {
    // Placeholder for a future sub-quadratic divider. Today it forwards to the
    // exact Knuth division so every caller already gets a correct result; only
    // the asymptotic speed of very large divisions will improve when a Newton or
    // Burnikel-Ziegler implementation replaces this body.
    BigInt q, r;
    divmod_knuth(a, b, q, r);
    return q;
}

// ----------------------------------------------------------------------------
//  Integer square root: floor(sqrt(s)).
//
//  Newton's method for sqrt iterates  x <- (x + s/x) / 2. Started from any value
//  >= sqrt(s) and using FLOOR division throughout, the sequence decreases and
//  lands exactly on floor(sqrt(s)); we stop as soon as it stops decreasing. Each
//  step costs one big division (currently Knuth), and the quadratic convergence
//  means only ~log2(bits) steps are needed.
// ----------------------------------------------------------------------------
BigInt isqrt(const BigInt& s) {
    if (s.sign <= 0) return BigInt();            // sqrt(0)=0; negatives -> 0 by convention
    size_t bits = s.bit_length();
    if (bits <= 2) return BigInt((int64_t)1);    // s in {1,2,3} -> floor sqrt = 1

    // Initial overestimate x0 = 2^(ceil(bits/2)) >= sqrt(s) (since s < 2^bits).
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

// Slow, obviously-correct reference: the classic base-2 digit-by-digit sqrt.
// Used only by the test suite to validate the Newton version.
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

// ----------------------------------------------------------------------------
//  Decimal output: convert a base-2^32 magnitude to a base-10 string.
//
//  We repeatedly divide the number by 10^9 (which fits in a single 32-bit limb),
//  and each remainder is the next group of up to 9 decimal digits, produced from
//  least significant to most significant. Then we assemble the groups: the most
//  significant group has no leading zeros; every other group is padded to exactly
//  9 digits. This is O(d^2) in the digit count d, but uses only the very fast
//  single-limb division inner loop. (A sub-quadratic divide-and-conquer variant
//  is described in docs/08_scaling_to_billions.md.)
// ----------------------------------------------------------------------------
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
        while (!cur.empty() && cur.back() == 0) cur.pop_back(); // normalize
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

} // namespace pidigits
