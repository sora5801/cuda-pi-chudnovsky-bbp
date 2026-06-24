// ============================================================================
//  chudnovsky.cpp  --  Chudnovsky series + binary splitting -> pi
// ============================================================================
//
//  This file turns the math in chudnovsky.h into running code. It has two parts:
//
//    1. chudnovsky_bs(a, b): the recursive binary splitting that produces the
//       integer triple (P, Q, T) for the term range [a, b).
//    2. compute_pi_chudnovsky(digits): the final assembly that evaluates
//       pi = 426880 * sqrt(10005) * Q / T in fixed-point integer arithmetic and
//       formats the result as a decimal string.
//
//  Every multiplication here ultimately flows through BigInt::operator*, which
//  dispatches to schoolbook / Karatsuba / CPU-NTT / CUDA-NTT by size. So simply
//  by leaving the multiply backend on Auto, the big multiplications near the root
//  of the splitting tree run on the GPU.
// ============================================================================

#include "chudnovsky.h"
#include "bignum.h"

#include <chrono>
#include <cstdio>
#include <cmath>

namespace pidigits {

// --- Chudnovsky numeric constants (verified; see docs/01_chudnovsky.md) -----
namespace {
const uint64_t A_CONST   = 13591409ULL;            // constant term of the linear factor
const uint64_t B_CONST   = 545140134ULL;           // linear coefficient
// C3_OVER_24 = 640320^3 / 24 = 10939058860032000 (exact). Appears in q(k)=k^3 * this.
const uint64_t C3_OVER_24 = 10939058860032000ULL;
const uint64_t L_CONST   = 426880ULL;              // leading constant (= 640320 * 2/3)
const uint64_t R_CONST   = 10005ULL;               // radicand: pi uses sqrt(10005)
// Each term is worth this many decimal digits: log10(C3_OVER_24 / 72).
const double DIGITS_PER_TERM = 14.181647462725477;

// 10^e via binary exponentiation (square-and-multiply). Unlike the cached pow10
// in bignum.cpp, this does NOT keep every intermediate power, so it is safe to
// call with the very large exponents used for fixed-point scaling.
BigInt pow10_big(uint64_t e) {
    BigInt result((int64_t)1);
    BigInt base((int64_t)10);
    while (e) {
        if (e & 1) result = result * base;
        base = base * base;
        e >>= 1;
    }
    return result;
}
} // anonymous namespace

uint64_t chudnovsky_terms_for_digits(uint64_t digits) {
    // +2 gives a little headroom so series truncation error stays below the
    // requested precision.
    uint64_t n = (uint64_t)((double)digits / DIGITS_PER_TERM) + 2;
    return n < 2 ? 2 : n;
}

// ----------------------------------------------------------------------------
//  Binary splitting for the half-open range [a, b).
// ----------------------------------------------------------------------------
PQT chudnovsky_bs(uint64_t a, uint64_t b) {
    PQT r;

    // --- Base case: a single term, range width 1 ----------------------------
    if (b - a == 1) {
        if (a == 0) {
            // The k=0 term: p(0) and q(0) are special-cased to 1 so that the term
            // reduces to the constant 13591409. (See the derivation in the docs.)
            r.P = BigInt((int64_t)1);
            r.Q = BigInt((int64_t)1);
            r.T = BigInt::from_u64(A_CONST);
        } else {
            // p(a) = (6a-5)(2a-1)(6a-1). Each factor fits in 64 bits for any a we
            // will ever use, so we build them with from_u64 and multiply.
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

    // --- Recursive case: split at the midpoint and merge --------------------
    uint64_t m = (a + b) / 2;
    PQT L = chudnovsky_bs(a, m);
    PQT R = chudnovsky_bs(m, b);

    // Merge rules (the heart of binary splitting):
    r.P = L.P * R.P;
    r.Q = L.Q * R.Q;
    r.T = L.T * R.Q + L.P * R.T;
    return r;
}

// ----------------------------------------------------------------------------
//  Full pi computation and formatting.
// ----------------------------------------------------------------------------
ChudnovskyResult compute_pi_chudnovsky(uint64_t digits, bool verbose) {
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();
    auto stamp = [&](const char* label, clock::time_point a, clock::time_point b) {
        if (verbose)
            std::fprintf(stderr, "  [chudnovsky] %-22s %8.1f ms\n", label,
                         std::chrono::duration<double, std::milli>(b - a).count());
    };

    // Guard digits absorb the tiny floor() errors from isqrt and the final
    // division, so the digits we keep are all correct.
    const uint64_t guard = 16;
    const uint64_t prec  = digits + guard;
    const uint64_t N     = chudnovsky_terms_for_digits(prec);

    // 1) Binary splitting -> (P, Q, T).
    auto t0 = clock::now();
    PQT r = chudnovsky_bs(0, N);
    auto t1 = clock::now();
    stamp("binary splitting", t0, t1);

    // 2) sqrt(10005) scaled to `prec` digits: isqrt(10005 * 10^(2*prec))
    //    yields floor(sqrt(10005) * 10^prec).
    BigInt one = pow10_big(prec);                       // 10^prec
    BigInt radicand = BigInt::from_u64(R_CONST) * one * one; // 10005 * 10^(2 prec)
    auto t2 = clock::now();
    BigInt sqrtC = isqrt(radicand);
    auto t3 = clock::now();
    stamp("sqrt(10005)", t2, t3);

    // 3) Numerator = 426880 * Q * sqrtC, then pi*10^prec = numerator / T.
    BigInt numerator = mul_small(r.Q, (int64_t)L_CONST) * sqrtC;
    auto t4 = clock::now();
    BigInt pi_scaled = div_fast(numerator, r.T);        // floor(pi * 10^prec)
    auto t5 = clock::now();
    stamp("final division", t4, t5);

    // 4) Format. pi_scaled is "314159..." with prec+1 digits; drop the guard
    //    digits and insert the decimal point after the leading 3.
    std::string s = pi_scaled.to_decimal();
    std::string out;
    if (digits == 0) {
        out = "3";
    } else {
        out.reserve(digits + 2);
        out.push_back(s[0]);                 // the integer part '3'
        out.push_back('.');
        out.append(s, 1, (size_t)digits);    // the first `digits` fractional digits
    }
    auto t6 = clock::now();
    stamp("format", t5, t6);

    ChudnovskyResult res;
    res.pi = std::move(out);
    res.terms = N;
    res.seconds = std::chrono::duration<double>(clock::now() - t_start).count();
    return res;
}

} // namespace pidigits
