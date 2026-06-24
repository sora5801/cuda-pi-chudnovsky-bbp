// ============================================================================
//  bbp_kernel.h  --  The Bailey-Borwein-Plouffe (BBP) digit-extraction math,
//                    written ONCE to run on both the CPU and the GPU.
// ============================================================================
//
//  WHAT MAKES BBP SPECIAL
//  ----------------------
//  The Chudnovsky algorithm must compute ALL digits up to position n to know the
//  n-th one. The BBP formula (1995) can compute the n-th *hexadecimal* digit of
//  pi DIRECTLY, using O(n log n) simple operations and O(1) memory -- without the
//  preceding digits. The formula is:
//
//      pi = sum_{k=0}^{inf} (1/16^k) ( 4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6) )
//
//  Define S_j = sum_{k>=0} 1 / (16^k (8k+j)). Then the hex digits of pi starting
//  just after position d are the leading hex digits of the fractional part of
//
//      16^d * pi  ==  4*{16^d S_1} - 2*{16^d S_4} - {16^d S_5} - {16^d S_6}   (mod 1)
//
//  where {x} is the fractional part. The magic is computing {16^d S_j} cheaply:
//
//      {16^d S_j} = { sum_{k=0}^{d}  (16^(d-k) mod (8k+j)) / (8k+j)
//                   + sum_{k=d+1}^{inf}     16^(d-k)       / (8k+j) }
//
//      * HEAD (k <= d): the exponent d-k is >= 0. We only care about the
//        FRACTIONAL part of 16^(d-k)/(8k+j), so we may replace the numerator by
//        16^(d-k) mod (8k+j), computed EXACTLY with modular exponentiation. Then
//        divide by (8k+j) in floating point and keep the running fractional part.
//      * TAIL (k > d): the exponent is negative, so 16^(d-k) shrinks like 16^-1,
//        16^-2, ...; only a dozen or so terms exceed double precision, so we sum
//        them directly and stop when they vanish.
//
//  EMBARRASSINGLY PARALLEL
//  -----------------------
//  Each hex position is computed completely independently -- no shared state, no
//  communication. That is a perfect fit for a GPU: we launch one thread PER
//  POSITION and each thread runs the routine below for its own d. This file holds
//  that routine, marked BBP_HD so the SAME code compiles for CPU and CUDA.
//
//  PRECISION CAVEAT (important, and honest)
//  ----------------------------------------
//  We accumulate the head/tail sums in IEEE-754 double (53-bit mantissa). That is
//  enough to get roughly the first ~10 hex digits of {16^d pi} correct. Since each
//  thread extracts only ONE hex digit (the leading one) and has ~10 digits of
//  slack, the result is reliable across the whole range we use, and the test suite
//  verifies it against known digits. Extracting many correct digits at one
//  position would instead require extended precision -- see docs/03_bbp.md.
// ============================================================================

#pragma once

#include "goldilocks.h"   // reuse gl_mulhi() for the portable 64x64 -> high-64 multiply
#include <cstdint>
#include <cmath>          // floor

// BBP_HD mirrors GL_HD: host+device under nvcc, plain inline otherwise.
#if defined(__CUDACC__)
#  define BBP_HD __host__ __device__ __forceinline__
#else
#  define BBP_HD inline
#endif

namespace pidigits {

// --- (a * b) mod m, exact -- the inner primitive of modular exponentiation -----
// When m < 2^32 the product a*b fits in 64 bits and a single hardware modulo
// suffices (the fast common path). For larger moduli (positions beyond ~500
// million) we form the full 128-bit product and reduce it 16 bits at a time,
// which is exact for any m < 2^48 (positions up to ~3.5e13).
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

// --- 16^e mod m, by binary (square-and-multiply) exponentiation ----------------
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

// --- {16^d * S_j} : the fractional part of 16^d times the j-th BBP sub-series ---
BBP_HD double bbp_series(uint32_t j, uint64_t d) {
    double sum = 0.0;

    // HEAD: k = 0 .. d. Each numerator is 16^(d-k) mod (8k+j), computed exactly.
    for (uint64_t k = 0; k <= d; ++k) {
        uint64_t m = 8ull * k + j;
        uint64_t r = bbp_modpow16(d - k, m);
        sum += (double)r / (double)m;
        sum -= floor(sum);                   // keep only the fractional part
    }

    // TAIL: k = d+1 .. while the term is still representable. Here 16^(d-k) is
    // 16^-1, 16^-2, ... shrinking geometrically.
    double p16 = 1.0 / 16.0;                 // 16^(d-k) for k = d+1
    uint64_t k = d + 1;
    while (p16 > 1e-18) {
        uint64_t m = 8ull * k + j;
        sum += p16 / (double)m;
        p16 /= 16.0;
        ++k;
    }
    sum -= floor(sum);
    return sum;
}

// --- The hex digit (0..15) at 1-based fractional position `pos` ----------------
// Convention (verified against Bailey): the first hex digit after the radix point
// is position 1. Computing {16^d * pi} with d = pos-1 yields the digit at `pos` as
// its leading hex digit.
BBP_HD int bbp_digit_value(uint64_t pos) {
    uint64_t d = pos - 1;
    double s = 4.0 * bbp_series(1, d)
             - 2.0 * bbp_series(4, d)
             - 1.0 * bbp_series(5, d)
             - 1.0 * bbp_series(6, d);
    s -= floor(s);                           // fractional part in [0, 1)
    if (s < 0.0) s += 1.0;                    // guard against tiny negative rounding
    return (int)(16.0 * s);                  // leading hex digit
}

} // namespace pidigits
