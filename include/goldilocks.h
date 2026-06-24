// ============================================================================
//  goldilocks.h  --  Arithmetic in the "Goldilocks" finite field GF(p)
//                    with  p = 2^64 - 2^32 + 1
// ============================================================================
//
//  WHAT IS THIS AND WHY DOES PI NEED IT?
//  -------------------------------------
//  Our fast big-integer multiply (see ntt.h / ntt_cpu.cpp / ntt_cuda.cu) is a
//  Number Theoretic Transform (NTT) -- an FFT done with INTEGERS modulo a prime
//  instead of complex floating-point numbers. Doing the transform in a prime
//  field means every intermediate value is an EXACT integer, so the final
//  product has no rounding error at all. That exactness is essential when you
//  want millions of *correct* digits of pi.
//
//  An NTT needs a prime p such that the transform length N (a power of two)
//  divides p-1, so that a primitive N-th "root of unity" exists in GF(p) -- the
//  integer analogue of e^{2*pi*i/N}. We use the celebrated GOLDILOCKS PRIME:
//
//      p = 2^64 - 2^32 + 1 = 18446744069414584321 = 0xFFFFFFFF00000001
//
//  It is famous in zero-knowledge-proof systems (Plonky2, RISC Zero, ...) for
//  three beautiful properties:
//
//    1. It is just under 2^64, so a field element fits in a single uint64_t.
//    2. p - 1 = 2^32 * (2^32 - 1) = 2^32 * 3 * 5 * 17 * 257 * 65537.
//       The factor 2^32 means transform lengths up to 2^32 are available
//       (its "two-adicity" is 32) -- astronomically more than we will ever use.
//    3. Its special shape gives a branch-light modular reduction (no division!):
//          2^64 == 2^32 - 1  (mod p)        [because p = 2^64 - (2^32 - 1)]
//          2^96 == -1        (mod p)
//       so a 128-bit product can be folded back into [0, p) with a few shifts,
//       multiplies by the small constant EPSILON = 2^32 - 1, adds, and subtracts.
//       This is a "Solinas reduction". On a GPU, avoiding 64-bit integer
//       division (which is emulated and slow) is a big win.
//
//  WHY A SINGLE PRIME IS ENOUGH (the "no CRT needed" argument)
//  ----------------------------------------------------------
//  To multiply two big integers we slice each into 16-bit pieces and convolve
//  them. A convolution output coefficient is a sum of products of 16-bit pieces:
//      c_k = sum a_i * b_j  with a_i, b_j < 2^16, so each product < 2^32,
//  and there are at most N/2 such products, giving  c_k < (N/2) * 2^32 = N*2^31.
//  For any transform length N <= 2^32 this is < 2^63 < p. Because every true
//  coefficient is already smaller than p, the field result EQUALS the integer
//  result -- nothing wraps around -- so one prime recovers the exact product.
//  (Numbers big enough to break this have ~1.6 billion digits; we run out of GPU
//  memory long before that.) Implementations that pack wider digits must combine
//  several primes with the Chinese Remainder Theorem; we sidestep all of that.
//
//  HOST + DEVICE
//  -------------
//  Every function here is marked GL_HD so the SAME code compiles for the CPU
//  (plain C++ in a .cpp) and the GPU (device code in a .cu). The only piece that
//  differs per platform is the 64x64 -> high-64-bits multiply, because there is
//  no portable 128-bit integer type on MSVC/Windows; we dispatch to the right
//  intrinsic (`__umul64hi` on the GPU, `__umulh` on MSVC host).
// ============================================================================

#pragma once

#include <cstdint>

// On the host (MSVC) we need <intrin.h> for __umulh (high 64 bits of a 64x64
// product). nvcc defines __CUDACC__ while compiling .cu files; for the host
// passes of those files we still want the intrinsic, so include it whenever we
// are not generating device code.
#if defined(_MSC_VER)
#  include <intrin.h>
#endif

// GL_HD expands to the CUDA host/device qualifiers when compiled by nvcc, and to
// nothing when compiled by a plain C++ compiler. This lets the identical source
// serve both worlds.
#if defined(__CUDACC__)
#  define GL_HD __host__ __device__ __forceinline__
#else
#  define GL_HD inline
#endif

namespace pidigits {

// --- Field constants --------------------------------------------------------
// The prime, the small reduction constant, and a primitive root of the field.
constexpr uint64_t GL_P   = 0xFFFFFFFF00000001ULL; // 2^64 - 2^32 + 1
constexpr uint64_t GL_EPS = 0xFFFFFFFFULL;         // 2^32 - 1  (== 2^64 mod p)
constexpr uint64_t GL_G   = 7ULL;                  // smallest primitive root of GF(p)

// --- 64x64 -> high 64 bits of the product -----------------------------------
// Returns floor((a*b) / 2^64). Combined with the wrapping low product `a*b`,
// this gives the full 128-bit product without needing a 128-bit type.
GL_HD uint64_t gl_mulhi(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);                 // device intrinsic
#elif defined(_MSC_VER) && defined(_M_X64)
    return __umulh(a, b);                     // MSVC x64 host intrinsic
#else
    return (uint64_t)(((unsigned __int128)a * b) >> 64); // portable fallback (non-MSVC)
#endif
}

// --- Modular addition in [0, p) ---------------------------------------------
// a + b might exceed 2^64. If it wraps (carry out of bit 63->64), the true sum
// is (wrapped + 2^64); since 2^64 == EPS (mod p) we add EPS to fold the carry.
// Then a single conditional subtract canonicalizes into [0, p).
GL_HD uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a) s += GL_EPS;       // carry out of 64 bits -> add 2^64 == EPS (mod p)
    if (s >= GL_P) s -= GL_P;      // canonicalize down into [0, p)
    return s;
}

// --- Modular subtraction in [0, p) ------------------------------------------
// If a < b the bare a-b borrows (wraps by +2^64). The true value is a-b+p; since
// p = 2^64 - EPS, that equals (wrapped value) - EPS, so we subtract EPS.
GL_HD uint64_t gl_sub(uint64_t a, uint64_t b) {
    uint64_t d = a - b;
    if (a < b) d -= GL_EPS;        // borrow -> result is short by EPS, correct it
    return d;
}

// --- Solinas reduction of a 128-bit value (hi:lo) into [0, p) ---------------
// This is the heart of Goldilocks arithmetic. It is the plonky2 "reduce128"
// algorithm, which exploits 2^64 == EPS and 2^96 == -1 (mod p). Derivation:
//   x = lo + 2^64 * hi
//     = lo + 2^64 * (hi_lo + 2^32 * hi_hi)              [split hi into 32-bit halves]
//     = lo + hi_lo * 2^64        + hi_hi * 2^96
//     = lo + hi_lo * EPS         - hi_hi                (mod p)   [apply the two congruences]
// We compute lo - hi_hi (with borrow correction), then add hi_lo*EPS (which fits
// in 64 bits because both factors are < 2^32), folding any carry, and finally
// canonicalize. The result is exactly x mod p in [0, p).
GL_HD uint64_t gl_reduce128(uint64_t lo, uint64_t hi) {
    uint32_t hi_hi = (uint32_t)(hi >> 32);          // bits [96,128) of x, weight 2^96 == -1
    uint32_t hi_lo = (uint32_t)(hi & 0xFFFFFFFFu);  // bits [64,96)  of x, weight 2^64 == EPS

    // t0 = lo - hi_hi  (the "2^96 == -1" contribution). On borrow, the wrap is by
    // 2^64; since 2^64 == EPS (mod p), subtract EPS to correct.
    uint64_t t0 = lo - (uint64_t)hi_hi;
    if (lo < (uint64_t)hi_hi) t0 -= GL_EPS;

    // t1 = hi_lo * EPS  (the "2^64 == EPS" contribution). hi_lo < 2^32 and
    // EPS < 2^32, so the product is < 2^64 and cannot overflow.
    uint64_t t1 = (uint64_t)hi_lo * GL_EPS;

    // t2 = t0 + t1, folding a possible carry (again 2^64 == EPS). Two foldings at
    // most can be needed in pathological cases, so we loop the carry fold and then
    // canonicalize with a short loop (runs at most twice).
    uint64_t t2 = t0 + t1;
    if (t2 < t0) {                 // carry out of 64 bits
        uint64_t before = t2;
        t2 += GL_EPS;
        if (t2 < before) t2 += GL_EPS; // extremely rare second carry
    }
    while (t2 >= GL_P) t2 -= GL_P;  // canonicalize into [0, p)
    return t2;
}

// --- Modular multiplication in [0, p) ---------------------------------------
// Full 128-bit product (low = a*b wrapping, high = gl_mulhi), then reduce.
GL_HD uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b;            // low 64 bits (wraps mod 2^64 -- exactly what we want)
    uint64_t hi = gl_mulhi(a, b);   // high 64 bits
    return gl_reduce128(lo, hi);
}

// --- Modular exponentiation: base^exp mod p (binary / square-and-multiply) ---
GL_HD uint64_t gl_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp) {
        if (exp & 1) result = gl_mul(result, base);
        base = gl_mul(base, base);
        exp >>= 1;
    }
    return result;
}

// --- Modular inverse via Fermat's little theorem: a^(p-2) == a^{-1} (mod p) --
// Valid for any a != 0. We only ever invert the transform length N and roots of
// unity, all of which are non-zero, so Fermat is the simplest correct choice.
GL_HD uint64_t gl_inv(uint64_t a) {
    return gl_pow(a, GL_P - 2);
}

// --- A primitive N-th root of unity, for N a power of two with N | (p-1) ------
// omega = g^((p-1)/N) has multiplicative order exactly N, i.e. omega^N == 1 and
// no smaller power equals 1. This is the integer stand-in for e^{2*pi*i/N} that
// makes the NTT a genuine transform. (For N <= 2^32 the division (p-1)/N is exact
// because 2^32 divides p-1.)
GL_HD uint64_t gl_root_of_unity(uint64_t N) {
    return gl_pow(GL_G, (GL_P - 1) / N);
}

} // namespace pidigits
