// ============================================================================
//  bignum.h  --  Arbitrary-precision (big) integer type for the Pi project
// ============================================================================
//
//  WHY DO WE NEED THIS AT ALL?
//  ---------------------------
//  To compute pi to (potentially) billions of digits we must do exact integer
//  arithmetic on numbers that are MILLIONS of bits long. A 64-bit `unsigned
//  long long` holds about 19 decimal digits. A number with 1,000,000 decimal
//  digits needs ~3.3 million bits, i.e. ~52,000 such 64-bit words. No built-in
//  C++ type can hold that, so we build our own "big integer".
//
//  The Chudnovsky algorithm (see chudnovsky.h) works by computing a single
//  gigantic rational number P/Q and a numerator T using *binary splitting*,
//  which is nothing but a tree of big-integer multiplications and additions.
//  So the performance of the whole program is dominated by how fast we can
//  MULTIPLY two enormous integers. That is exactly why this project lives in a
//  directory called CUDA_MATH_FAST_MUL: the interesting work is fast big-integer
//  multiplication, accelerated on the GPU.
//
//  MEMORY LAYOUT (this matters for performance and is worth internalizing)
//  ----------------------------------------------------------------------
//  A BigInt stores its magnitude as a `std::vector<uint32_t>` called `mag`.
//
//      mag[0] is the LEAST significant 32-bit "limb" (little-endian).
//      mag[1] is the next 32 bits up, and so on.
//
//  The mathematical value of the magnitude is:
//
//      |value| = sum over i of  mag[i] * (2^32)^i
//
//  We call each element a "limb" (standard bignum terminology, like a limb of a
//  tree). The base of our positional number system is B = 2^32 = 4294967296.
//  We choose 2^32 because:
//    * A product of two 32-bit limbs fits in a 64-bit register (32+32 = 64 bits),
//      so schoolbook multiplication never overflows a uint64_t accumulator
//      until we add many of them -- and even then carries are easy to manage.
//    * Bit shifts (needed by Newton's method for division and square root) are
//      natural in a power-of-two base.
//    * Repacking into the 16-bit words that the NTT wants is a trivial split.
//
//  We keep the vector *normalized*: there are never leading zero limbs, so the
//  most-significant limb (mag.back()) is always non-zero, except for the number
//  zero whose `mag` is empty.
//
//  SIGN
//  ----
//  `sign` is one of {-1, 0, +1}. It is 0 if and only if the magnitude is zero.
//  This "sign-magnitude" representation keeps the magnitude code (the hard,
//  performance-critical part) completely sign-agnostic; the few places that
//  care about sign (add/sub/compare) handle it explicitly.
//
//  This header declares the type and its operations. The actual algorithms live
//  in src/bignum.cpp (basic arithmetic, schoolbook & Karatsuba multiply,
//  division, square root, decimal conversion). The fast multiply backends live
//  in src/ntt_cpu.cpp and src/ntt_cuda.cu and are reached through mul_dispatch().
// ============================================================================

#pragma once

#include <cstdint>   // fixed-width integer types: uint32_t, uint64_t, ...
#include <vector>    // std::vector, our limb container
#include <string>    // std::string, for decimal I/O
#include <cstddef>   // size_t

namespace pidigits {

// ----------------------------------------------------------------------------
//  Multiplication backend selection.
//
//  We support several big-integer multiplication algorithms, from the simple
//  and obviously-correct to the fast and GPU-accelerated. Tests run ALL of them
//  against each other to prove the fast ones are correct (see tests/). At run
//  time, mul_dispatch() picks an algorithm based on operand size and the user's
//  configured preference.
// ----------------------------------------------------------------------------
enum class MulBackend {
    Auto,        // choose automatically by size (the normal mode)
    Schoolbook,  // O(n^2)        -- the definition of multiplication; reference
    Karatsuba,   // O(n^1.585)    -- divide & conquer; classic sub-quadratic
    NttCpu,      // O(n log n)    -- Number Theoretic Transform on the CPU
    NttCuda      // O(n log n)    -- the same NTT, but the transforms run on the GPU
};

// Global knobs that control how multiplication behaves. These are plain globals
// (defined in bignum.cpp) so the CLI, demo, and tests can flip them easily.
// In a larger codebase you might wrap these in a context object; for a study
// project, visible global state keeps the data flow easy to follow.
struct MulConfig {
    MulBackend backend = MulBackend::Auto; // which algorithm family to use
    // Size thresholds (measured in 32-bit limbs of the smaller operand) at which
    // Auto mode switches algorithms. Below karatsuba_threshold we use schoolbook;
    // between the two we use Karatsuba; at/above ntt_threshold we use an NTT.
    size_t karatsuba_threshold = 32;
    size_t ntt_threshold       = 256;
    bool   prefer_cuda         = true;  // when an NTT is chosen, use the GPU if available
    bool   verbose             = false; // print which backend each big multiply used
};

// The single global multiply configuration.
extern MulConfig g_mul;

// ----------------------------------------------------------------------------
//  The big integer type.
// ----------------------------------------------------------------------------
class BigInt {
public:
    std::vector<uint32_t> mag; // little-endian base-2^32 magnitude, normalized
    int sign = 0;              // -1, 0, or +1; 0 iff mag is empty

    // --- Constructors -------------------------------------------------------
    BigInt() = default;                 // value 0
    BigInt(int64_t v);                  // from a signed 64-bit integer
    static BigInt from_u64(uint64_t v); // from an unsigned 64-bit integer
    static BigInt from_decimal(const std::string& s); // parse "-123..." (any length)

    // --- Basic queries ------------------------------------------------------
    bool is_zero() const { return sign == 0; }
    bool is_negative() const { return sign < 0; }
    size_t limb_count() const { return mag.size(); }     // number of 32-bit limbs
    size_t bit_length() const;                           // index of highest set bit + 1

    // Remove any leading zero limbs and force sign to 0 if the magnitude is zero.
    // Every operation that builds a BigInt ends by calling this so the invariant
    // "no leading zeros, sign==0 iff zero" always holds.
    void normalize();

    // --- Comparison ---------------------------------------------------------
    // cmp_mag compares ONLY magnitudes (ignores sign): returns -1, 0, +1.
    int cmp_mag(const BigInt& o) const;
    // cmp is a full signed comparison: returns -1, 0, +1.
    int cmp(const BigInt& o) const;

    // --- Sign manipulation --------------------------------------------------
    BigInt operator-() const;           // arithmetic negation
    BigInt abs() const;                 // absolute value

    // --- Bit shifts (by a number of BITS, not limbs) ------------------------
    // These implement multiply/divide by powers of two, which Newton's method
    // for reciprocals and square roots relies on heavily.
    BigInt shl(size_t bits) const;      // value * 2^bits
    BigInt shr(size_t bits) const;      // floor(value / 2^bits)  (toward -inf for negatives? see .cpp)

    // --- Decimal conversion -------------------------------------------------
    // Convert to a base-10 string. Uses a divide-and-conquer scheme built on the
    // fast multiply and division so it stays sub-quadratic for huge numbers.
    std::string to_decimal() const;

    // A tiny helper: the low 64 bits of the magnitude as a uint64_t (ignores
    // sign and anything above bit 64). Handy for tests and small constants.
    uint64_t low64() const;
};

// ----------------------------------------------------------------------------
//  Arithmetic operators (free functions; defined in bignum.cpp).
//  These handle sign; the heavy magnitude math is in internal helpers.
// ----------------------------------------------------------------------------
BigInt operator+(const BigInt& a, const BigInt& b);
BigInt operator-(const BigInt& a, const BigInt& b);
BigInt operator*(const BigInt& a, const BigInt& b);

inline BigInt& operator+=(BigInt& a, const BigInt& b) { a = a + b; return a; }
inline BigInt& operator-=(BigInt& a, const BigInt& b) { a = a - b; return a; }
inline BigInt& operator*=(BigInt& a, const BigInt& b) { a = a * b; return a; }

// Multiply by a small integer (fast path used a lot by binary splitting, whose
// leaves multiply by things like (6k-5)(2k-1)(6k-1) and the linear factor).
BigInt mul_small(const BigInt& a, uint32_t m);
BigInt mul_small(const BigInt& a, int64_t m);

// ----------------------------------------------------------------------------
//  Magnitude-level multiply dispatch and backends.
//
//  All of these operate on *magnitudes only*: little-endian base-2^32 limb
//  vectors with no sign. They return the product magnitude (normalized, i.e.
//  no trailing -- which is to say most-significant -- zero limbs). The signed
//  operator* above wraps these and applies the sign rule (sign = sa*sb).
// ----------------------------------------------------------------------------

// Pick a backend by size / configuration and multiply.
std::vector<uint32_t> mul_dispatch(const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b);

// O(n*m) schoolbook ("long multiplication"). Always correct; our ground truth.
std::vector<uint32_t> mul_schoolbook(const std::vector<uint32_t>& a,
                                     const std::vector<uint32_t>& b);

// O(n^1.585) Karatsuba. Splits each operand in half and uses 3 (not 4)
// half-size multiplications via the identity in bignum.cpp.
std::vector<uint32_t> mul_karatsuba(const std::vector<uint32_t>& a,
                                    const std::vector<uint32_t>& b);

// O(n log n) NTT multiply on the CPU. Declared here, defined in ntt_cpu.cpp.
std::vector<uint32_t> mul_ntt_cpu(const std::vector<uint32_t>& a,
                                  const std::vector<uint32_t>& b);

// O(n log n) NTT multiply on the GPU. Declared here, defined in ntt_cuda.cu.
// If the program was built without CUDA, the implementation in ntt_cuda_stub.cpp
// forwards to the CPU NTT so everything still works (just without the GPU).
std::vector<uint32_t> mul_ntt_cuda(const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b);

// Returns true if a usable CUDA device is present (defined in ntt_cuda.cu or the
// stub). Lets host code decide whether the GPU path is available at run time.
bool cuda_available();

// ----------------------------------------------------------------------------
//  Division, square root, and other higher-level operations (bignum.cpp).
// ----------------------------------------------------------------------------

// Floor division with remainder: given a >= 0 and b > 0, computes q = floor(a/b)
// and r = a - q*b (so 0 <= r < b). Uses Knuth's Algorithm D -- the exact,
// reference long-division algorithm. This is our correctness ground truth for
// division and is also used by decimal conversion.
void divmod_knuth(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r);

// Fast floor division a/b for large operands, using Newton's method to compute a
// reciprocal of b with the fast multiply, then a single multiply. Falls back to
// divmod_knuth for small inputs. Returns floor(a/b) for a,b > 0.
BigInt div_fast(const BigInt& a, const BigInt& b);

// Integer square root: floor(sqrt(s)) for s >= 0. Used to evaluate sqrt(10005)
// at full precision in the Chudnovsky final assembly. Implemented with a
// multiply-only Newton iteration on the reciprocal square root, then corrected
// to be exactly the floor. A simple bit-by-bit version (isqrt_bitwise) is kept
// for cross-checking in tests.
BigInt isqrt(const BigInt& s);
BigInt isqrt_bitwise(const BigInt& s); // slow but obviously-correct reference

// 10^exp as a BigInt (used for fixed-point scaling and base conversion). Cached.
const BigInt& pow10(size_t exp);

} // namespace pidigits
