// ============================================================================
//  ntt.h  --  Shared helpers for NTT-based big-integer multiplication
// ============================================================================
//
//  This header holds the pieces that the CPU NTT (ntt_cpu.cpp) and the CUDA NTT
//  (ntt_cuda.cu) BOTH need, so the two implementations agree bit-for-bit on how
//  numbers are sliced up and reassembled:
//
//     * repack_to16()      : split base-2^32 limbs into 16-bit transform digits
//     * carry_propagate()  : turn convolution coefficients back into base-2^32
//     * ntt_length()       : choose the power-of-two transform size
//
//  The actual transforms (forward/inverse butterflies) live in the two .cpp/.cu
//  files because one runs on CPU threads and the other in CUDA kernels. The math
//  they implement is identical and is documented in docs/05_ntt_goldilocks.md.
//
//  WHY 16-BIT DIGITS?
//  ------------------
//  See the long comment in goldilocks.h: packing the big integer into 16-bit
//  pieces keeps every convolution coefficient below the prime p, so a single
//  Goldilocks prime recovers the exact product with no Chinese-Remainder step.
//  The price is 2x as many transform points as a 32-bit packing would use; the
//  payoff is dramatically simpler, obviously-correct code.
// ============================================================================

#pragma once

#include "goldilocks.h"
#include <cstdint>
#include <vector>
#include <cstddef>

namespace pidigits {

// Smallest power of two that is >= x (and at least 1). Used to pick the
// transform length, which must be a power of two for a radix-2 NTT.
inline size_t ntt_next_pow2(size_t x) {
    size_t n = 1;
    while (n < x) n <<= 1;
    return n;
}

// Choose the transform length for multiplying an `na`-limb number by an
// `nb`-limb number. Each 32-bit limb becomes two 16-bit digits, so the operands
// have 2*na and 2*nb digits; their product (a linear convolution) has up to
// 2*na + 2*nb - 1 digits. We round that up to a power of two.
inline size_t ntt_length(size_t na, size_t nb) {
    size_t conv_len = 2 * na + 2 * nb;     // (2na + 2nb - 1) rounded a touch high
    return ntt_next_pow2(conv_len);
}

// Split a base-2^32 magnitude into 16-bit transform digits (as field elements).
// Limb L becomes two digits: the low 16 bits (L & 0xFFFF) then the high 16 bits
// (L >> 16). The result is zero-padded out to `length` (the transform size).
// Storing them as uint64_t up front means the transform can run in place.
inline std::vector<uint64_t> repack_to16(const std::vector<uint32_t>& a, size_t length) {
    std::vector<uint64_t> out(length, 0);
    size_t k = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        out[k++] = a[i] & 0xFFFFu;          // low 16 bits  (digit weight 2^(32i))
        out[k++] = (a[i] >> 16) & 0xFFFFu;  // high 16 bits (digit weight 2^(32i+16))
    }
    return out;
}

// Turn the convolution coefficients (each an exact non-negative integer < p,
// the product sums in base 2^16) back into a normalized base-2^32 magnitude.
// We walk low-to-high, add the running carry, peel off 16 bits per digit, then
// glue digit pairs into 32-bit limbs.
inline std::vector<uint32_t> carry_propagate(const std::vector<uint64_t>& conv) {
    std::vector<uint32_t> out;
    out.reserve(conv.size() / 2 + 2);
    uint64_t carry = 0;
    // Emit 16-bit digits two at a time, combining each pair into one 32-bit limb.
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

// Bit-reversal of the low `log2n` bits of `i`. The iterative radix-2 NTT visits
// inputs in bit-reversed order; both the CPU and GPU transforms use this.
inline uint32_t ntt_bit_reverse(uint32_t i, int log2n) {
    uint32_t r = 0;
    for (int b = 0; b < log2n; ++b) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

// Integer log2 of a power of two.
inline int ntt_log2(size_t n) {
    int l = 0;
    while ((size_t(1) << l) < n) ++l;
    return l;
}

} // namespace pidigits
