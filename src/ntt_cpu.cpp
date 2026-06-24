// ============================================================================
//  ntt_cpu.cpp  --  Big-integer multiplication via a CPU Number Theoretic
//                   Transform over the Goldilocks field GF(2^64 - 2^32 + 1)
// ============================================================================
//
//  THE BIG PICTURE
//  ---------------
//  Multiplying two integers is the same as convolving their digit sequences:
//  if A has digits a_i and B has digits b_j (in some base), then the product's
//  digit-weights are c_k = sum_{i+j=k} a_i*b_j  (before carrying). Schoolbook
//  computes that convolution directly in O(n^2). The CONVOLUTION THEOREM lets us
//  do it in O(n log n) instead:
//
//      convolution(A, B) = InverseTransform( Transform(A) .* Transform(B) )
//
//  where ".*" is element-wise (pointwise) multiplication. With a Fast Fourier
//  Transform the "Transform" is the DFT; here we use the NUMBER THEORETIC
//  TRANSFORM (NTT), which is the exact same Cooley-Tukey butterfly structure but
//  carried out in a finite field GF(p) using a primitive root of unity omega in
//  place of the complex number e^{2*pi*i/N}. Working in a field of integers means
//  every value is exact -- no floating-point rounding -- which is exactly what we
//  need for provably-correct digits of pi.
//
//  This file is the CPU reference for that idea. ntt_cuda.cu implements the very
//  same algorithm on the GPU; the test suite multiplies the two against each
//  other and against schoolbook to prove all three agree.
//
//  PIPELINE (mul_ntt_cpu)
//  ----------------------
//     1. Repack each operand's 32-bit limbs into 16-bit transform digits
//        (so coefficients stay below p -- see goldilocks.h).
//     2. Zero-pad both to a common power-of-two length N.
//     3. Forward-NTT each operand.
//     4. Pointwise-multiply the two spectra in GF(p).
//     5. Inverse-NTT the product spectrum -> the exact convolution coefficients.
//     6. Carry-propagate the coefficients back into base-2^32 limbs.
// ============================================================================

#include "ntt.h"
#include "bignum.h"

#include <vector>
#include <utility>  // std::swap

namespace pidigits {

// ----------------------------------------------------------------------------
//  The in-place iterative radix-2 Cooley-Tukey NTT.
//
//  `a` is both input and output, length N (a power of two). `inverse == false`
//  computes the forward transform; `inverse == true` computes the inverse
//  transform AND applies the 1/N normalization at the end.
//
//  Structure:
//    * First permute the array into bit-reversed index order. This is what lets
//      the transform run "in place" with the classic butterfly pattern.
//    * Then combine size-2 blocks into size-4, size-8, ... up to size-N. At each
//      block size `len`, we use a primitive len-th root of unity w_len and walk a
//      "twiddle" w = w_len^j across each half-block, doing butterflies:
//          a[i+j]      = u + w*v
//          a[i+j+len/2]= u - w*v        (all arithmetic in GF(p))
//      where u = a[i+j], v = a[i+j+len/2].
// ----------------------------------------------------------------------------
static void ntt_transform(std::vector<uint64_t>& a, bool inverse) {
    const size_t n = a.size();
    if (n <= 1) return;
    const int logn = ntt_log2(n);

    // --- Step 1: bit-reversal permutation -----------------------------------
    // Move a[i] to a[bitreverse(i)]. We only swap when i < j so each pair is
    // exchanged exactly once.
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t j = ntt_bit_reverse(i, logn);
        if (i < j) std::swap(a[i], a[j]);
    }

    // --- Step 2: butterfly stages, block size 2, 4, 8, ..., N ---------------
    for (size_t len = 2; len <= n; len <<= 1) {
        // The primitive len-th root of unity for this stage (its inverse when we
        // are doing the inverse transform). omega^len == 1 in GF(p).
        uint64_t w_len = gl_root_of_unity((uint64_t)len);
        if (inverse) w_len = gl_inv(w_len);

        const size_t half = len >> 1;
        for (size_t i = 0; i < n; i += len) {      // each block of size len
            uint64_t w = 1;                         // twiddle starts at w_len^0 = 1
            for (size_t j = 0; j < half; ++j) {
                uint64_t u = a[i + j];
                uint64_t v = gl_mul(a[i + j + half], w);
                a[i + j]        = gl_add(u, v);     // even output
                a[i + j + half] = gl_sub(u, v);     // odd output
                w = gl_mul(w, w_len);               // advance twiddle: w *= w_len
            }
        }
    }

    // --- Step 3: normalization for the inverse transform --------------------
    // The forward-then-inverse round trip multiplies everything by N, so the
    // inverse must divide by N, i.e. multiply by N^{-1} in the field.
    if (inverse) {
        uint64_t ninv = gl_inv((uint64_t)n);
        for (size_t i = 0; i < n; ++i) a[i] = gl_mul(a[i], ninv);
    }
}

// ----------------------------------------------------------------------------
//  The public entry point: multiply two magnitudes via the CPU NTT.
//  (Declared in bignum.h; reached through mul_dispatch when the operands are
//   large and the configured backend is NttCpu.)
// ----------------------------------------------------------------------------
std::vector<uint32_t> mul_ntt_cpu(const std::vector<uint32_t>& a,
                                  const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};            // x * 0 = 0

    const size_t n = ntt_length(a.size(), b.size());  // power-of-two transform size

    std::vector<uint64_t> fa = repack_to16(a, n);     // operand A as padded 16-bit digits
    std::vector<uint64_t> fb = repack_to16(b, n);     // operand B as padded 16-bit digits

    ntt_transform(fa, false);                         // A -> spectrum
    ntt_transform(fb, false);                         // B -> spectrum

    for (size_t i = 0; i < n; ++i)                    // pointwise product of spectra
        fa[i] = gl_mul(fa[i], fb[i]);

    ntt_transform(fa, true);                          // spectrum -> convolution coefficients

    return carry_propagate(fa);                       // coefficients -> base-2^32 magnitude
}

} // namespace pidigits
