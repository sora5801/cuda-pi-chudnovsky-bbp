// ============================================================================
//  ntt_cuda.cu  --  GPU-accelerated big-integer multiplication
//                   (the same Goldilocks NTT as ntt_cpu.cpp, run on CUDA)
// ============================================================================
//
//  This is the payoff file of the whole project: the Number Theoretic Transform
//  that powers our fast multiply, executed on thousands of GPU threads. The math
//  is IDENTICAL to ntt_cpu.cpp -- read that file first for the "why". Here we are
//  only concerned with mapping that algorithm onto CUDA efficiently and safely.
//
//  HOW AN ITERATIVE NTT MAPS ONTO A GPU
//  ------------------------------------
//  The transform proceeds in log2(N) "stages", each combining blocks of size
//  `len` = 2, 4, 8, ..., N. Within a stage all N/2 butterflies are completely
//  independent, so each maps to its own GPU thread. BUT stage s+1 reads outputs
//  of stage s, so stages must be separated by a global barrier. On a GPU the
//  cheapest reliable global barrier is "end the kernel and launch the next one".
//  Hence: one kernel launch per stage (log2(N) launches), with N/2 threads each.
//
//  TWIDDLE FACTORS
//  ---------------
//  A butterfly multiplies by a power of a root of unity ("twiddle"). Recomputing
//  omega^j inside every thread would cost an extra O(log N) per butterfly. Instead
//  we precompute, ONCE, a table W[t] = omega_N^t for t in [0, N/2), where omega_N
//  is the primitive N-th root. The twiddle needed at stage `len`, position j, is
//  omega_len^j = omega_N^{ j * (N/len) }, so a stage just indexes W with a stride
//  of N/len. One table serves every stage. (See docs/05_ntt_goldilocks.md.)
//
//  ROBUSTNESS
//  ----------
//  If there is no CUDA device, or any CUDA call fails (e.g. the transform is too
//  big for GPU memory), mul_ntt_cuda() transparently falls back to the CPU NTT so
//  the program always produces a correct answer. Correctness never depends on the
//  GPU being present; the GPU only makes it faster.
// ============================================================================

#include "ntt.h"
#include "bignum.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

namespace pidigits {

// ----------------------------------------------------------------------------
//  A tiny CUDA error-checking helper. On any failure it records that the GPU
//  path is unusable for this call so the caller can fall back to the CPU.
// ----------------------------------------------------------------------------
#define CUDA_TRY(expr)                                                          \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "[cuda] %s failed: %s\n", #expr,               \
                         cudaGetErrorString(_e));                               \
            ok = false;                                                         \
        }                                                                       \
    } while (0)

// ----------------------------------------------------------------------------
//  Device kernels. Every one of these calls the gl_* field operations from
//  goldilocks.h, which are compiled for the device because they are __host__
//  __device__.
// ----------------------------------------------------------------------------

// Reverse the low `logn` bits of an index. CUDA's __brev reverses all 32 bits in
// one instruction; we shift the reversed value down so only `logn` bits remain.
__device__ __forceinline__ uint32_t dev_bitrev(uint32_t i, int logn) {
    return __brev(i) >> (32 - logn);
}

// Bit-reversal permutation: place a[i] at index bitrev(i). Each thread owns one
// index i and swaps with j = bitrev(i) only when i < j, so every pair is touched
// exactly once and there is no race.
__global__ void k_bit_reverse(uint64_t* a, uint32_t n, int logn) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t j = dev_bitrev(i, logn);
    if (i < j) {
        uint64_t tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}

// Build the twiddle table W[t] = omega^t, t in [0, n/2). Each thread computes one
// entry with a modular exponentiation (square-and-multiply). This is O((n/2) log n)
// total work but fully parallel and runs once per transform direction.
__global__ void k_build_twiddles(uint64_t* W, uint32_t half, uint64_t omega) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= half) return;
    W[t] = gl_pow(omega, (uint64_t)t);
}

// One butterfly stage for block size `len`. There are n/2 butterflies; thread b
// handles butterfly b. We decode b into its block and intra-block position j,
// fetch the twiddle from the shared table with the per-stage stride, and write
// the two outputs. `stride` = n/len selects omega_len^j out of the omega_n table.
__global__ void k_butterfly(uint64_t* a, uint32_t n, uint32_t len,
                            const uint64_t* W, uint32_t stride) {
    uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = len >> 1;
    if (b >= (n >> 1)) return;

    uint32_t block = b / half;          // which size-`len` block this butterfly is in
    uint32_t j     = b % half;          // position within the block's first half
    uint32_t i     = block * len;       // base index of the block

    uint64_t w = W[(uint64_t)j * stride]; // twiddle = omega_n^{ j*(n/len) } = omega_len^j
    uint64_t u = a[i + j];
    uint64_t v = gl_mul(a[i + j + half], w);
    a[i + j]        = gl_add(u, v);
    a[i + j + half] = gl_sub(u, v);
}

// Pointwise (Hadamard) product of two spectra: a[i] = a[i] * b[i] in GF(p).
__global__ void k_pointwise(uint64_t* a, const uint64_t* b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = gl_mul(a[i], b[i]);
}

// Final inverse-transform normalization: multiply every element by N^{-1}.
__global__ void k_scale(uint64_t* a, uint32_t n, uint64_t ninv) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = gl_mul(a[i], ninv);
}

// ----------------------------------------------------------------------------
//  Host-side orchestration of one transform on a device array d_a (length n).
//  `W` is a device twiddle table for the chosen direction (omega for forward,
//  omega^{-1} for inverse). For the inverse we also scale by n^{-1} afterward.
//  Returns false if any kernel launch failed.
// ----------------------------------------------------------------------------
static bool run_transform(uint64_t* d_a, uint32_t n, int logn,
                          const uint64_t* W, bool inverse) {
    bool ok = true;
    const int TPB = 256;                          // threads per block

    // Stage 0: reorder into bit-reversed indices.
    k_bit_reverse<<<(n + TPB - 1) / TPB, TPB>>>(d_a, n, logn);
    CUDA_TRY(cudaGetLastError());

    // Stages 1..logn: butterflies for len = 2, 4, ..., n.
    uint32_t halfN = n >> 1;
    for (uint32_t len = 2; len <= n; len <<= 1) {
        uint32_t stride = n / len;                // index step into the twiddle table
        k_butterfly<<<(halfN + TPB - 1) / TPB, TPB>>>(d_a, n, len, W, stride);
        CUDA_TRY(cudaGetLastError());
    }

    if (inverse) {
        uint64_t ninv = gl_inv((uint64_t)n);      // computed on the host (cheap)
        k_scale<<<(n + TPB - 1) / TPB, TPB>>>(d_a, n, ninv);
        CUDA_TRY(cudaGetLastError());
    }
    return ok;
}

// ----------------------------------------------------------------------------
//  cuda_available(): is there at least one usable CUDA device? Cached after the
//  first query so we do not probe the driver repeatedly.
// ----------------------------------------------------------------------------
bool cuda_available() {
    static int cached = -1;                       // -1 unknown, 0 no, 1 yes
    if (cached < 0) {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        cached = (e == cudaSuccess && count > 0) ? 1 : 0;
    }
    return cached == 1;
}

// ----------------------------------------------------------------------------
//  mul_ntt_cuda(): the public GPU multiply. Same pipeline as the CPU version but
//  the three transforms and the pointwise product run on the GPU.
// ----------------------------------------------------------------------------
std::vector<uint32_t> mul_ntt_cuda(const std::vector<uint32_t>& a,
                                   const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};
    if (!cuda_available()) return mul_ntt_cpu(a, b);  // graceful CPU fallback

    const uint32_t n = (uint32_t)ntt_length(a.size(), b.size());
    const int logn = ntt_log2(n);

    // Repack operands into 16-bit transform digits on the host.
    std::vector<uint64_t> ha = repack_to16(a, n);
    std::vector<uint64_t> hb = repack_to16(b, n);

    bool ok = true;
    uint64_t *d_a = nullptr, *d_b = nullptr, *d_Wf = nullptr, *d_Wi = nullptr;
    const size_t bytesN = (size_t)n * sizeof(uint64_t);
    const size_t bytesH = (size_t)(n / 2) * sizeof(uint64_t);

    // Allocate device buffers: the two operands and two twiddle tables (forward
    // and inverse). If any allocation fails we bail out to the CPU path.
    CUDA_TRY(cudaMalloc(&d_a, bytesN));
    CUDA_TRY(cudaMalloc(&d_b, bytesN));
    CUDA_TRY(cudaMalloc(&d_Wf, bytesH));
    CUDA_TRY(cudaMalloc(&d_Wi, bytesH));

    if (ok) {
        CUDA_TRY(cudaMemcpy(d_a, ha.data(), bytesN, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_b, hb.data(), bytesN, cudaMemcpyHostToDevice));

        // Build twiddle tables: forward uses omega_n, inverse uses omega_n^{-1}.
        const int TPB = 256;
        uint32_t half = n / 2;
        uint64_t omega = gl_root_of_unity((uint64_t)n);
        k_build_twiddles<<<(half + TPB - 1) / TPB, TPB>>>(d_Wf, half, omega);
        CUDA_TRY(cudaGetLastError());
        k_build_twiddles<<<(half + TPB - 1) / TPB, TPB>>>(d_Wi, half, gl_inv(omega));
        CUDA_TRY(cudaGetLastError());

        // Forward-transform both operands, multiply pointwise, inverse-transform.
        if (ok) ok = run_transform(d_a, n, logn, d_Wf, /*inverse=*/false);
        if (ok) ok = run_transform(d_b, n, logn, d_Wf, /*inverse=*/false);
        if (ok) {
            k_pointwise<<<(n + TPB - 1) / TPB, TPB>>>(d_a, d_b, n);
            CUDA_TRY(cudaGetLastError());
        }
        if (ok) ok = run_transform(d_a, n, logn, d_Wi, /*inverse=*/true);

        // Wait for the GPU and surface any asynchronous launch error.
        CUDA_TRY(cudaDeviceSynchronize());
    }

    std::vector<uint32_t> result;
    if (ok) {
        std::vector<uint64_t> hc(n);
        CUDA_TRY(cudaMemcpy(hc.data(), d_a, bytesN, cudaMemcpyDeviceToHost));
        if (ok) result = carry_propagate(hc);
    }

    // Always free what we allocated.
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_Wf); cudaFree(d_Wi);

    // If anything went wrong on the GPU, fall back to the CPU so the caller still
    // gets the correct product.
    if (!ok) return mul_ntt_cpu(a, b);
    return result;
}

} // namespace pidigits
