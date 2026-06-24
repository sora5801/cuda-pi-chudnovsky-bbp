// ============================================================================
//  bbp_cuda.cu  --  GPU BBP digit extraction: one thread per hex position
// ============================================================================
//
//  BBP is "embarrassingly parallel": the hex digit at each position is computed
//  with no reference to any other position. So the GPU mapping is the simplest
//  possible -- launch `count` threads, thread i computes the digit at position
//  start+i, writes it to out[i]. No shared memory, no synchronization, no inter-
//  thread communication. This is the cleanest illustration in the whole project
//  of CUDA's data-parallel model.
//
//  The per-digit math is the shared __host__ __device__ routine bbp_digit_value()
//  from bbp_kernel.h, identical to what the CPU runs.
// ============================================================================

#include "bbp.h"
#include "bbp_kernel.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdio>

namespace pidigits {

// Declared in ntt_cuda.cu (or the stub): is a CUDA device present?
bool cuda_available();

// One thread per position. Writes the digit value (0..15) into out[i].
__global__ void k_bbp(unsigned char* out, uint64_t start, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = (unsigned char)bbp_digit_value(start + (uint64_t)i);
}

static inline char to_hex_char(int v) {
    return (char)(v < 10 ? ('0' + v) : ('A' + (v - 10)));
}

std::string bbp_hex_digits_cuda(uint64_t start, uint32_t count) {
    if (count == 0) return std::string();
    if (!cuda_available()) return bbp_hex_digits_cpu(start, count); // graceful fallback

    bool ok = true;
    unsigned char* d_out = nullptr;
    const size_t bytes = (size_t)count * sizeof(unsigned char);

    cudaError_t e = cudaMalloc(&d_out, bytes);
    if (e != cudaSuccess) { std::fprintf(stderr, "[cuda] bbp malloc: %s\n", cudaGetErrorString(e)); ok = false; }

    std::vector<unsigned char> host(count);
    if (ok) {
        const int TPB = 256;                              // threads per block
        k_bbp<<<(count + TPB - 1) / TPB, TPB>>>(d_out, start, count);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "[cuda] bbp launch: %s\n", cudaGetErrorString(e)); ok = false; }
        if (ok) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::fprintf(stderr, "[cuda] bbp sync: %s\n", cudaGetErrorString(e)); ok = false; }
        if (ok) e = cudaMemcpy(host.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { std::fprintf(stderr, "[cuda] bbp copy: %s\n", cudaGetErrorString(e)); ok = false; }
    }
    cudaFree(d_out);

    if (!ok) return bbp_hex_digits_cpu(start, count);     // correctness never depends on the GPU

    std::string out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) out.push_back(to_hex_char(host[i]));
    return out;
}

} // namespace pidigits
