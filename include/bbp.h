// ============================================================================
//  bbp.h  --  Public interface for BBP hexadecimal digit extraction
// ============================================================================
//
//  The math lives in bbp_kernel.h (shared host/device). These are the host-facing
//  entry points that compute a contiguous BLOCK of hex digits:
//
//      * bbp_hex_digits_cpu : loops over positions on the CPU (single-threaded
//        reference; also used when no GPU is present).
//      * bbp_hex_digits_cuda: launches one GPU thread per position -- the
//        embarrassingly-parallel showcase. Falls back to the CPU automatically if
//        CUDA is unavailable or errors.
//      * bbp_hex_digit      : the single-digit convenience wrapper.
//
//  "position" is 1-based: position 1 is the first hex digit after the radix point
//  (which is 2, since pi = 3.243F6A88...).
// ============================================================================

#pragma once

#include <string>
#include <cstdint>

namespace pidigits {

// The hex digit (as a character '0'..'9','A'..'F') at 1-based fractional position.
char bbp_hex_digit(uint64_t position);

// `count` hex digits starting at 1-based `start`, computed on the CPU.
std::string bbp_hex_digits_cpu(uint64_t start, uint32_t count);

// Same, computed on the GPU (one thread per position). Falls back to CPU if no
// usable CUDA device is available.
std::string bbp_hex_digits_cuda(uint64_t start, uint32_t count);

} // namespace pidigits
