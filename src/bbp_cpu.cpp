// ============================================================================
//  bbp_cpu.cpp  --  CPU reference implementation of BBP digit extraction
// ============================================================================
//
//  This is the single-threaded reference: it walks the requested positions and
//  calls the shared per-digit routine from bbp_kernel.h for each one. The GPU
//  version in bbp_cuda.cu computes exactly the same values, just thousands at a
//  time. The test suite checks the two against each other and against the known
//  hex digits of pi.
// ============================================================================

#include "bbp.h"
#include "bbp_kernel.h"

#include <string>

namespace pidigits {

// Map a nibble value 0..15 to its uppercase hex character.
static inline char to_hex_char(int v) {
    return (char)(v < 10 ? ('0' + v) : ('A' + (v - 10)));
}

char bbp_hex_digit(uint64_t position) {
    return to_hex_char(bbp_digit_value(position));
}

std::string bbp_hex_digits_cpu(uint64_t start, uint32_t count) {
    std::string out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        out.push_back(to_hex_char(bbp_digit_value(start + i)));
    return out;
}

} // namespace pidigits
