// ============================================================================
//  demo.cu  --  A narrated, run-everything guided tour (built as pi_demo)
// ============================================================================
//
//  Run this for a self-explaining walkthrough of the whole project. It prints
//  what it is about to do, does it, and shows the result and timing -- so you can
//  read the console top to bottom and understand both the math and the CUDA
//  acceleration. Nothing here is needed to USE the calculator (that's `pi`); this
//  is purely a teaching aid.
//
//  Sections:
//    A. Your GPU
//    B. Chudnovsky: compute and pretty-print pi, verified against reference
//    C. Why "fast multiply" matters: schoolbook vs Karatsuba vs CPU-NTT vs GPU-NTT
//    D. Chudnovsky scaling: time a few digit counts
//    E. BBP: extract hex digits anywhere, with no preceding digits (CPU vs GPU)
// ============================================================================

#include "bignum.h"
#include "chudnovsky.h"
#include "bbp.h"
#include "pi_reference.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>

using namespace pidigits;
using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static void rule(const char* title) {
    std::printf("\n========================================================\n");
    std::printf("  %s\n", title);
    std::printf("========================================================\n");
}

// deterministic random big number
static uint64_t s = 12345678901234567ULL;
static uint32_t rr() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)s; }
static BigInt randbig(size_t limbs) { BigInt r; r.mag.resize(limbs); for (auto& l : r.mag) l = rr(); r.sign = 1; r.normalize(); return r; }

// Pretty-print a long digit string in blocks of 10, 5 blocks per line, indented.
static void print_digits_blocked(const std::string& digits) {
    for (size_t i = 0; i < digits.size(); i += 10) {
        if (i % 50 == 0) std::printf("\n    ");
        std::printf("%s ", digits.substr(i, 10).c_str());
    }
    std::printf("\n");
}

int main() {
    std::printf("############################################################\n");
    std::printf("#   pi to many digits with CUDA: Chudnovsky + BBP demo     #\n");
    std::printf("############################################################\n");

    // --- A. GPU info --------------------------------------------------------
    rule("A.  Your GPU");
    if (cuda_available()) {
        cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
        std::printf("  Device      : %s\n", p.name);
        std::printf("  Compute cap : %d.%d  (sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
        std::printf("  SMs         : %d\n", p.multiProcessorCount);
        std::printf("  Global mem  : %.1f GB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("  -> The NTT transforms and BBP digit extraction run here.\n");
    } else {
        std::printf("  No CUDA device found -- everything will run on the CPU.\n");
    }

    // --- B. Chudnovsky pi, verified ----------------------------------------
    rule("B.  Chudnovsky series: compute pi, verify against reference");
    {
        const uint64_t D = 1000;
        std::printf("  Computing pi to %llu digits via Chudnovsky + binary splitting...\n",
                    (unsigned long long)D);
        g_mul.backend = MulBackend::Auto;
        auto r = compute_pi_chudnovsky(D, false);
        std::string frac = r.pi.substr(2);
        bool ok = frac.compare(0, D, std::string(PI_REF_DEC, D)) == 0;
        std::printf("  Used %llu series terms in %.1f ms.  Verify vs reference: %s\n",
                    (unsigned long long)r.terms, r.seconds * 1000.0, ok ? "PASS" : "FAIL");
        std::printf("  pi = 3.");
        print_digits_blocked(frac);
    }

    // --- C. The point of the project: fast multiplication -------------------
    rule("C.  Why fast multiply matters (multiply two huge numbers)");
    {
        size_t N = 60000; // limbs; each operand ~ 578,000 decimal digits
        std::printf("  Multiplying two %zu-limb integers (~%zu decimal digits each)\n",
                    N, (size_t)(N * 9.63));
        BigInt a = randbig(N), b = randbig(N);
        struct Run { const char* name; MulBackend be; };
        Run runs[] = {
            {"schoolbook  O(n^2)     ", MulBackend::Schoolbook},
            {"karatsuba   O(n^1.585) ", MulBackend::Karatsuba},
            {"ntt-cpu     O(n log n) ", MulBackend::NttCpu},
            {"ntt-cuda    O(n log n) ", MulBackend::NttCuda},
        };
        std::vector<uint32_t> reference;
        for (auto& run : runs) {
            g_mul.backend = run.be;
            auto t = Clock::now();
            auto prod = (a * b).mag;
            double e = ms_since(t);
            if (reference.empty()) reference = prod;
            std::printf("    %s %9.1f ms   %s\n", run.name, e,
                        prod == reference ? "(matches)" : "(MISMATCH!)");
        }
        std::printf("  Note how the GPU NTT turns a multi-second multiply into milliseconds.\n");
        g_mul.backend = MulBackend::Auto;
    }

    // --- D. Chudnovsky scaling ---------------------------------------------
    rule("D.  Chudnovsky scaling (digits vs time, GPU-accelerated multiply)");
    {
        g_mul.backend = MulBackend::Auto;
        for (uint64_t D : {1000ull, 10000ull, 100000ull}) {
            auto r = compute_pi_chudnovsky(D, false);
            std::string frac = r.pi.substr(2);
            size_t chk = frac.size() < (size_t)PI_REF_DEC_LEN ? frac.size() : (size_t)PI_REF_DEC_LEN;
            bool ok = frac.compare(0, chk, std::string(PI_REF_DEC, chk)) == 0;
            std::printf("    %8llu digits : %8.1f ms  (%llu terms)  first %zu digits %s\n",
                        (unsigned long long)D, r.seconds * 1000.0,
                        (unsigned long long)r.terms, chk, ok ? "verified" : "WRONG");
        }
    }

    // --- E. BBP: digits from nowhere ---------------------------------------
    rule("E.  BBP: hexadecimal digits, and the GPU's parallel throughput");
    {
        // E1: throughput. BBP parallelizes ACROSS positions -- one GPU thread per
        // hex digit -- so the GPU wins big when we want MANY digits at once. We
        // compute the first 4096 hex digits on both devices and compare.
        const uint64_t start = 1; const uint32_t count = 4096;
        std::printf("  Computing the first %u hex digits of pi (one thread per digit):\n", count);

        auto tg = Clock::now();
        std::string gpu = bbp_hex_digits_cuda(start, count);
        double gms = ms_since(tg);

        auto tc = Clock::now();
        std::string cpu = bbp_hex_digits_cpu(start, count);
        double cms = ms_since(tc);

        bool ok = (gpu == cpu) && (gpu.compare(0, 64, std::string(PI_REF_HEX, 64)) == 0);
        std::printf("    first 32: 3.%s...\n", gpu.substr(0, 32).c_str());
        std::printf("    GPU: %8.1f ms     CPU: %8.1f ms     speedup: %.0fx   (%s)\n",
                    gms, cms, cms / gms, ok ? "verified" : "MISMATCH");

        // E2: the "magic" -- a single digit deep inside pi, computed directly with
        // no preceding digits. (One position = one thread, so a CPU core is plenty
        // fast; the GPU's advantage is throughput, not single-digit latency.)
        std::printf("\n  The single hex digit at position 1,000,000 -- computed DIRECTLY,\n");
        std::printf("  without the preceding 999,999 digits:\n");
        auto t1 = Clock::now();
        char d = bbp_hex_digit(1000000);
        double dms = ms_since(t1);
        std::printf("    digit #1,000,000 = %c   (%.0f ms)   Bailey: 2 6 C 6 5 E ... -> %s\n",
                    d, dms, d == '2' ? "PASS" : "FAIL");
        std::printf("    That is the BBP formula's superpower: O(1) memory, any position.\n");
    }

    std::printf("\nDemo complete. Try the CLI:  pi --digits 100000 --verify\n");
    std::printf("                       and:  pi --algo bbp --bbp-start 1000000 --bbp-count 16\n");
    return 0;
}
