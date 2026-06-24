// ============================================================================
//  main.cpp  --  Command-line front-end for the pi calculator
// ============================================================================
//
//  This is the program you run. It ties the two algorithms together behind a
//  small, well-documented command-line interface:
//
//      * Chudnovsky series (chudnovsky.h) -- compute the first N DECIMAL digits.
//      * BBP formula (bbp.h)              -- compute a block of HEX digits at an
//                                            arbitrary position, on CPU or GPU.
//
//  It also lets you pick the big-integer multiply backend (schoolbook, Karatsuba,
//  CPU-NTT, GPU-NTT) so you can SEE the effect of fast multiplication on timing,
//  and it can verify freshly computed digits against the trusted reference table.
//
//  Run `pi --help` for usage. The argument parser below is deliberately tiny and
//  dependency-free so the control flow is easy to read.
// ============================================================================

#include "bignum.h"
#include "chudnovsky.h"
#include "bbp.h"
#include "pi_reference.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <fstream>

using namespace pidigits;

// ----------------------------------------------------------------------------
//  Help text.
// ----------------------------------------------------------------------------
static void print_usage(const char* prog) {
    std::printf(
"pi -- compute the digits of pi (Chudnovsky decimal, or BBP hexadecimal)\n"
"\n"
"USAGE:\n"
"  %s [options]\n"
"\n"
"COMMON OPTIONS:\n"
"  --algo <chudnovsky|bbp>   Which algorithm to run (default: chudnovsky)\n"
"  --backend <name>          Big-integer multiply backend used by Chudnovsky:\n"
"                            auto | schoolbook | karatsuba | ntt-cpu | ntt-cuda\n"
"                            (default: auto -- picks the fastest by operand size)\n"
"  --out <file>              Write the digits to <file> instead of stdout\n"
"  --verify                  Check the result against the built-in reference digits\n"
"  --quiet                   Print only the digits (no banners or timing)\n"
"  --help                    Show this help\n"
"\n"
"CHUDNOVSKY (decimal digits):\n"
"  --digits <N>              Number of digits AFTER the decimal point (default 1000)\n"
"\n"
"BBP (hexadecimal digit extraction):\n"
"  --bbp-start <P>           1-based hex position to start at (default 1)\n"
"  --bbp-count <C>           How many hex digits to extract (default 32)\n"
"  --bbp-device <cpu|gpu>    Where to run BBP (default: gpu if available)\n"
"\n"
"EXAMPLES:\n"
"  %s --digits 10000 --verify\n"
"  %s --backend ntt-cuda --digits 1000000 --out pi.txt\n"
"  %s --algo bbp --bbp-start 1000000 --bbp-count 16 --bbp-device gpu\n",
        prog, prog, prog, prog);
}

// Parse a backend name into the MulBackend enum and configure g_mul. Returns
// false on an unknown name.
static bool set_backend(const std::string& name) {
    if (name == "auto")            { g_mul.backend = MulBackend::Auto; g_mul.prefer_cuda = true; }
    else if (name == "schoolbook") { g_mul.backend = MulBackend::Schoolbook; }
    else if (name == "karatsuba")  { g_mul.backend = MulBackend::Karatsuba; }
    else if (name == "ntt-cpu")    { g_mul.backend = MulBackend::NttCpu; }
    else if (name == "ntt-cuda")   { g_mul.backend = MulBackend::NttCuda; }
    else return false;
    return true;
}

// Write a string either to a file (if `path` non-empty) or to stdout.
static void emit(const std::string& path, const std::string& data, bool quiet) {
    if (!path.empty()) {
        std::ofstream f(path, std::ios::binary);
        f.write(data.data(), (std::streamsize)data.size());
        f.put('\n');
        if (!quiet) std::printf("wrote %zu characters to %s\n", data.size(), path.c_str());
    } else {
        std::fwrite(data.data(), 1, data.size(), stdout);
        std::putchar('\n');
    }
}

int main(int argc, char** argv) {
    // --- Defaults -----------------------------------------------------------
    std::string algo = "chudnovsky";
    uint64_t digits = 1000;
    uint64_t bbp_start = 1;
    uint32_t bbp_count = 32;
    std::string bbp_device = "gpu";
    std::string out_path;
    bool verify = false;
    bool quiet = false;

    // --- Parse arguments ----------------------------------------------------
    // Tiny hand-rolled parser: each known flag consumes its value (if any).
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
        else if (a == "--algo")        algo = need("--algo");
        else if (a == "--digits")      digits = std::strtoull(need("--digits").c_str(), nullptr, 10);
        else if (a == "--backend")     { if (!set_backend(need("--backend"))) { std::fprintf(stderr, "error: unknown backend\n"); return 2; } }
        else if (a == "--bbp-start")   bbp_start = std::strtoull(need("--bbp-start").c_str(), nullptr, 10);
        else if (a == "--bbp-count")   bbp_count = (uint32_t)std::strtoul(need("--bbp-count").c_str(), nullptr, 10);
        else if (a == "--bbp-device")  bbp_device = need("--bbp-device");
        else if (a == "--out")         out_path = need("--out");
        else if (a == "--verify")      verify = true;
        else if (a == "--quiet")       quiet = true;
        else { std::fprintf(stderr, "error: unknown option '%s' (try --help)\n", a.c_str()); return 2; }
    }

    // --- Run the chosen algorithm ------------------------------------------
    if (algo == "chudnovsky") {
        if (!quiet) {
            std::printf("Chudnovsky series + binary splitting\n");
            std::printf("  digits : %llu\n", (unsigned long long)digits);
            std::printf("  backend: %s\n",
                g_mul.backend == MulBackend::Auto ? (cuda_available() ? "auto (GPU available)" : "auto (CPU only)") :
                g_mul.backend == MulBackend::Schoolbook ? "schoolbook" :
                g_mul.backend == MulBackend::Karatsuba ? "karatsuba" :
                g_mul.backend == MulBackend::NttCpu ? "ntt-cpu" : "ntt-cuda");
        }
        ChudnovskyResult r = compute_pi_chudnovsky(digits, /*verbose=*/!quiet);
        if (!quiet)
            std::printf("  terms  : %llu\n  time   : %.3f s\n",
                        (unsigned long long)r.terms, r.seconds);

        if (verify) {
            // Compare as many fractional digits as we have a reference for.
            std::string frac = r.pi.substr(2);                  // drop "3."
            size_t n = frac.size() < (size_t)PI_REF_DEC_LEN ? frac.size() : (size_t)PI_REF_DEC_LEN;
            std::string ref(PI_REF_DEC, n);
            bool good = frac.compare(0, n, ref) == 0;
            std::printf("  verify : %s (checked %zu digits against reference)\n",
                        good ? "PASS" : "FAIL", n);
            if (!good) {
                size_t k = 0; while (k < n && frac[k] == ref[k]) ++k;
                std::printf("           first mismatch at fractional digit %zu\n", k + 1);
            }
        }
        emit(out_path, r.pi, quiet);
    }
    else if (algo == "bbp") {
        bool use_gpu = (bbp_device == "gpu");
        if (!quiet) {
            std::printf("BBP hexadecimal digit extraction\n");
            std::printf("  start  : %llu (1-based)\n  count  : %u\n  device : %s\n",
                        (unsigned long long)bbp_start, bbp_count,
                        use_gpu ? (cuda_available() ? "gpu" : "gpu requested, falling back to cpu") : "cpu");
        }
        std::string hex = use_gpu ? bbp_hex_digits_cuda(bbp_start, bbp_count)
                                  : bbp_hex_digits_cpu(bbp_start, bbp_count);

        if (verify && bbp_start >= 1 && bbp_start + bbp_count - 1 <= 64) {
            // Positions 1..64 are covered by the reference hex prefix.
            std::string ref(PI_REF_HEX + (bbp_start - 1), bbp_count);
            std::printf("  verify : %s\n", hex == ref ? "PASS" : "FAIL");
        }
        emit(out_path, hex, quiet);
    }
    else {
        std::fprintf(stderr, "error: unknown --algo '%s' (use chudnovsky or bbp)\n", algo.c_str());
        return 2;
    }
    return 0;
}
