// ============================================================================
//  self_test.cu  --  The project's correctness test suite (built as pi_selftest)
// ============================================================================
//
//  This single program exercises every layer of the project and PROVES the fast,
//  GPU-accelerated paths agree with the slow, obviously-correct ones, and that the
//  final pi digits match an independent reference. It exits non-zero if anything
//  is wrong, so it doubles as a CI check (CMake registers it via add_test).
//
//  Layers tested, bottom to top:
//    1. Goldilocks field algebra + roots of unity (the NTT's foundation)
//    2. NTT multiply: CPU and CUDA both equal schoolbook
//    3. Division (Knuth) invariant, isqrt vs bit-by-bit reference, decimal I/O
//    4. Chudnovsky pi vs the trusted reference digits (incl. a GPU-NTT run)
//    5. BBP hex digits vs reference (CPU + CUDA), and a high-position checkpoint
//
//  It is intentionally kept to a few seconds: the only slow primitive, the
//  single-threaded BBP reference, is exercised only on the cheap low positions.
// ============================================================================

#include "bignum.h"
#include "goldilocks.h"
#include "chudnovsky.h"
#include "bbp.h"
#include "pi_reference.h"

#include <cstdio>
#include <string>

using namespace pidigits;

// --- tiny test harness ------------------------------------------------------
static int g_fail = 0;
static int g_pass = 0;
#define CHECK(cond, msg) do { if (cond) { ++g_pass; } else { ++g_fail; std::printf("  FAIL: %s\n", msg); } } while (0)

// deterministic xorshift RNG so runs are reproducible
static uint64_t s_state = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd64() { s_state ^= s_state << 13; s_state ^= s_state >> 7; s_state ^= s_state << 17; return s_state; }
static uint32_t rnd32() { return (uint32_t)rnd64(); }
static BigInt randbig(size_t limbs) {
    BigInt r; r.mag.resize(limbs); for (auto& l : r.mag) l = rnd32(); r.sign = 1; r.normalize(); return r;
}

// ----------------------------------------------------------------------------
static void test_field() {
    std::printf("[1] Goldilocks field algebra + roots of unity\n");
    for (int t = 0; t < 20000; ++t) {
        uint64_t a = rnd64() % GL_P, b = rnd64() % GL_P, c = rnd64() % GL_P;
        CHECK(gl_add(a, b) == gl_add(b, a), "add commutative");
        CHECK(gl_sub(gl_add(a, b), b) == a, "(a+b)-b == a");
        CHECK(gl_mul(a, gl_add(b, c)) == gl_add(gl_mul(a, b), gl_mul(a, c)), "distributive");
        if (a) CHECK(gl_mul(a, gl_inv(a)) == 1, "a * inv(a) == 1");
    }
    CHECK(gl_pow(GL_G, GL_P - 1) == 1, "Fermat: g^(p-1)==1");
    for (size_t lg = 1; lg <= 22; ++lg) {
        uint64_t N = (uint64_t)1 << lg;
        uint64_t w = gl_root_of_unity(N);
        CHECK(gl_pow(w, N) == 1, "omega^N == 1");
        CHECK(gl_pow(w, N / 2) == GL_P - 1, "omega^(N/2) == -1");
    }
}

static void test_multiply() {
    std::printf("[2] NTT multiply (CPU + CUDA) == schoolbook\n");
    std::printf("    cuda_available = %s\n", cuda_available() ? "yes" : "no");
    for (int t = 0; t < 120; ++t) {
        size_t na = 1 + rnd32() % 1500, nb = 1 + rnd32() % 1500;
        BigInt a = randbig(na), b = randbig(nb);
        auto sb  = mul_schoolbook(a.mag, b.mag);
        auto kar = mul_karatsuba(a.mag, b.mag);
        auto cpu = mul_ntt_cpu(a.mag, b.mag);
        auto gpu = mul_ntt_cuda(a.mag, b.mag);
        CHECK(sb == kar, "karatsuba == schoolbook");
        CHECK(sb == cpu, "ntt-cpu == schoolbook");
        CHECK(sb == gpu, "ntt-cuda == schoolbook");
    }
}

static void test_div_sqrt_dec() {
    std::printf("[3] division / isqrt / decimal conversion\n");
    g_mul.backend = MulBackend::Auto;
    for (int t = 0; t < 800; ++t) {
        BigInt a = randbig(1 + rnd32() % 60), b = randbig(1 + rnd32() % 30);
        if (b.is_zero()) b = BigInt((int64_t)1);
        BigInt q, r; divmod_knuth(a, b, q, r);
        CHECK((q * b + r).cmp(a) == 0, "q*b + r == a");
        CHECK(r.sign >= 0 && r.cmp(b) < 0, "0 <= r < b");
    }
    for (int t = 0; t < 400; ++t) {
        BigInt s = randbig(1 + rnd32() % 30);
        BigInt x = isqrt(s);
        CHECK(x.cmp(isqrt_bitwise(s)) == 0, "isqrt == bitwise reference");
        CHECK((x * x).cmp(s) <= 0 && s.cmp((x + BigInt((int64_t)1)) * (x + BigInt((int64_t)1))) < 0, "isqrt range");
    }
    for (int t = 0; t < 400; ++t) {
        BigInt x = randbig(1 + rnd32() % 40);
        if (rnd32() & 1) x.sign = -1;
        CHECK(BigInt::from_decimal(x.to_decimal()).cmp(x) == 0, "decimal round-trip");
    }
}

static void test_chudnovsky() {
    std::printf("[4] Chudnovsky pi vs reference digits\n");
    auto verify = [](uint64_t digits, MulBackend be) {
        g_mul.backend = be;
        ChudnovskyResult r = compute_pi_chudnovsky(digits, false);
        std::string frac = r.pi.substr(2);
        size_t n = frac.size() < (size_t)PI_REF_DEC_LEN ? frac.size() : (size_t)PI_REF_DEC_LEN;
        bool ok = frac.compare(0, n, std::string(PI_REF_DEC, n)) == 0 && frac.size() == digits;
        CHECK(ok, "chudnovsky digits match reference");
        std::printf("    %6llu digits  terms=%-5llu  %s\n",
                    (unsigned long long)digits, (unsigned long long)r.terms, ok ? "ok" : "MISMATCH");
    };
    verify(100, MulBackend::Auto);
    verify(1000, MulBackend::Auto);
    verify(3500, MulBackend::Auto);
    // A larger run that pushes the big multiplications onto the GPU NTT, verified
    // against the 3500 reference digits we have.
    verify(8000, MulBackend::NttCuda);
    g_mul.backend = MulBackend::Auto;
}

static void test_bbp() {
    std::printf("[5] BBP hex digits vs reference (CPU + CUDA)\n");
    std::string cpu = bbp_hex_digits_cpu(1, 64);
    std::string gpu = bbp_hex_digits_cuda(1, 64);
    std::string ref(PI_REF_HEX, 64);
    CHECK(cpu == ref, "BBP cpu 1..64 == reference");
    CHECK(gpu == ref, "BBP gpu 1..64 == reference");
    // CPU vs GPU agree on a block (GPU does the heavy lifting; CPU block kept small)
    CHECK(bbp_hex_digits_cpu(500, 64) == bbp_hex_digits_cuda(500, 64), "BBP cpu==gpu @500");
    // High-position checkpoint (Bailey Table 1: position 10^6 -> 26C65E52CB4593).
    // GPU is fast even here; we trust the first ~10 digits with double precision.
    CHECK(bbp_hex_digits_cuda(1000000, 10) == std::string("26C65E52CB"), "BBP checkpoint @1e6");
}

int main() {
    std::printf("================ pi_selftest ================\n");
    test_field();
    test_multiply();
    test_div_sqrt_dec();
    test_chudnovsky();
    test_bbp();
    std::printf("=============================================\n");
    std::printf("%d checks passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) std::printf("ALL TESTS PASSED\n");
    return g_fail ? 1 : 0;
}
