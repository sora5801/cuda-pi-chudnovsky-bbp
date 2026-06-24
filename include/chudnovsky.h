// ============================================================================
//  chudnovsky.h  --  Compute pi with the Chudnovsky series + binary splitting
// ============================================================================
//
//  THE FORMULA
//  -----------
//  The Chudnovsky brothers' 1988 series is the workhorse behind every modern
//  pi record (trillions of digits). It converges blisteringly fast -- each term
//  adds about 14.18 correct decimal digits:
//
//      1/pi = 12 * sum_{k=0}^{inf} (-1)^k (6k)! (13591409 + 545140134 k)
//                                   -----------------------------------------
//                                   (3k)! (k!)^3 640320^(3k + 3/2)
//
//  Evaluated and rearranged, it gives the identity this code computes:
//
//      pi = (426880 * sqrt(10005) * Q) / T
//
//  where Q and T come from "binary splitting" the first N terms.
//
//  WHY BINARY SPLITTING?
//  ---------------------
//  Adding the terms one by one would force us to work at full precision the whole
//  way -- millions of digits times millions of terms. Binary splitting instead
//  keeps EXACT integer fractions and combines them in a balanced binary tree.
//  Each term k contributes integer atoms:
//
//      p(k) = (6k-5)(2k-1)(6k-1)          (for k >= 1)
//      q(k) = k^3 * C3_OVER_24            where C3_OVER_24 = 640320^3 / 24
//      t(k) = p(k) * (13591409 + 545140134 k),   negated when k is odd
//
//  For a range [a, b) we define P, Q, T so that summing terms a..b-1 of the
//  series equals T/Q after factoring out P. The merge of a left range L and a
//  right range R is:
//
//      P = P_L * P_R
//      Q = Q_L * Q_R
//      T = T_L * Q_R + P_L * T_R
//
//  with the base case for a single index. Because the tree is balanced, the only
//  truly enormous multiplications happen near the ROOT, where both operands have
//  ~all the digits -- and those are exactly the multiplications our NTT/CUDA fast
//  multiply accelerates. This is why "fast multiply" is the whole game.
//
//  See docs/01_chudnovsky.md and docs/02_binary_splitting.md for the full
//  derivation, and src/chudnovsky.cpp for the implementation.
// ============================================================================

#pragma once

#include "bignum.h"
#include <string>
#include <cstdint>

namespace pidigits {

// The binary-splitting triple for an index range [a, b). See the merge rules
// above. P and Q are positive; T may be negative (the series alternates).
struct PQT {
    BigInt P, Q, T;
};

// Compute (P, Q, T) for the half-open range [a, b) of Chudnovsky terms. Called
// as chudnovsky_bs(0, N) to sum the first N terms. Recursive, splitting at the
// midpoint -- the standard balanced binary splitting.
PQT chudnovsky_bs(uint64_t a, uint64_t b);

// Result of a full pi computation.
struct ChudnovskyResult {
    std::string pi;     // the digits as "3.1415926535..." with `digits` after the point
    uint64_t    terms;  // number of series terms summed (N)
    double      seconds;// wall-clock time for the whole computation
};

// Compute pi to `digits` digits AFTER the decimal point. If verbose, prints
// progress/timing of the phases (binary splitting, sqrt, division, formatting).
ChudnovskyResult compute_pi_chudnovsky(uint64_t digits, bool verbose = false);

// Number of Chudnovsky terms needed for `digits` correct digits (each term is
// worth ~14.1816 digits). Exposed so callers/tests can report it.
uint64_t chudnovsky_terms_for_digits(uint64_t digits);

} // namespace pidigits
