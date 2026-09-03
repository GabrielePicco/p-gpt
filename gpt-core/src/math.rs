//! Integer transcendentals for Q32.32: exp, ln, sqrt.
//!
//! Accuracy targets are set by their callers: `exp_fx` feeds softmax (relative
//! error ~1e-9 from a degree-7 Taylor tail), `sqrt_fx` feeds RMSNorm and Adam
//! (exact integer sqrt), `ln_fx` only reports the loss.

use crate::fixed::{fdiv, fmul, Fx, FRAC, ONE};

/// ln(2) in Q32.32.
pub const LN2: Fx = 2_977_044_472;

/// log2(e) in Q32.32.
pub const LOG2E: Fx = 6_196_328_019;

/// 2^(k/64) for k in 0..64, Q32.32 — the exp lookup table.
const EXP2_LUT: [i64; 64] = [
    4294967296, 4341736423, 4389014833, 4436808071, 4485121744, 4533961517, 4583333121, 4633242347,
    4683695048, 4734697143, 4786254615, 4838373510, 4891059943, 4944320094, 4998160210, 5052586606,
    5107605667, 5163223846, 5219447668, 5276283726, 5333738689, 5391819295, 5450532358, 5509884764,
    5569883475, 5630535530, 5691848042, 5753828203, 5816483285, 5879820635, 5943847684, 6008571941,
    6074001000, 6140142534, 6207004303, 6274594148, 6342919999, 6411989869, 6481811861, 6552394164,
    6623745059, 6695872913, 6768786189, 6842493438, 6917003306, 6992324534, 7068465956, 7145436504,
    7223245206, 7301901189, 7381413680, 7461792005, 7543045592, 7625183973, 7708216783, 7792153760,
    7877004752, 7962779710, 8049488696, 8137141881, 8225749546, 8315322086, 8405870007, 8497403930,
];

/// exp(x) for x <= 0 (the post max-subtraction softmax domain).
///
/// Softmax calls this ~90 times per forward position, so it is built for SBF:
/// exp(x) = 2^(x*log2e) with the fractional power split into a 6-bit table
/// lookup and a short quadratic correction — ~5 multiplies, no division.
/// Inputs below -22 underflow Q32.32 and return 0.
pub fn exp_fx(x: Fx) -> Fx {
    debug_assert!(x <= 0);
    if x < -22 * ONE {
        return 0;
    }
    let t = fmul(x, LOG2E); // <= 0
    let i = t >> FRAC; // floor
    let f = t - (i << FRAC); // [0, 1)
    let hi = (f >> 26) as usize; // top 6 bits
    let lo = f & ((1 << 26) - 1); // [0, 2^-6)

    // 2^lo ~ 1 + lo*ln2 + (lo*ln2)^2/2 (max error ~2e-7 over the window).
    let r = fmul(lo, LN2);
    let poly = ONE + r + (fmul(r, r) >> 1);
    fmul(EXP2_LUT[hi], poly) >> (-i) as u32
}

/// Integer sqrt of the u64 `z` by Newton's method. Native 64-bit divisions
/// only — SBF has no 128-bit ALU, so u128 arithmetic costs ~100x.
#[inline]
fn isqrt_u64(z: u64) -> u64 {
    if z == 0 {
        return 0;
    }
    let bits = 64 - z.leading_zeros();
    let mut s = 1u64 << bits.div_ceil(2);
    loop {
        let next = (s + z / s) >> 1;
        if next >= s {
            break;
        }
        s = next;
    }
    s
}

/// sqrt of a non-negative Q32.32 value, result in Q32.32.
///
/// isqrt over the raw u64 gives sqrt to 16 fractional bits; one Newton
/// refinement in u128 restores full Q32.32 precision.
pub fn sqrt_fx(x: Fx) -> Fx {
    debug_assert!(x >= 0);
    if x == 0 {
        return 0;
    }
    let n = (x as u128) << FRAC;
    // Seed strictly above the root ((floor+1) << 16 > true sqrt), so the
    // Newton iteration is monotone decreasing and converges.
    let mut s = ((isqrt_u64(x as u64) as u128) + 1) << 16;
    loop {
        let next = (s + n / s) >> 1;
        if next >= s {
            break;
        }
        s = next;
    }
    // Newton lands within one of the floor.
    while s * s > n {
        s -= 1;
    }
    s as i64
}

/// Coarse sqrt of a non-negative Q32.32 value: ~16 fractional bits of
/// precision, native u64 arithmetic only. Used in the Adam inner loop where
/// the denominator tolerance is far above 2^-16.
pub fn sqrt_fx_coarse(x: Fx) -> Fx {
    debug_assert!(x >= 0);
    if x <= 0 {
        return 0;
    }
    (isqrt_u64(x as u64) as i64) << 16
}

/// 1/sqrt(x) for x > 0 in Q32.32, by multiply-only Newton iteration.
///
/// 128-bit division costs thousands of native instructions on SBF, so the
/// hot paths (RMSNorm runs three times per forward position) use this
/// division-free form: y' = y * (3 - x*y*y) / 2, seeded from the exponent.
pub fn rsqrt_fx(x: Fx) -> Fx {
    debug_assert!(x > 0);
    // Seed: for x ~ 2^e (e relative to Q32.32 one), rsqrt ~ 2^(-e/2).
    let msb = 63 - x.leading_zeros() as i32; // bit index of the leading 1
    let e = msb - FRAC as i32;
    // y0 = 2^floor(-e/2), times sqrt(2) for odd exponents so the seed is
    // exactly 2^(-e/2) at powers of two.
    let half = if e >= 0 { -(e / 2) - (e & 1) } else { (-e) / 2 };
    let mut y: Fx = if half >= 0 { ONE << half } else { ONE >> (-half) };
    if e & 1 != 0 {
        const SQRT2: Fx = 6_074_001_000; // sqrt(2) in Q32.32
        y = fmul(y, SQRT2);
    }
    // Quadratic convergence: 4 iterations cover the seed's < 2x error.
    for _ in 0..4 {
        let correction = (3 * ONE) - fmul(x, fmul(y, y));
        y = fmul(y, correction) >> 1;
    }
    y
}

/// ln(x) for x > 0, via ln(x) = k*ln2 + 2*atanh((m-1)/(m+1)) with m in [1, 2).
pub fn ln_fx(x: Fx) -> Fx {
    debug_assert!(x > 0);
    let msb = 63 - x.leading_zeros() as i64; // bit index of the leading 1
    let exp = msb - FRAC as i64;
    let m = if exp >= 0 { x >> exp as u32 } else { x << (-exp) as u32 }; // [ONE, 2*ONE)

    let t = fdiv(m - ONE, m + ONE); // [0, 1/3)
    let t2 = fmul(t, t);
    let mut acc = 2 * ONE / 13;
    acc = 2 * ONE / 11 + fmul(t2, acc);
    acc = 2 * ONE / 9 + fmul(t2, acc);
    acc = 2 * ONE / 7 + fmul(t2, acc);
    acc = 2 * ONE / 5 + fmul(t2, acc);
    acc = 2 * ONE / 3 + fmul(t2, acc);
    acc = 2 * ONE + fmul(t2, acc);
    fmul(t, acc) + exp * LN2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f64(x: Fx) -> f64 {
        x as f64 / ONE as f64
    }
    fn from_f64(x: f64) -> Fx {
        (x * ONE as f64) as i64
    }

    #[test]
    fn exp_matches_std() {
        for i in 0..2200 {
            let x = -i as f64 * 0.01;
            let got = to_f64(exp_fx(from_f64(x)));
            let want = x.exp();
            assert!((got - want).abs() < 5e-7, "exp({x}): {got} vs {want}");
        }
    }

    #[test]
    fn sqrt_matches_std() {
        for i in 1..5000u32 {
            let x = i as f64 * 0.037;
            let got = to_f64(sqrt_fx(from_f64(x)));
            let want = x.sqrt();
            assert!((got - want).abs() < 1e-6, "sqrt({x}): {got} vs {want}");
        }
    }

    #[test]
    fn ln_matches_std() {
        for i in 1..5000u32 {
            let x = i as f64 * 0.0093;
            let got = to_f64(ln_fx(from_f64(x)));
            let want = x.ln();
            assert!((got - want).abs() < 1e-6, "ln({x}): {got} vs {want}");
        }
    }
}
