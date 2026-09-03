//! Q32.32 signed fixed-point scalar.
//!
//! One uniform format everywhere — weights, activations, gradients, optimizer
//! moments — so there is exactly one rounding rule to reason about. Products
//! go through i128 and are truncated (toward negative infinity via arithmetic
//! shift), which is the deterministic contract the replay proof relies on.

/// Q32.32 fixed-point value stored in an `i64`.
pub type Fx = i64;

/// Number of fractional bits.
pub const FRAC: u32 = 32;

/// 1.0 in Q32.32.
pub const ONE: Fx = 1 << FRAC;

/// Fixed-point multiply: (a * b) >> 32 with an exact i128 intermediate.
#[inline(always)]
pub const fn fmul(a: Fx, b: Fx) -> Fx {
    (((a as i128) * (b as i128)) >> FRAC) as i64
}

/// Fixed-point divide: (a << 32) / b with an exact i128 intermediate.
///
/// Truncates toward zero (i128 division semantics). `b` must be non-zero.
#[inline(always)]
pub const fn fdiv(a: Fx, b: Fx) -> Fx {
    (((a as i128) << FRAC) / (b as i128)) as i64
}

/// Integer to fixed-point.
#[inline(always)]
pub const fn from_int(i: i64) -> Fx {
    i << FRAC
}

/// Fast Q32.32 multiply: truncate both operands to Q16.16, multiply natively.
///
/// SBF has no 128-bit ALU — a full `fmul` costs ~35 CU through `__multi3`
/// while this is a handful of native instructions. Precision is fp16-like
/// (operands quantized at 2^-16), which is why the backward pass loss-scales
/// gradients by 2^LOSS_SCALE_SHIFT before flowing them through these kernels.
/// Operand magnitude must stay below 2^31 in Q16.16 (values < ~32K).
#[inline(always)]
pub const fn fmul16(a: Fx, b: Fx) -> Fx {
    (a >> 16) * (b >> 16)
}

/// Gradient loss-scaling: backward emits gradients scaled by 2^12 so they
/// survive Q16.16 truncation in the fast kernels; Adam consumes them in the
/// scaled domain (its eps is expressed scaled too).
pub const LOSS_SCALE_SHIFT: u32 = 12;

#[cfg(test)]
mod tests {
    use super::*;

    pub fn to_f64(x: Fx) -> f64 {
        x as f64 / ONE as f64
    }

    #[test]
    fn mul_div_roundtrip() {
        let a = ONE * 3 / 2; // 1.5
        let b = ONE * 5 / 4; // 1.25
        assert!((to_f64(fmul(a, b)) - 1.875).abs() < 1e-9);
        assert!((to_f64(fdiv(a, b)) - 1.2).abs() < 1e-9);
        assert_eq!(fmul(-a, b), -fmul(a, b));
    }
}
