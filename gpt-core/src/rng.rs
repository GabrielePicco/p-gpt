//! Deterministic PRNG: xorshift64* with a CLT Gaussian.
//!
//! Weight init and sampling both flow through this so a (seed, history) pair
//! fully determines the model — the replay proof depends on it.

use crate::fixed::{fmul, Fx, ONE};

/// xorshift64* state. Never zero.
#[derive(Clone, Copy)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1) as Q32.32.
    #[inline]
    pub fn next_unit(&mut self) -> Fx {
        (self.next_u64() >> 32) as i64
    }

    /// Standard Gaussian (mean 0, std 1) by the central limit theorem:
    /// sum of 12 uniforms minus 6. Plenty for weight init at std 0.08.
    pub fn next_gauss(&mut self) -> Fx {
        let mut acc: Fx = -6 * ONE;
        for _ in 0..12 {
            acc += self.next_unit();
        }
        acc
    }

    /// Gaussian scaled by `std` (Q32.32).
    pub fn next_gauss_scaled(&mut self, std: Fx) -> Fx {
        fmul(self.next_gauss(), std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_moments() {
        let mut rng = Rng::new(42);
        let n = 100_000;
        let (mut sum, mut sum2) = (0f64, 0f64);
        for _ in 0..n {
            let g = rng.next_gauss() as f64 / ONE as f64;
            sum += g;
            sum2 += g * g;
        }
        let mean = sum / n as f64;
        let var = sum2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }
}
