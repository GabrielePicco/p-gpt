//! Adam with bias correction, in fixed point, chunkable over parameter ranges.
//!
//! microgpt.py: lr 0.01, beta1 0.85, beta2 0.99, eps 1e-8, linear LR decay.
//! Perpetual twist: the decay stops at a floor instead of reaching zero.

use crate::backprop::train_fwd_bwd;
use crate::fixed::{fdiv, fmul, Fx, ONE};
use crate::math::sqrt_fx_coarse;
use crate::model::{Moments, Scratch, Weights, N_PARAMS};

pub const BETA1: Fx = 3_650_722_202; // 0.85
pub const BETA2: Fx = 4_252_017_623; // 0.99
pub const ONE_MINUS_BETA1: Fx = ONE - BETA1;
pub const ONE_MINUS_BETA2: Fx = ONE - BETA2;
/// Gradients arrive from the backward pass loss-scaled by 2^LOSS_SCALE_SHIFT.
/// The moments track them in that domain: without scaling, v = EMA(g^2)
/// underflows Q32.32 for |g| < 1.5e-5, leaving denom = eps and amplifying the
/// update ~1e3x. m_hat/sqrt(v_hat) is invariant under linear gradient
/// scaling, so only eps must be expressed in the scaled domain.
pub const EPS: Fx = 43 << crate::fixed::LOSS_SCALE_SHIFT; // 1e-8, scaled
pub const LR: Fx = 42_949_673; // 0.01
pub const LR_FLOOR: Fx = 4_294_967; // 0.001
pub const WARMDOWN_STEPS: u64 = 1000;

/// Bias-correction state: beta1^t and beta2^t, updated once per step by the
/// caller (stored in the model header on-chain). Starts at ONE for t = 0.
#[derive(Clone, Copy)]
pub struct AdamParams {
    pub lr_t: Fx,
    pub pow_beta1: Fx, // beta1^(t+1) at the time of the update
    pub pow_beta2: Fx,
}

/// microgpt.py's linear decay, floored so training can run forever.
pub fn lr_at_step(step: u64) -> Fx {
    let done = step.min(WARMDOWN_STEPS);
    let lr = LR * (WARMDOWN_STEPS - done) as i64 / WARMDOWN_STEPS as i64;
    lr.max(LR_FLOOR)
}

/// Apply Adam to params in `[start, end)`. Every parameter is updated each
/// step (moments decay even at zero gradient), exactly like the reference.
#[inline(never)]
pub fn adam_range(
    w: &mut [Fx; N_PARAMS],
    mom: &mut Moments,
    grads: &[Fx; N_PARAMS],
    start: usize,
    end: usize,
    p: AdamParams,
) {
    debug_assert!(start <= end && end <= N_PARAMS);
    // Bias corrections are constant over the loop: two divisions total,
    // multiplied per-param instead of divided (128-bit division costs ~100x
    // a multiply on SBF).
    let inv_bc1 = fdiv(ONE, ONE - p.pow_beta1);
    let inv_bc2 = fdiv(ONE, ONE - p.pow_beta2);
    for i in start..end {
        let g = grads[i];
        mom.m[i] = fmul(BETA1, mom.m[i]) + fmul(ONE_MINUS_BETA1, g);
        mom.v[i] = fmul(BETA2, mom.v[i]) + fmul(ONE_MINUS_BETA2, fmul(g, g));
        let m_hat = fmul(mom.m[i], inv_bc1);
        let v_hat = fmul(mom.v[i], inv_bc2);
        let denom = sqrt_fx_coarse(v_hat) + EPS;
        w[i] -= fmul(p.lr_t, fdiv_fast(m_hat, denom));
    }
}

/// a/b in Q32.32 with a native i64 fast path for the common small-|a| case.
#[inline(always)]
fn fdiv_fast(a: Fx, b: Fx) -> Fx {
    if a.unsigned_abs() < (1 << 30) {
        (a << 32) / b
    } else {
        fdiv(a, b)
    }
}

/// One full fused training step: forward + backward + Adam over all params.
///
/// `pow_beta1`/`pow_beta2` must hold beta^step on entry (ONE at step 0); they
/// are advanced to beta^(step+1) before the update, matching the reference's
/// `1 - beta**(step+1)` bias correction. Returns the loss.
pub fn train_doc(
    w: &mut Weights,
    mom: &mut Moments,
    s: &mut Scratch,
    tokens: &[u8],
    step: u64,
    pow_beta1: &mut Fx,
    pow_beta2: &mut Fx,
) -> Fx {
    let loss = train_fwd_bwd(w, s, tokens);
    *pow_beta1 = fmul(*pow_beta1, BETA1);
    *pow_beta2 = fmul(*pow_beta2, BETA2);
    let p = AdamParams { lr_t: lr_at_step(step), pow_beta1: *pow_beta1, pow_beta2: *pow_beta2 };
    adam_range(w.as_flat_mut(), mom, s.grads.as_flat(), 0, N_PARAMS, p);
    loss
}
