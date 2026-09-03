//! Model shape, zero-copy state layouts, forward pass, and sampling.
//!
//! `Weights`, `Moments`, and `Scratch` are `#[repr(C)]` so on-chain accounts
//! can be reinterpreted as them directly — the account data is the tensor.

use crate::fixed::{fdiv, fmul, fmul16, Fx, ONE};
use crate::math::{exp_fx, rsqrt_fx};
use crate::rng::Rng;

pub const VOCAB: usize = 27; // a-z + BOS
pub const BOS: u8 = 26;
pub const N_EMBD: usize = 16;
pub const BLOCK: usize = 16;
pub const N_HEAD: usize = 4;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const MLP_DIM: usize = 4 * N_EMBD;
pub const N_PARAMS: usize =
    VOCAB * N_EMBD + BLOCK * N_EMBD + 4 * N_EMBD * N_EMBD + 2 * MLP_DIM * N_EMBD + VOCAB * N_EMBD;

/// RMSNorm epsilon: 1e-5 in Q32.32.
pub const RMS_EPS: Fx = 42_950;
/// 1/sqrt(head_dim) = 0.5 in Q32.32.
pub const INV_SQRT_HD: Fx = ONE / 2;
/// Weight init std: 0.08 in Q32.32.
pub const INIT_STD: Fx = 343_597_384;

/// All 4,192 parameters, row-major `[out][in]`, matching microgpt.py's
/// `linear(x, w) = [dot(w_row, x)]`.
#[repr(C)]
#[derive(Clone)]
pub struct Weights {
    pub wte: [[Fx; N_EMBD]; VOCAB],
    pub wpe: [[Fx; N_EMBD]; BLOCK],
    pub wq: [[Fx; N_EMBD]; N_EMBD],
    pub wk: [[Fx; N_EMBD]; N_EMBD],
    pub wv: [[Fx; N_EMBD]; N_EMBD],
    pub wo: [[Fx; N_EMBD]; N_EMBD],
    pub w1: [[Fx; N_EMBD]; MLP_DIM],
    pub w2: [[Fx; MLP_DIM]; N_EMBD],
    pub lm: [[Fx; N_EMBD]; VOCAB],
}

const _: () = assert!(core::mem::size_of::<Weights>() == N_PARAMS * 8);

impl Weights {
    #[inline]
    pub fn as_flat(&self) -> &[Fx; N_PARAMS] {
        // SAFETY: repr(C) struct of nested Fx arrays is N_PARAMS contiguous i64s.
        unsafe { &*(self as *const Weights as *const [Fx; N_PARAMS]) }
    }

    #[inline]
    pub fn as_flat_mut(&mut self) -> &mut [Fx; N_PARAMS] {
        // SAFETY: as above.
        unsafe { &mut *(self as *mut Weights as *mut [Fx; N_PARAMS]) }
    }

    pub fn zero(&mut self) {
        self.as_flat_mut().fill(0);
    }

    /// PRNG-init parameters in `[start, end)` of the flat layout. Chunkable:
    /// the caller owns the Rng state, so ranges can span transactions as long
    /// as they are applied in order from a fresh Rng.
    pub fn init_range(&mut self, start: usize, end: usize, rng: &mut Rng) {
        for p in &mut self.as_flat_mut()[start..end] {
            *p = rng.next_gauss_scaled(INIT_STD);
        }
    }
}

/// Adam first/second moment buffers, flat, same parameter order as `Weights`.
#[repr(C)]
pub struct Moments {
    pub m: [Fx; N_PARAMS],
    pub v: [Fx; N_PARAMS],
}

/// Activation workspace for one document plus the gradient accumulator.
///
/// Everything the backward pass needs is written here by the forward pass;
/// on-chain this lives in the (delegated, never-committed) scratch account.
#[repr(C)]
pub struct Scratch {
    // Per-position activations. Row t is position t.
    pub x0: [[Fx; N_EMBD]; BLOCK], // wte + wpe
    pub x1: [[Fx; N_EMBD]; BLOCK], // rmsnorm(x0) — residual stream base
    pub xa: [[Fx; N_EMBD]; BLOCK], // rmsnorm(x1) — attention input
    pub q: [[Fx; N_EMBD]; BLOCK],
    pub k: [[Fx; N_EMBD]; BLOCK], // KV cache (doubles as generation cache)
    pub v: [[Fx; N_EMBD]; BLOCK],
    pub o: [[Fx; N_EMBD]; BLOCK],            // concatenated head outputs
    pub x2: [[Fx; N_EMBD]; BLOCK],           // x1 + attn out
    pub xm: [[Fx; N_EMBD]; BLOCK],           // rmsnorm(x2) — MLP input
    pub x3: [[Fx; N_EMBD]; BLOCK],           // x2 + mlp out
    pub att: [[[Fx; BLOCK]; N_HEAD]; BLOCK], // att[t][h][s], softmaxed, s <= t
    pub hpre: [[Fx; MLP_DIM]; BLOCK],
    pub h: [[Fx; MLP_DIM]; BLOCK],
    pub probs: [[Fx; VOCAB]; BLOCK],
    // RMSNorm scales, one per site per position (needed by backward).
    pub s_emb: [Fx; BLOCK],
    pub s_att: [Fx; BLOCK],
    pub s_mlp: [Fx; BLOCK],
    // Gradient accumulators for the KV cache (filled during backward).
    pub dk: [[Fx; N_EMBD]; BLOCK],
    pub dv: [[Fx; N_EMBD]; BLOCK],
    // Parameter gradients, same shape as the weights.
    pub grads: Weights,
}

pub const SCRATCH_LEN: usize = core::mem::size_of::<Scratch>();

// -- Kernels -----------------------------------------------------------------

/// y = W x. Native-multiply kernel (see `fmul16`).
#[inline]
pub fn linear<const OUT: usize, const IN: usize>(w: &[[Fx; IN]; OUT], x: &[Fx; IN]) -> [Fx; OUT] {
    let mut y = [0 as Fx; OUT];
    for (yo, row) in y.iter_mut().zip(w.iter()) {
        let mut acc: Fx = 0;
        for (wi, xi) in row.iter().zip(x.iter()) {
            acc += fmul16(*wi, *xi);
        }
        *yo = acc;
    }
    y
}

/// dx = W^T dy. Native-multiply kernel.
#[inline]
pub fn linear_t<const OUT: usize, const IN: usize>(
    w: &[[Fx; IN]; OUT],
    dy: &[Fx; OUT],
) -> [Fx; IN] {
    let mut dx = [0 as Fx; IN];
    for (row, dyo) in w.iter().zip(dy.iter()) {
        let d = *dyo >> 16;
        for (dxi, wi) in dx.iter_mut().zip(row.iter()) {
            *dxi += (*wi >> 16) * d;
        }
    }
    dx
}

/// dW += dy ⊗ x. Native-multiply kernel.
#[inline]
pub fn outer_acc<const OUT: usize, const IN: usize>(
    dw: &mut [[Fx; IN]; OUT],
    dy: &[Fx; OUT],
    x: &[Fx; IN],
) {
    for (row, dyo) in dw.iter_mut().zip(dy.iter()) {
        let d = *dyo >> 16;
        for (dwi, xi) in row.iter_mut().zip(x.iter()) {
            *dwi += d * (*xi >> 16);
        }
    }
}

/// Dot product. Native-multiply kernel.
#[inline]
pub fn dot<const N: usize>(a: &[Fx; N], b: &[Fx; N]) -> Fx {
    let mut acc: Fx = 0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc += fmul16(*ai, *bi);
    }
    acc
}

/// RMSNorm: y = x / sqrt(mean(x^2) + eps). Returns (y, scale) — backward
/// needs the scale.
#[inline]
pub fn rmsnorm(x: &[Fx; N_EMBD]) -> ([Fx; N_EMBD], Fx) {
    let ms = dot(x, x) / N_EMBD as i64 + RMS_EPS;
    let scale = rsqrt_fx(ms);
    let mut y = [0 as Fx; N_EMBD];
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = fmul(*xi, scale);
    }
    (y, scale)
}

/// dL/dx for y = x * scale(x): dx = s*dy - (s^3/N) * dot(dy, x) * x.
#[inline]
pub fn rmsnorm_bwd(x: &[Fx; N_EMBD], scale: Fx, dy: &[Fx; N_EMBD]) -> [Fx; N_EMBD] {
    let d = dot(dy, x);
    let s3 = fmul(fmul(scale, scale), scale);
    let c = fmul(s3, d) / N_EMBD as i64;
    let mut dx = [0 as Fx; N_EMBD];
    for i in 0..N_EMBD {
        dx[i] = fmul(scale, dy[i]) - fmul(c, x[i]);
    }
    dx
}

/// In-place softmax over `logits[..n]`: subtract max, exp, normalize.
#[inline]
pub fn softmax_inplace(logits: &mut [Fx], n: usize) {
    let max = logits[..n].iter().copied().max().unwrap_or(0);
    let mut sum: Fx = 0;
    for l in logits[..n].iter_mut() {
        *l = exp_fx(*l - max);
        sum += *l;
    }
    // One reciprocal + multiplies: a 128-bit division per element would
    // dominate the whole forward pass on SBF.
    let inv_sum = fdiv(ONE, sum.max(1));
    for l in logits[..n].iter_mut() {
        *l = fmul(*l, inv_sum);
    }
}

// -- Forward -----------------------------------------------------------------

/// Run one position through the transformer, filling the KV cache and all
/// activations for position `t` in the scratch. Positions must be processed
/// in order from t = 0. Returns the raw logits.
#[inline(never)]
pub fn forward_pos(w: &Weights, s: &mut Scratch, t: usize, token: usize) -> [Fx; VOCAB] {
    debug_assert!(t < BLOCK && token < VOCAB);

    for i in 0..N_EMBD {
        s.x0[t][i] = w.wte[token][i] + w.wpe[t][i];
    }
    let (x1, s_emb) = rmsnorm(&s.x0[t]);
    s.x1[t] = x1;
    s.s_emb[t] = s_emb;

    // Attention block.
    let (xa, s_att) = rmsnorm(&s.x1[t]);
    s.xa[t] = xa;
    s.s_att[t] = s_att;
    s.q[t] = linear(&w.wq, &s.xa[t]);
    s.k[t] = linear(&w.wk, &s.xa[t]);
    s.v[t] = linear(&w.wv, &s.xa[t]);

    for head in 0..N_HEAD {
        let hs = head * HEAD_DIM;
        let att = &mut s.att[t][head];
        for past in 0..=t {
            let mut acc: Fx = 0;
            for j in 0..HEAD_DIM {
                acc += fmul16(s.q[t][hs + j], s.k[past][hs + j]);
            }
            att[past] = acc >> 1; // * 1/sqrt(head_dim) = 0.5 exactly
        }
        softmax_inplace(att, t + 1);
        for j in 0..HEAD_DIM {
            let mut acc: Fx = 0;
            for past in 0..=t {
                acc += fmul16(att[past], s.v[past][hs + j]);
            }
            s.o[t][hs + j] = acc;
        }
    }
    let xo = linear(&w.wo, &s.o[t]);
    for i in 0..N_EMBD {
        s.x2[t][i] = s.x1[t][i] + xo[i];
    }

    // MLP block.
    let (xm, s_mlp) = rmsnorm(&s.x2[t]);
    s.xm[t] = xm;
    s.s_mlp[t] = s_mlp;
    s.hpre[t] = linear(&w.w1, &s.xm[t]);
    for i in 0..MLP_DIM {
        s.h[t][i] = s.hpre[t][i].max(0);
    }
    let xf = linear(&w.w2, &s.h[t]);
    for i in 0..N_EMBD {
        s.x3[t][i] = s.x2[t][i] + xf[i];
    }

    linear(&w.lm, &s.x3[t])
}

// -- Sampling ----------------------------------------------------------------

/// Autoregressive sampling, faithful to microgpt.py inference: start from BOS,
/// optionally teacher-force `prefix`, sample with `temp` in (0, 1] until BOS
/// or the context window ends. Returns the number of tokens written to `out`
/// (prefix included).
pub fn generate(
    w: &Weights,
    s: &mut Scratch,
    prefix: &[u8],
    temp: Fx,
    rng: &mut Rng,
    out: &mut [u8; BLOCK],
) -> usize {
    debug_assert!(temp > 0 && prefix.len() < BLOCK);
    let mut token = BOS;
    let mut n_out = 0;
    for pos in 0..BLOCK {
        let mut logits = forward_pos(w, s, pos, token as usize);
        let next = if pos < prefix.len() {
            prefix[pos]
        } else {
            let inv_temp = fdiv(ONE, temp);
            for l in logits.iter_mut() {
                *l = fmul(*l, inv_temp);
            }
            softmax_inplace(&mut logits, VOCAB);
            let mut r = fmul(rng.next_unit(), logits.iter().sum());
            let mut picked = (VOCAB - 1) as u8;
            for (i, p) in logits.iter().enumerate() {
                if r < *p {
                    picked = i as u8;
                    break;
                }
                r -= *p;
            }
            picked
        };
        if next == BOS {
            break;
        }
        out[n_out] = next;
        n_out += 1;
        token = next;
    }
    n_out
}

/// Tokenize an ASCII lowercase name. Returns None on invalid chars.
pub fn tokenize(ch: u8) -> Option<u8> {
    if ch.is_ascii_lowercase() {
        Some(ch - b'a')
    } else {
        None
    }
}

/// Token id back to ASCII.
pub fn detokenize(tok: u8) -> u8 {
    debug_assert!(tok < BOS);
    b'a' + tok
}
