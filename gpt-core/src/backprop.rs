//! Hand-derived backward pass through the full transformer.
//!
//! No autograd graph: cross-entropy + softmax fuse to `probs - onehot`, and
//! each block's gradients are written directly into `scratch.grads`. The
//! attention KV cache couples positions (k[s], v[s] receive gradient from
//! every later position t >= s), so backward walks positions in descending
//! order accumulating `dk`/`dv`; by the time position t is processed, its
//! own cache gradients are complete.
//!
//! Frame discipline: SBF stack frames are a fixed 4KB and overflow corrupts
//! the caller's frame silently, so the per-position passes are `inline(never)`
//! — each gets its own frame for its activation-sized temporaries.

use crate::fixed::{fmul16, Fx, LOSS_SCALE_SHIFT, ONE};
use crate::math::ln_fx;
use crate::model::{
    forward_pos, linear_t, outer_acc, rmsnorm_bwd, Scratch, Weights, BLOCK, HEAD_DIM, MLP_DIM,
    N_EMBD, N_HEAD, VOCAB,
};

/// Forward + backward over one document.
///
/// `tokens` is the BOS-delimited sequence (BOS, c_1 .. c_L, BOS); position t
/// consumes `tokens[t]` and predicts `tokens[t + 1]`, for t in 0..n where
/// n = min(BLOCK, tokens.len() - 1) — exactly microgpt.py's loop.
///
/// Gradients (averaged over the n positions) are accumulated into
/// `s.grads`; returns the mean cross-entropy loss in Q32.32.
pub fn train_fwd_bwd(w: &Weights, s: &mut Scratch, tokens: &[u8]) -> Fx {
    let n = BLOCK.min(tokens.len() - 1);
    let loss = begin_doc(w, s, tokens);
    for t in (0..n).rev() {
        backward_position(w, s, t, tokens[t], tokens[t + 1], n);
    }
    loss
}

/// Phase 0 of a training step: zero the gradient accumulators and run the
/// forward pass over the whole document, keeping every activation in the
/// scratch. Returns the mean loss. Split-transaction training (the sub-1.4M
/// CU path) calls this once, then `backward_position` per position (highest
/// first), then Adam.
pub fn begin_doc(w: &Weights, s: &mut Scratch, tokens: &[u8]) -> Fx {
    let n = BLOCK.min(tokens.len() - 1);
    debug_assert!(n >= 1);

    zero_grads(s);
    let mut loss: Fx = 0;
    for t in 0..n {
        loss -= forward_pos_loss(w, s, t, tokens[t], tokens[t + 1]);
    }
    loss / n as i64
}

/// Reset the gradient and KV-cache-gradient accumulators for a new step.
pub fn zero_grads(s: &mut Scratch) {
    s.grads.zero();
    for t in 0..BLOCK {
        s.dk[t] = [0; N_EMBD];
        s.dv[t] = [0; N_EMBD];
    }
}

/// Forward one position and store softmax probs; returns ln(p_target).
#[inline(never)]
pub fn forward_pos_loss(w: &Weights, s: &mut Scratch, t: usize, token: u8, target: u8) -> Fx {
    let mut logits = forward_pos(w, s, t, token as usize);
    crate::model::softmax_inplace(&mut logits, VOCAB);
    s.probs[t] = logits;
    ln_fx(s.probs[t][target as usize].max(1)) // clamp: ln(0) is -inf
}

/// Backward through one position (positions must be processed in descending
/// order after `begin_doc`), accumulating into `s.grads` / `s.dk` / `s.dv`.
/// Own stack frame: its temporaries are activation-sized.
#[inline(never)]
pub fn backward_position(w: &Weights, s: &mut Scratch, t: usize, token: u8, target: u8, n: usize) {
    // Fused softmax + cross-entropy (mean over n positions), loss-scaled by
    // 2^LOSS_SCALE_SHIFT so small gradients survive the Q16.16 kernels.
    let mut dlogits = s.probs[t];
    dlogits[target as usize] -= ONE;
    for d in dlogits.iter_mut() {
        *d = (*d << LOSS_SCALE_SHIFT) / n as i64;
    }

    // LM head: logits = lm @ x3.
    outer_acc(&mut s.grads.lm, &dlogits, &s.x3[t]);
    let dx3 = linear_t(&w.lm, &dlogits);

    // MLP block: x3 = x2 + w2 @ relu(w1 @ rmsnorm(x2)).
    let mut dx2 = dx3;
    mlp_backward(w, s, t, &dx3, &mut dx2);

    // Attention block: x2 = x1 + wo @ concat(heads(rmsnorm(x1))).
    let mut dx1 = dx2;
    attention_backward(w, s, t, &dx2, &mut dx1);

    // Embedding: x0 = wte[token] + wpe[t]; x1 = rmsnorm(x0).
    let dx0 = rmsnorm_bwd(&s.x0[t], s.s_emb[t], &dx1);
    for i in 0..N_EMBD {
        s.grads.wte[token as usize][i] += dx0[i];
        s.grads.wpe[t][i] += dx0[i];
    }
}

/// dLoss/dx2 += MLP-path gradient; accumulates w1/w2 grads.
#[inline(never)]
fn mlp_backward(
    w: &Weights,
    s: &mut Scratch,
    t: usize,
    dx3: &[Fx; N_EMBD],
    dx2: &mut [Fx; N_EMBD],
) {
    outer_acc(&mut s.grads.w2, dx3, &s.h[t]);
    let dh = linear_t(&w.w2, dx3);
    let mut dhpre = [0 as Fx; MLP_DIM];
    for i in 0..MLP_DIM {
        if s.hpre[t][i] > 0 {
            dhpre[i] = dh[i];
        }
    }
    outer_acc(&mut s.grads.w1, &dhpre, &s.xm[t]);
    let dxm = linear_t(&w.w1, &dhpre);
    let dx2_norm = rmsnorm_bwd(&s.x2[t], s.s_mlp[t], &dxm);
    for i in 0..N_EMBD {
        dx2[i] += dx2_norm[i];
    }
}

/// dLoss/dx1 += attention-path gradient; accumulates wq/wk/wv/wo grads and
/// the KV-cache gradients for earlier positions.
#[inline(never)]
fn attention_backward(
    w: &Weights,
    s: &mut Scratch,
    t: usize,
    dx2: &[Fx; N_EMBD],
    dx1: &mut [Fx; N_EMBD],
) {
    outer_acc(&mut s.grads.wo, dx2, &s.o[t]);
    let do_ = linear_t(&w.wo, dx2);

    let mut dq = [0 as Fx; N_EMBD];
    for head in 0..N_HEAD {
        let hs = head * HEAD_DIM;
        let att = &s.att[t][head];

        // da[s] = do_h . v_h[s]; softmax backward gives the score grads.
        let mut da = [0 as Fx; BLOCK];
        let mut dsum: Fx = 0;
        for past in 0..=t {
            let mut acc: Fx = 0;
            for j in 0..HEAD_DIM {
                acc += fmul16(do_[hs + j], s.v[past][hs + j]);
            }
            da[past] = acc;
            dsum += fmul16(att[past], da[past]);
        }
        for past in 0..=t {
            let dscore = fmul16(att[past], da[past] - dsum) >> 1; // * 1/sqrt(hd)
            for j in 0..HEAD_DIM {
                dq[hs + j] += fmul16(dscore, s.k[past][hs + j]);
                s.dk[past][hs + j] += fmul16(dscore, s.q[t][hs + j]);
                s.dv[past][hs + j] += fmul16(att[past], do_[hs + j]);
            }
        }
    }

    // Project cache gradients for position t (now complete) and the query
    // gradient back through the input projections.
    outer_acc(&mut s.grads.wq, &dq, &s.xa[t]);
    let mut dxa = linear_t(&w.wq, &dq);
    outer_acc(&mut s.grads.wk, &s.dk[t], &s.xa[t]);
    let dxa_k = linear_t(&w.wk, &s.dk[t]);
    outer_acc(&mut s.grads.wv, &s.dv[t], &s.xa[t]);
    let dxa_v = linear_t(&w.wv, &s.dv[t]);
    for i in 0..N_EMBD {
        dxa[i] += dxa_k[i] + dxa_v[i];
    }

    let dx1_norm = rmsnorm_bwd(&s.x1[t], s.s_att[t], &dxa);
    for i in 0..N_EMBD {
        dx1[i] += dx1_norm[i];
    }
}
