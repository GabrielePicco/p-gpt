//! The complete microGPT training and inference algorithm in deterministic
//! Q32.32 fixed-point arithmetic. Everything else is just efficiency.
//!
//! This crate is `no_std` and dependency-free: it runs identically on the host
//! (tests, parity harness) and inside the SVM (the p-gpt program). All state
//! lives in caller-provided `#[repr(C)]` buffers so the on-chain account data
//! *is* the tensor.
//!
//! Faithful to `reference/microgpt.py` (Karpathy): 1 transformer layer,
//! n_embd 16, 4 heads, block size 16, vocab 27 (a-z + BOS), RMSNorm, ReLU MLP,
//! no biases, Adam with bias correction.

#![cfg_attr(not(test), no_std)]

pub mod adam;
pub mod backprop;
pub mod fixed;
pub mod math;
pub mod model;
pub mod rng;

pub use adam::{adam_range, lr_at_step, train_doc, AdamParams};
pub use backprop::{backward_position, begin_doc, forward_pos_loss, train_fwd_bwd, zero_grads};
pub use fixed::{fdiv, fmul, fmul16, Fx, FRAC, LOSS_SCALE_SHIFT, ONE};
pub use model::{
    forward_pos, generate, Moments, Scratch, Weights, BLOCK, BOS, HEAD_DIM, MLP_DIM, N_EMBD,
    N_HEAD, N_PARAMS, VOCAB,
};
pub use rng::Rng;
