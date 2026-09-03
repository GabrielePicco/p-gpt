//! Zero-copy account layouts. Every struct is `#[repr(C)]` with 8-byte
//! alignment; account data is reinterpreted in place, never deserialized.

use crate::{GENLOG_CAPACITY, LOSS_RING_LEN};
use gpt_core::Fx;

pub const MODEL_MAGIC: [u8; 4] = *b"pGPT";
pub const STATE_VERSION: u8 = 1;

/// Header of the model account; followed immediately by `gpt_core::Weights`.
#[repr(C)]
pub struct ModelHeader {
    pub magic: [u8; 4],
    pub version: u8,
    /// PDA bumps, indexed by `crate::bump_ix`.
    pub bumps: [u8; 6],
    /// Bit 0: weights fully initialized.
    pub flags: u8,
    pub _pad: [u8; 4],
    pub authority: [u8; 32],
    /// Init seed — with the transaction history, this determines the weights.
    pub seed: u64,
    /// PRNG state for chunked weight init.
    pub rng_state: u64,
    /// Number of parameters initialized so far.
    pub init_cursor: u64,
    /// Completed training steps.
    pub step: u64,
    /// beta1^step, beta2^step (Q32.32) for Adam bias correction.
    pub pow_beta1: Fx,
    pub pow_beta2: Fx,
    /// Exponential moving average of the training loss (Q32.32).
    pub loss_ema: Fx,
    pub last_loss: Fx,
    /// Total names sampled via `Generate`.
    pub gen_count: u64,
    /// Next write slot in `loss_ring` (monotonic; index is `% LOSS_RING_LEN`).
    pub ring_pos: u64,
    /// Recent per-step losses (Q32.32); entry at `(ring_pos - 1) % LEN` is
    /// the loss of step `step - 1`.
    pub loss_ring: [Fx; LOSS_RING_LEN],
    // -- Split-transaction training state (the sub-1.4M CU path) ------------
    /// Current phase: PHASE_FORWARD, PHASE_BACKWARD or PHASE_ADAM.
    pub phase: u8,
    /// PHASE_BACKWARD: next position to backprop (descending).
    /// PHASE_ADAM: next parameter chunk index.
    pub phase_cursor: u8,
    /// Length in tokens of `doc` (doc_len + 2 BOS delimiters).
    pub doc_tokens_len: u8,
    pub _pad2: [u8; 5],
    /// The BOS-delimited document of the in-flight step.
    pub doc: [u8; 18],
    pub _pad3: [u8; 6],
    /// Loss of the in-flight step, applied to the ring when the step lands.
    pub pending_loss: Fx,
    pub _reserved: [u8; 80],
}

pub const FLAG_WEIGHTS_READY: u8 = 1 << 0;

/// Split-training phases. Micro-op granularity is set by the ER crank's
/// per-tick budget (the default ~200K CU per instruction — a tick cannot
/// request more): one forward position, one backward position, or one small
/// Adam chunk per transaction.
pub const PHASE_PICK: u8 = 0;
pub const PHASE_FORWARD: u8 = 1;
pub const PHASE_BACKWARD: u8 = 2;
pub const PHASE_ADAM: u8 = 3;

/// Adam parameters processed per `TrainMicro` transaction.
pub const ADAM_CHUNK: usize = 256;

const _: () = assert!(core::mem::size_of::<ModelHeader>() % 8 == 0);

/// Header shared by the dataset and community accounts; followed by
/// `capacity` fixed 16-byte doc records (len byte + up to 15 token ids).
#[repr(C)]
pub struct DocsHeader {
    pub magic: [u8; 4],
    pub _pad: [u8; 4],
    pub capacity: u64,
    pub count: u64,
}

/// One generated name. Generation can emit up to BLOCK (16) tokens — one per
/// context position — unlike dataset names, which cap at 15.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GenRecord {
    /// Training step at which this name was sampled.
    pub step: u64,
    pub len: u8,
    /// Token ids (0..26), valid up to `len`.
    pub name: [u8; 16],
    pub _pad: [u8; 7],
}

/// Header of the generation log; followed by `GENLOG_CAPACITY` records used
/// as a ring buffer.
#[repr(C)]
pub struct GenLogHeader {
    pub magic: [u8; 4],
    pub _pad: [u8; 4],
    pub capacity: u64,
    /// Total generations ever; the ring index is `total % capacity`.
    pub total: u64,
    pub _reserved: [u8; 8],
}

const _: () = assert!(GENLOG_CAPACITY.is_power_of_two());
