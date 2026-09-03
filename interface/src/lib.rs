//! p-gpt interface: account layouts, PDA seeds, and instruction encoding.
//!
//! Shared by the on-chain program, tests, and clients. Contains no pinocchio
//! types so any consumer can depend on it; addresses are plain `[u8; 32]`.

#![no_std]

use gpt_core::{Fx, Moments, Scratch, Weights, BLOCK};

pub mod instruction;
pub mod state;

/// Program ID: 6wPpJuYKKPbLYfYZpVeytPwxcq7TdGsgEHwyhYBangEC
pub const PROGRAM_ID: [u8; 32] = [
    0x58, 0x39, 0x98, 0x98, 0xa9, 0x63, 0x8d, 0xfe, 0x62, 0xa2, 0x35, 0xf1, 0x69, 0xf5, 0x52, 0x9d,
    0xef, 0x8b, 0x94, 0xf3, 0xc4, 0x68, 0xd6, 0x18, 0x54, 0x80, 0x37, 0x7a, 0x91, 0x12, 0x55, 0x91,
];

/// PDA seeds. All accounts are singletons — one perpetual model per program.
pub mod seeds {
    pub const MODEL: &[u8] = b"model";
    pub const OPTIMIZER: &[u8] = b"opt";
    pub const SCRATCH: &[u8] = b"scratch";
    pub const DATASET: &[u8] = b"data";
    pub const COMMUNITY: &[u8] = b"community";
    pub const GENLOG: &[u8] = b"gen";
    /// Checkpoint shards: seeds [SHARD, [k]] for k in 0..SHARD_COUNT.
    pub const SHARD: &[u8] = b"shard";
}

/// The model account (header + weights) exceeds what the delegation program
/// can commit on a vanilla runtime, so checkpoints go through shard accounts:
/// each holds a slice of the model image and stays under 10,240 bytes.
pub const SHARD_COUNT: usize = 4;
pub const SHARD_LEN: usize = MODEL_ACCOUNT_LEN.div_ceil(SHARD_COUNT);

/// Index of each PDA's bump in `ModelHeader::bumps`. Values 6..10 address
/// the checkpoint shards (bumps derived on use, not stored).
pub mod bump_ix {
    pub const MODEL: usize = 0;
    pub const OPTIMIZER: usize = 1;
    pub const SCRATCH: usize = 2;
    pub const DATASET: usize = 3;
    pub const COMMUNITY: usize = 4;
    pub const GENLOG: usize = 5;
    pub const SHARD0: usize = 6;
}

/// Deterministic pseudo-shuffle stride, co-prime with the dataset size.
pub const DOC_STRIDE: u64 = 9973;
/// Every Nth step trains on a community-contributed doc (when any exist).
pub const COMMUNITY_EVERY: u64 = 8;
/// EMA shift for the reported loss: ema += (loss - ema) >> 6.
pub const LOSS_EMA_SHIFT: u32 = 6;

pub const DATASET_CAPACITY: u64 = 32_768;
/// Sized so the community account stays under 10,240 bytes — the delegation
/// program cannot commit/undelegate larger accounts on a vanilla runtime
/// (commit-state creation is capped by the CPI realloc limit).
pub const COMMUNITY_CAPACITY: u64 = 512;
pub const GENLOG_CAPACITY: u64 = 256;
pub const LOSS_RING_LEN: usize = 256;

/// A name, padded: byte 0 = length (1..=15), bytes 1..=15 = token ids (0..26).
pub const DOC_RECORD_LEN: usize = 16;
pub const MAX_NAME_LEN: usize = BLOCK - 1;

pub const MODEL_ACCOUNT_LEN: usize =
    core::mem::size_of::<state::ModelHeader>() + core::mem::size_of::<Weights>();
pub const OPTIMIZER_ACCOUNT_LEN: usize = core::mem::size_of::<Moments>();
/// Two independent workspaces: training (first half) and generation (second
/// half), so `Generate` can never corrupt an in-flight split training step.
pub const SCRATCH_ACCOUNT_LEN: usize = 2 * core::mem::size_of::<Scratch>();
pub const GEN_SCRATCH_OFFSET: usize = core::mem::size_of::<Scratch>();
pub const DATASET_ACCOUNT_LEN: usize =
    core::mem::size_of::<state::DocsHeader>() + DATASET_CAPACITY as usize * DOC_RECORD_LEN;
pub const COMMUNITY_ACCOUNT_LEN: usize =
    core::mem::size_of::<state::DocsHeader>() + COMMUNITY_CAPACITY as usize * DOC_RECORD_LEN;
pub const GENLOG_ACCOUNT_LEN: usize = core::mem::size_of::<state::GenLogHeader>()
    + GENLOG_CAPACITY as usize * core::mem::size_of::<state::GenRecord>();

/// Q32.32 alias re-exported for clients.
pub type Q32 = Fx;
