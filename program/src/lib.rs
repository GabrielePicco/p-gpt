//! p-gpt: a pinocchio-based GPT program.
//!
//! Karpathy's microGPT (4,192 parameters) trained perpetually on a MagicBlock
//! Ephemeral Rollup and checkpointed to Solana. Every gradient step is a
//! transaction; the weights account is the tensor.

#![no_std]

pub mod entrypoint;
pub mod processor;

use pinocchio::Address;

/// 6wPpJuYKKPbLYfYZpVeytPwxcq7TdGsgEHwyhYBangEC
pub const ID: Address = Address::new_from_array(p_gpt_interface::PROGRAM_ID);
