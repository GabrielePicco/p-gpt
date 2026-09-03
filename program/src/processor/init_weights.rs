use gpt_core::{Rng, N_PARAMS};
use p_gpt_interface::instruction::InitWeightsArgs;
use p_gpt_interface::state::FLAG_WEIGHTS_READY;
use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use super::shared::{check_model, check_owned, model_parts};

/// PRNG-initialize the next chunk of weights (Gaussian, std 0.08).
///
/// Chunks must be applied in order; the PRNG state is persisted in the header
/// so the full init is a pure function of the seed regardless of chunking.
pub fn process_init_weights(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = InitWeightsArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;

    let [model] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    check_owned(model)?;

    let mut data = model.try_borrow_mut()?;
    let (header, weights) = model_parts(&mut data)?;
    check_model(header)?;

    if header.flags & FLAG_WEIGHTS_READY != 0 {
        return Err(ProgramError::AccountAlreadyInitialized);
    }

    let start = header.init_cursor as usize;
    let end = N_PARAMS.min(start + args.count.max(1) as usize);
    let mut rng = Rng(header.rng_state);
    weights.init_range(start, end, &mut rng);
    header.rng_state = rng.0;
    header.init_cursor = end as u64;

    if end == N_PARAMS {
        header.flags |= FLAG_WEIGHTS_READY;
    }
    Ok(())
}
