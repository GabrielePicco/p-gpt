use p_gpt_interface::instruction::GrowArgs;
use p_gpt_interface::{
    bump_ix, COMMUNITY_ACCOUNT_LEN, DATASET_ACCOUNT_LEN, GENLOG_ACCOUNT_LEN, MODEL_ACCOUNT_LEN,
    OPTIMIZER_ACCOUNT_LEN, SCRATCH_ACCOUNT_LEN,
};
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult, Resize};
use pinocchio_system::instructions::Transfer;

use super::shared::{rent_exempt_minimum, seeds_for};

/// The runtime caps account data growth at this many bytes per instruction.
pub const MAX_GROW: usize = 10_240;

/// Target size of each program account.
pub fn target_len(which: usize) -> Result<usize, ProgramError> {
    Ok(match which {
        bump_ix::MODEL => MODEL_ACCOUNT_LEN,
        bump_ix::OPTIMIZER => OPTIMIZER_ACCOUNT_LEN,
        bump_ix::SCRATCH => SCRATCH_ACCOUNT_LEN,
        bump_ix::DATASET => DATASET_ACCOUNT_LEN,
        bump_ix::COMMUNITY => COMMUNITY_ACCOUNT_LEN,
        bump_ix::GENLOG => GENLOG_ACCOUNT_LEN,
        k if k >= bump_ix::SHARD0 && k < bump_ix::SHARD0 + p_gpt_interface::SHARD_COUNT => {
            p_gpt_interface::SHARD_LEN
        }
        _ => return Err(ProgramError::InvalidInstructionData),
    })
}

/// Grow one program account by up to 10,240 bytes toward its target size,
/// topping up rent from the payer. Called repeatedly after `InitModel` until
/// every account reaches its full size (before delegation).
pub fn process_grow(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = GrowArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;
    let target = target_len(args.which as usize)?;
    let mut seed_buf: [&'static [u8]; 2] = [&[], &[]];
    let seeds = seeds_for(args.which as usize, &mut seed_buf)?;

    let [payer, pda, _system] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    if !payer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    let (expected, _) = Address::find_program_address(seeds, &crate::ID);
    if pda.address() != &expected {
        return Err(ProgramError::InvalidSeeds);
    }
    if !pda.owned_by(&crate::ID) {
        return Err(ProgramError::InvalidAccountOwner);
    }

    let current = pda.data_len();
    if current >= target {
        return Ok(()); // idempotent
    }
    let new_len = target.min(current + MAX_GROW);

    let required = rent_exempt_minimum(new_len)?;
    if required > pda.lamports() {
        Transfer { from: payer, to: pda, lamports: required - pda.lamports() }.invoke()?;
    }
    pda.resize(new_len)
}
