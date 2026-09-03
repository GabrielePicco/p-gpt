use p_gpt_interface::{seeds, SHARD_COUNT, SHARD_LEN};
use pinocchio::cpi::{Seed, Signer};
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult};
use pinocchio_system::instructions::CreateAccount;

use super::shared::rent_exempt_minimum;

/// Create the checkpoint shard accounts. Each holds a slice of the model
/// image and stays under the 10,240-byte creation (and delegation-commit)
/// limit, so the full model can reach the base layer through them.
pub fn process_init_shards(accounts: &mut [AccountView], _data: &[u8]) -> ProgramResult {
    let [payer, shards @ .., _system] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    if !payer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if shards.len() != SHARD_COUNT {
        return Err(ProgramError::NotEnoughAccountKeys);
    }

    for (k, shard) in shards.iter().enumerate() {
        let index = [k as u8];
        let (expected, bump) = Address::find_program_address(&[seeds::SHARD, &index], &crate::ID);
        if shard.address() != &expected {
            return Err(ProgramError::InvalidSeeds);
        }
        if !shard.is_data_empty() {
            continue; // idempotent
        }
        let bump_seed = [bump];
        let signer_seeds = [Seed::from(seeds::SHARD), Seed::from(&index), Seed::from(&bump_seed)];
        CreateAccount {
            from: payer,
            to: shard,
            lamports: rent_exempt_minimum(SHARD_LEN)?,
            space: SHARD_LEN as u64,
            owner: &crate::ID,
        }
        .invoke_signed(&[Signer::from(&signer_seeds)])?;
    }
    Ok(())
}
