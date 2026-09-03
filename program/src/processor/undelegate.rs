use ephemeral_rollups_pinocchio::consts::DELEGATION_PROGRAM_ID;
use ephemeral_rollups_pinocchio::intent_bundle::MagicIntentBundleBuilder;
use ephemeral_rollups_pinocchio::pda::undelegate_buffer_pda_from_delegated_account;
use p_gpt_interface::seeds;
use pinocchio::cpi::{Seed, Signer};
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult};
use pinocchio_system::instructions::CreateAccount;

use super::shared::{cast_ref, check_model, check_owned, rent_exempt_minimum};

/// Commit and undelegate the committable accounts — community, genlog and
/// the checkpoint shards. The large working accounts (model, optimizer,
/// scratch) exceed what the delegation program can commit on a vanilla
/// runtime and stay delegated; the model's full image lives on in the
/// shards, from which it is reconstructible.
pub fn process_undelegate(accounts: &mut [AccountView], _data: &[u8]) -> ProgramResult {
    let [payer, magic_context, magic_program, model, community, genlog, shards @ ..] = accounts
    else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    if !payer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    // Only the model authority may pull accounts back to the base layer.
    let (model_expected, _) = Address::find_program_address(&[seeds::MODEL], &crate::ID);
    if model.address() != &model_expected {
        return Err(ProgramError::InvalidSeeds);
    }
    {
        let data = model.try_borrow()?;
        let header: &p_gpt_interface::state::ModelHeader = cast_ref(&data)?;
        check_model(header)?;
        if payer.address().as_array() != &header.authority {
            return Err(ProgramError::IncorrectAuthority);
        }
    }
    check_owned(community)?;
    check_owned(genlog)?;
    let (community_expected, _) = Address::find_program_address(&[seeds::COMMUNITY], &crate::ID);
    let (genlog_expected, _) = Address::find_program_address(&[seeds::GENLOG], &crate::ID);
    if community.address() != &community_expected || genlog.address() != &genlog_expected {
        return Err(ProgramError::InvalidSeeds);
    }
    if shards.len() != p_gpt_interface::SHARD_COUNT {
        return Err(ProgramError::NotEnoughAccountKeys);
    }
    for (k, shard) in shards.iter().enumerate() {
        let index = [k as u8];
        let (expected, _) = Address::find_program_address(&[seeds::SHARD, &index], &crate::ID);
        if shard.address() != &expected {
            return Err(ProgramError::InvalidSeeds);
        }
        check_owned(shard)?;
    }

    let committed = [
        community.clone(),
        genlog.clone(),
        shards[0].clone(),
        shards[1].clone(),
        shards[2].clone(),
        shards[3].clone(),
    ];
    let mut buf = [0u8; 1024];
    MagicIntentBundleBuilder::new(payer.clone(), magic_context.clone(), magic_program.clone())
        .commit_and_undelegate(&committed)
        .build_and_invoke(&mut buf)
}

/// Callback CPI'd by the delegation program while finalizing an undelegation:
/// moves the buffered state back into the PDA re-created under this program.
///
/// Reimplements the SDK helper because it funds the re-created account via
/// pinocchio's `Rent::try_minimum_balance`, which returns the per-byte-year
/// rate (half the true rent-exempt minimum) — the delegation program's
/// post-CPI validator balance check then rejects the whole undelegation
/// with `InvalidValidatorBalanceAfterCPI`.
pub fn process_undelegate_callback(
    accounts: &mut [AccountView],
    mut callback_args: &[u8],
) -> ProgramResult {
    let [delegated, buffer, payer, _system] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };

    if !buffer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !buffer.owned_by(&DELEGATION_PROGRAM_ID) {
        return Err(ProgramError::InvalidAccountOwner);
    }
    if buffer.address() != &undelegate_buffer_pda_from_delegated_account(delegated.address()) {
        return Err(ProgramError::InvalidSeeds);
    }

    // Parse the serialized PDA seeds: u32 count, then (u32 len, bytes) each.
    fn read_u32(bytes: &mut &[u8]) -> Result<usize, ProgramError> {
        let (v, rest) =
            bytes.split_first_chunk::<4>().ok_or(ProgramError::InvalidInstructionData)?;
        *bytes = rest;
        Ok(u32::from_le_bytes(*v) as usize)
    }
    let seeds_len = read_u32(&mut callback_args)?;
    if seeds_len == 0 || seeds_len > 4 {
        return Err(ProgramError::InvalidInstructionData);
    }
    let mut seed_refs: [&[u8]; 4] = [&[]; 4];
    for seed in seed_refs.iter_mut().take(seeds_len) {
        let len = read_u32(&mut callback_args)?;
        if callback_args.len() < len {
            return Err(ProgramError::InvalidInstructionData);
        }
        *seed = &callback_args[..len];
        callback_args = &callback_args[len..];
    }
    if !callback_args.is_empty() {
        return Err(ProgramError::InvalidInstructionData);
    }
    let pda_seeds = &seed_refs[..seeds_len];

    let (expected, bump) = Address::find_program_address(pda_seeds, &crate::ID);
    if delegated.address() != &expected {
        return Err(ProgramError::InvalidSeeds);
    }

    // Re-create the PDA fully rent-exempt and restore the buffered state.
    let bump_slice = [bump];
    let empty: &[u8] = &[];
    let mut signer_seeds: [Seed; 5] = [
        Seed::from(empty),
        Seed::from(empty),
        Seed::from(empty),
        Seed::from(empty),
        Seed::from(&bump_slice),
    ];
    for (slot, seed) in signer_seeds.iter_mut().zip(pda_seeds.iter()) {
        *slot = Seed::from(*seed);
    }
    signer_seeds[seeds_len] = Seed::from(&bump_slice);
    let signer = Signer::from(&signer_seeds[..seeds_len + 1]);

    let space = buffer.data_len();
    CreateAccount {
        from: payer,
        to: delegated,
        lamports: rent_exempt_minimum(space)?,
        space: space as u64,
        owner: &crate::ID,
    }
    .invoke_signed(&[signer])?;

    let mut data = delegated.try_borrow_mut()?;
    let source = buffer.try_borrow()?;
    data.copy_from_slice(&source);

    Ok(())
}
