use ephemeral_rollups_pinocchio::intent_bundle::MagicIntentBundleBuilder;
use p_gpt_interface::state::{ModelHeader, PHASE_ADAM};
use p_gpt_interface::{seeds, MODEL_ACCOUNT_LEN, SHARD_COUNT, SHARD_LEN};
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult};

use super::shared::{cast_ref, check_model, check_owned};

/// Commit the model state from the ER back to the base layer — the perpetual
/// model's heartbeat to Solana. Permissionless.
///
/// The model account itself is too large for the delegation program to
/// commit on a vanilla runtime, so its full image (header + weights) is
/// first copied into the sub-10KB shard accounts, and the shards + genlog
/// are committed.
pub fn process_checkpoint(accounts: &mut [AccountView], _data: &[u8]) -> ProgramResult {
    let [payer, magic_context, magic_program, model, genlog, shards @ ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    if !payer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    check_owned(model)?;
    check_owned(genlog)?;
    if shards.len() != SHARD_COUNT {
        return Err(ProgramError::NotEnoughAccountKeys);
    }
    // Bind to the canonical PDAs — several program-owned accounts are large
    // enough (and share the magic) to be substituted otherwise.
    let (model_expected, _) = Address::find_program_address(&[seeds::MODEL], &crate::ID);
    if model.address() != &model_expected {
        return Err(ProgramError::InvalidSeeds);
    }
    let (genlog_expected, _) = Address::find_program_address(&[seeds::GENLOG], &crate::ID);
    if genlog.address() != &genlog_expected {
        return Err(ProgramError::InvalidSeeds);
    }

    // Sync: copy the model image into the shards.
    let image = model.try_borrow()?;
    if image.len() < MODEL_ACCOUNT_LEN {
        return Err(ProgramError::InvalidAccountData);
    }
    let header: &ModelHeader = cast_ref(&image)?;
    check_model(header)?;
    // Weights are torn only while Adam chunks are being applied; every other
    // phase is a consistent post-step snapshot.
    if header.phase == PHASE_ADAM {
        return Err(ProgramError::InvalidAccountData);
    }
    for (k, shard) in shards.iter_mut().enumerate() {
        let index = [k as u8];
        let (expected, _) = Address::find_program_address(&[seeds::SHARD, &index], &crate::ID);
        if shard.address() != &expected {
            return Err(ProgramError::InvalidSeeds);
        }
        check_owned(shard)?;
        let start = k * SHARD_LEN;
        let end = MODEL_ACCOUNT_LEN.min(start + SHARD_LEN);
        let mut out = shard.try_borrow_mut()?;
        out[..end - start].copy_from_slice(&image[start..end]);
    }
    drop(image);

    // Commit: genlog + shards.
    let committed = [
        genlog.clone(),
        shards[0].clone(),
        shards[1].clone(),
        shards[2].clone(),
        shards[3].clone(),
    ];
    let mut buf = [0u8; 1024];
    MagicIntentBundleBuilder::new(payer.clone(), magic_context.clone(), magic_program.clone())
        .commit(&committed)
        .build_and_invoke(&mut buf)
}
