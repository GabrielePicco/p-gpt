use gpt_core::{generate, Rng, Scratch, BLOCK, BOS, ONE};
use p_gpt_interface::instruction::GenerateArgs;
use p_gpt_interface::state::{GenRecord, FLAG_WEIGHTS_READY};
use p_gpt_interface::{bump_ix, seeds, MAX_NAME_LEN};
use pinocchio::sysvars::{clock::Clock, Sysvar};
use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use super::shared::{cast_mut, check_model, check_owned, check_pda, genlog_parts, model_parts};

/// Sample a name from the live model, optionally continuing a prefix.
///
/// Randomness mixes caller entropy with the slot and the generation counter;
/// it is deterministic under replay (the slot is part of the record).
pub fn process_generate(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = GenerateArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;
    // Temperatures below 1/64 overflow the Q32.32 reciprocal (and sample
    // effectively greedily anyway).
    if args.temperature < ONE / 64 || args.temperature > ONE {
        return Err(ProgramError::InvalidInstructionData);
    }
    if args.prefix.len() > MAX_NAME_LEN || args.prefix.iter().any(|t| *t >= BOS) {
        return Err(ProgramError::InvalidInstructionData);
    }

    let [model, scratch, genlog] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    check_owned(model)?;
    check_owned(scratch)?;
    check_owned(genlog)?;

    let mut model_data = model.try_borrow_mut()?;
    let (header, weights) = model_parts(&mut model_data)?;
    check_model(header)?;
    if header.flags & FLAG_WEIGHTS_READY == 0 {
        return Err(ProgramError::UninitializedAccount);
    }
    check_pda(scratch, seeds::SCRATCH, header.bumps[bump_ix::SCRATCH])?;
    check_pda(genlog, seeds::GENLOG, header.bumps[bump_ix::GENLOG])?;

    // Use the generation half of the scratch so an in-flight split training
    // step's activations are untouched.
    let mut scratch_data = scratch.try_borrow_mut()?;
    if scratch_data.len() < p_gpt_interface::SCRATCH_ACCOUNT_LEN {
        return Err(ProgramError::InvalidAccountData);
    }
    let workspace: &mut Scratch =
        cast_mut(&mut scratch_data[p_gpt_interface::GEN_SCRATCH_OFFSET..])?;

    let clock = Clock::get()?;
    let mut rng = Rng::new(
        args.seed
            ^ clock.slot.rotate_left(17)
            ^ header.gen_count.wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );

    let mut out = [0u8; BLOCK];
    let n = generate(weights, workspace, args.prefix, args.temperature, &mut rng, &mut out);
    header.gen_count += 1;

    // Record it in the ring.
    let mut genlog_data = genlog.try_borrow_mut()?;
    let (log_header, records) = genlog_parts(&mut genlog_data)?;
    let mut name = [0u8; 16];
    name[..n].copy_from_slice(&out[..n]);
    records[(log_header.total % log_header.capacity) as usize] =
        GenRecord { step: header.step, len: n as u8, name, _pad: [0; 7] };
    log_header.total += 1;

    #[cfg(feature = "logging")]
    {
        let mut ascii = [0u8; 16];
        for i in 0..n {
            ascii[i] = b'a' + out[i];
        }
        if let Ok(text) = core::str::from_utf8(&ascii[..n]) {
            pinocchio_log::log!("p-gpt: gen step={} name={}", header.step, text);
        }
    }

    Ok(())
}
