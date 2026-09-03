use ephemeral_rollups_pinocchio::consts::{BUFFER, DELEGATION_PROGRAM_ID};
use ephemeral_rollups_pinocchio::instruction::{delegate_account, fill_seeds};
use ephemeral_rollups_pinocchio::types::{DelegateAccountArgs, DelegateConfig};
use ephemeral_rollups_pinocchio::utils::{close_pda_acc, cpi_delegate, make_seed_buf};
use p_gpt_interface::instruction::{DelegateArgs, GrowArgs};
use pinocchio::cpi::{Seed, Signer};
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult, Resize};
use pinocchio_system::instructions::{Assign, CreateAccount, Transfer};

use super::grow::{target_len, MAX_GROW};
use super::shared::{cast_ref, check_model, rent_exempt_minimum, seeds_for};

/// The model authority must sign delegation-lifecycle instructions; the
/// canonical model account is passed as the trailing account so the check
/// works for every target (for the model itself it is the same account).
fn require_authority(accounts: &[AccountView]) -> Result<(), ProgramError> {
    let payer = accounts.first().ok_or(ProgramError::NotEnoughAccountKeys)?;
    if !payer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    let model = accounts.last().ok_or(ProgramError::NotEnoughAccountKeys)?;
    let (expected, _) = Address::find_program_address(&[p_gpt_interface::seeds::MODEL], &crate::ID);
    if model.address() != &expected {
        return Err(ProgramError::InvalidSeeds);
    }
    let data = model.try_borrow()?;
    let header: &p_gpt_interface::state::ModelHeader = cast_ref(&data)?;
    check_model(header)?;
    if payer.address().as_array() != &header.authority {
        return Err(ProgramError::IncorrectAuthority);
    }
    Ok(())
}

/// Delegate one program PDA to the ephemeral rollup.
///
/// Accounts <= 10KB go through the SDK path directly. Larger accounts (all of
/// p-gpt's) need their delegate buffer pre-created via `DelegatePrep` — the
/// runtime caps account creation at 10,240 bytes per instruction — after
/// which this instruction performs the same flow as the SDK with the creation
/// skipped: copy into buffer, zero + assign the PDA, CPI the delegation
/// program, close the buffer back to the payer.
pub fn process_delegate(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = DelegateArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;
    let mut seed_buf: [&'static [u8]; 2] = [&[], &[]];
    let seeds = seeds_for(args.which as usize, &mut seed_buf)?;
    let config = DelegateConfig {
        commit_frequency_ms: args.commit_frequency_ms,
        validator: args.validator.map(Address::new_from_array),
    };

    // Trailing accounts: the delegation program (for the CPI) and the model
    // account (for the authority check).
    require_authority(accounts)?;
    let [payer, pda, owner_program, buffer, delegation_record, delegation_metadata, system_program, ..] =
        accounts
    else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let (expected, bump) = Address::find_program_address(seeds, &crate::ID);
    if pda.address() != &expected {
        return Err(ProgramError::InvalidSeeds);
    }
    // Never delegate a partially grown account: its buffer (and later its
    // commits) would freeze the wrong size.
    if pda.data_len() != target_len(args.which as usize)? {
        return Err(ProgramError::InvalidAccountData);
    }

    let data_len = pda.data_len();
    if data_len <= MAX_GROW {
        return delegate_account(&mut accounts[..7], seeds, bump, config);
    }

    // -- Large-account path --------------------------------------------------
    if !buffer.owned_by(&crate::ID) || buffer.data_len() != data_len {
        // Run `DelegatePrep` until the buffer matches the account size.
        return Err(ProgramError::UninitializedAccount);
    }

    // Copy the account into the buffer, then zero the account.
    {
        let source = pda.try_borrow()?;
        let mut copy = buffer.try_borrow_mut()?;
        copy.copy_from_slice(&source);
    }
    pda.try_borrow_mut()?.fill(0);

    // Hand the (zeroed) PDA over to the delegation program.
    let mut seed_buf = make_seed_buf();
    let filled = fill_seeds(&mut seed_buf, seeds, &bump);
    let signer_seeds = Signer::from(filled);

    // SAFETY: the PDA is program-owned with zeroed data; reassigning to the
    // system program is the documented delegation handoff.
    unsafe { pda.assign(&pinocchio_system::ID) };
    Assign { account: pda, owner: &DELEGATION_PROGRAM_ID }
        .invoke_signed(core::array::from_ref(&signer_seeds))?;

    cpi_delegate(
        payer,
        pda,
        owner_program,
        buffer,
        delegation_record,
        delegation_metadata,
        system_program,
        DelegateAccountArgs {
            commit_frequency_ms: config.commit_frequency_ms,
            seeds,
            validator: config.validator,
        },
        signer_seeds,
    )?;

    close_pda_acc(payer, buffer)
}

/// Create / grow the delegate buffer PDA for a large account, 10KB per call.
/// The rent deposited here returns to the payer when `Delegate` closes the
/// buffer.
pub fn process_delegate_prep(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = GrowArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;
    let mut seed_buf: [&'static [u8]; 2] = [&[], &[]];
    let seeds = seeds_for(args.which as usize, &mut seed_buf)?;

    require_authority(accounts)?;
    let [payer, pda, buffer, _system, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    let (expected, _) = Address::find_program_address(seeds, &crate::ID);
    if pda.address() != &expected {
        return Err(ProgramError::InvalidSeeds);
    }
    let pda_key = *pda.address().as_array();
    let (buffer_expected, buffer_bump) =
        Address::find_program_address(&[BUFFER, &pda_key], &crate::ID);
    if buffer.address() != &buffer_expected {
        return Err(ProgramError::InvalidSeeds);
    }

    let target = pda.data_len();
    if buffer.is_data_empty() && !buffer.owned_by(&crate::ID) {
        let initial = target.min(MAX_GROW);
        let bump_seed = [buffer_bump];
        let signer_seeds =
            [Seed::from(BUFFER), Seed::from(pda_key.as_ref()), Seed::from(&bump_seed)];
        return CreateAccount {
            from: payer,
            to: buffer,
            lamports: rent_exempt_minimum(initial)?,
            space: initial as u64,
            owner: &crate::ID,
        }
        .invoke_signed(&[Signer::from(&signer_seeds)]);
    }

    let current = buffer.data_len();
    if current >= target {
        return Ok(()); // idempotent
    }
    let new_len = target.min(current + MAX_GROW);
    let required = rent_exempt_minimum(new_len)?;
    if required > buffer.lamports() {
        Transfer { from: payer, to: buffer, lamports: required - buffer.lamports() }.invoke()?;
    }
    buffer.resize(new_len)
}
