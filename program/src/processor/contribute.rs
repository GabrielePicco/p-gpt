use gpt_core::BOS;
use p_gpt_interface::state::MODEL_MAGIC;
use p_gpt_interface::{seeds, COMMUNITY_CAPACITY, DOC_RECORD_LEN, MAX_NAME_LEN};
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult};

use super::shared::{check_owned, docs_parts_mut};

/// Contribute a name to the community dataset: your name becomes training
/// data on the next community step. Instruction data: token ids (0..26).
pub fn process_contribute(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let [contributor, community] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    if !contributor.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    check_owned(community)?;
    let (expected, _) = Address::find_program_address(&[seeds::COMMUNITY], &crate::ID);
    if community.address() != &expected {
        return Err(ProgramError::InvalidSeeds);
    }

    if data.is_empty() || data.len() > MAX_NAME_LEN || data.iter().any(|t| *t >= BOS) {
        return Err(ProgramError::InvalidInstructionData);
    }

    let mut community_data = community.try_borrow_mut()?;
    let (docs, records) = docs_parts_mut(&mut community_data)?;
    if docs.magic != MODEL_MAGIC
        || docs.capacity != COMMUNITY_CAPACITY
        || docs.count > docs.capacity
        || records.len() < docs.capacity as usize * DOC_RECORD_LEN
    {
        return Err(ProgramError::InvalidAccountData);
    }
    if docs.count >= docs.capacity {
        return Err(ProgramError::AccountDataTooSmall);
    }

    let offset = docs.count as usize * DOC_RECORD_LEN;
    let slot = &mut records[offset..offset + DOC_RECORD_LEN];
    slot.fill(0);
    slot[0] = data.len() as u8;
    slot[1..=data.len()].copy_from_slice(data);
    docs.count += 1;

    Ok(())
}
