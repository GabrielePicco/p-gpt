use gpt_core::BOS;
use p_gpt_interface::{bump_ix, seeds, DOC_RECORD_LEN, MAX_NAME_LEN};
use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use super::shared::{check_model, check_owned, check_pda, docs_parts_mut, model_parts};

/// Append tokenized doc records to the dataset. Authority-gated.
///
/// Instruction data: packed 16-byte records — `[len, token_ids[15]]` with
/// `len` in 1..=15 and token ids < BOS.
pub fn process_load_docs(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let [authority, model, dataset] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };

    if !authority.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    check_owned(model)?;
    check_owned(dataset)?;

    let mut model_data = model.try_borrow_mut()?;
    let (header, _) = model_parts(&mut model_data)?;
    check_model(header)?;
    if authority.address().as_array() != &header.authority {
        return Err(ProgramError::IncorrectAuthority);
    }
    check_pda(dataset, seeds::DATASET, header.bumps[bump_ix::DATASET])?;

    if data.is_empty() || data.len() % DOC_RECORD_LEN != 0 {
        return Err(ProgramError::InvalidInstructionData);
    }

    let mut dataset_data = dataset.try_borrow_mut()?;
    let (docs, records) = docs_parts_mut(&mut dataset_data)?;

    for record in data.chunks_exact(DOC_RECORD_LEN) {
        let len = record[0] as usize;
        if len == 0 || len > MAX_NAME_LEN || record[1..=len].iter().any(|t| *t >= BOS) {
            return Err(ProgramError::InvalidInstructionData);
        }
        if docs.count >= docs.capacity {
            return Err(ProgramError::AccountDataTooSmall);
        }
        let offset = docs.count as usize * DOC_RECORD_LEN;
        let slot = &mut records[offset..offset + DOC_RECORD_LEN];
        slot.fill(0);
        slot[0] = record[0];
        slot[1..=len].copy_from_slice(&record[1..=len]);
        docs.count += 1;
    }
    Ok(())
}
