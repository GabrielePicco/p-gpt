//! Zero-copy account access and PDA validation shared by all processors.

use core::mem::{align_of, size_of};

use gpt_core::Weights;
use p_gpt_interface::state::{
    DocsHeader, GenLogHeader, GenRecord, ModelHeader, MODEL_MAGIC, STATE_VERSION,
};
use p_gpt_interface::{seeds, DOC_RECORD_LEN};
use pinocchio::{error::ProgramError, AccountView, Address};

/// Rent-exempt minimum for `len` bytes.
///
/// pinocchio's `Rent` reads the sysvar's first u64 (`lamports_per_byte_year`)
/// and ignores the 2.0 exemption threshold, which under-funds accounts by
/// half — mollusk doesn't enforce the post-execution rent check, but real
/// validators do. Scale by the (universal) threshold here.
#[inline(always)]
pub fn rent_exempt_minimum(len: usize) -> Result<u64, ProgramError> {
    use pinocchio::sysvars::{rent::Rent, Sysvar};
    Ok(Rent::get()?.try_minimum_balance(len)? * 2)
}

/// Reinterpret account bytes as a `#[repr(C)]` state struct.
#[inline(always)]
pub fn cast_mut<T>(data: &mut [u8]) -> Result<&mut T, ProgramError> {
    if data.len() < size_of::<T>() || (data.as_ptr() as usize) % align_of::<T>() != 0 {
        return Err(ProgramError::InvalidAccountData);
    }
    // SAFETY: length and alignment checked; T is plain integers.
    Ok(unsafe { &mut *(data.as_mut_ptr() as *mut T) })
}

#[inline(always)]
pub fn cast_ref<T>(data: &[u8]) -> Result<&T, ProgramError> {
    if data.len() < size_of::<T>() || (data.as_ptr() as usize) % align_of::<T>() != 0 {
        return Err(ProgramError::InvalidAccountData);
    }
    // SAFETY: as above.
    Ok(unsafe { &*(data.as_ptr() as *const T) })
}

/// Split the model account into its header and weights.
#[inline(always)]
pub fn model_parts(data: &mut [u8]) -> Result<(&mut ModelHeader, &mut Weights), ProgramError> {
    if data.len() < p_gpt_interface::MODEL_ACCOUNT_LEN {
        return Err(ProgramError::InvalidAccountData);
    }
    let (header, weights) = data.split_at_mut(size_of::<ModelHeader>());
    Ok((cast_mut(header)?, cast_mut(weights)?))
}

/// Validate the model header invariants.
#[inline(always)]
pub fn check_model(header: &ModelHeader) -> Result<(), ProgramError> {
    if header.magic != MODEL_MAGIC || header.version != STATE_VERSION {
        return Err(ProgramError::InvalidAccountData);
    }
    Ok(())
}

/// Split a docs account (dataset / community) into header and records.
#[inline(always)]
pub fn docs_parts(data: &[u8]) -> Result<(&DocsHeader, &[u8]), ProgramError> {
    if data.len() < size_of::<DocsHeader>() {
        return Err(ProgramError::InvalidAccountData);
    }
    let (header, records) = data.split_at(size_of::<DocsHeader>());
    let header: &DocsHeader = cast_ref(header)?;
    if records.len() < header.capacity as usize * DOC_RECORD_LEN {
        return Err(ProgramError::InvalidAccountData);
    }
    Ok((header, records))
}

#[inline(always)]
pub fn docs_parts_mut(data: &mut [u8]) -> Result<(&mut DocsHeader, &mut [u8]), ProgramError> {
    if data.len() < size_of::<DocsHeader>() {
        return Err(ProgramError::InvalidAccountData);
    }
    let (header, records) = data.split_at_mut(size_of::<DocsHeader>());
    Ok((cast_mut(header)?, records))
}

/// Fixed 16-byte doc record at `index`.
#[inline(always)]
pub fn doc_record(records: &[u8], index: usize) -> &[u8; DOC_RECORD_LEN] {
    records[index * DOC_RECORD_LEN..(index + 1) * DOC_RECORD_LEN].try_into().unwrap()
}

/// Split the generation log into header and record ring.
#[inline(always)]
pub fn genlog_parts(
    data: &mut [u8],
) -> Result<(&mut GenLogHeader, &mut [GenRecord]), ProgramError> {
    if data.len() < size_of::<GenLogHeader>() {
        return Err(ProgramError::InvalidAccountData);
    }
    let (header, records) = data.split_at_mut(size_of::<GenLogHeader>());
    let header: &mut GenLogHeader = cast_mut(header)?;
    let n = header.capacity as usize;
    if records.len() < n * size_of::<GenRecord>()
        || (records.as_ptr() as usize) % align_of::<GenRecord>() != 0
    {
        return Err(ProgramError::InvalidAccountData);
    }
    // SAFETY: length and alignment checked; GenRecord is plain integers.
    let records = unsafe { core::slice::from_raw_parts_mut(records.as_mut_ptr().cast(), n) };
    Ok((header, records))
}

/// Shard index byte constants so shard seeds have 'static lifetime.
pub static SHARD_INDEX: [[u8; 1]; p_gpt_interface::SHARD_COUNT] = [[0], [1], [2], [3]];

/// The PDA seeds for each `bump_ix` slot, built at runtime into `buf`
/// (shards use two seed parts; nested static seed tables miscompile on SBF).
pub fn seeds_for<'a>(
    which: usize,
    buf: &'a mut [&'static [u8]; 2],
) -> Result<&'a [&'static [u8]], ProgramError> {
    let single: [&'static [u8]; 6] = [
        seeds::MODEL,
        seeds::OPTIMIZER,
        seeds::SCRATCH,
        seeds::DATASET,
        seeds::COMMUNITY,
        seeds::GENLOG,
    ];
    if which < single.len() {
        buf[0] = single[which];
        Ok(&buf[..1])
    } else if which - single.len() < p_gpt_interface::SHARD_COUNT {
        buf[0] = seeds::SHARD;
        buf[1] = &SHARD_INDEX[which - single.len()];
        Ok(&buf[..2])
    } else {
        Err(ProgramError::InvalidInstructionData)
    }
}

/// Check that `account` is the PDA for `seed` with the stored `bump`.
#[inline(always)]
pub fn check_pda(account: &AccountView, seed: &[u8], bump: u8) -> Result<(), ProgramError> {
    let expected = Address::create_program_address(&[seed, &[bump]], &crate::ID)
        .map_err(|_| ProgramError::InvalidSeeds)?;
    if account.address() != &expected {
        return Err(ProgramError::InvalidSeeds);
    }
    Ok(())
}

/// Program-owned check: true on base layer before delegation and inside the
/// ER for delegated clones (where the owner is restored to this program).
#[inline(always)]
pub fn check_owned(account: &AccountView) -> Result<(), ProgramError> {
    if !account.owned_by(&crate::ID) {
        return Err(ProgramError::InvalidAccountOwner);
    }
    Ok(())
}
