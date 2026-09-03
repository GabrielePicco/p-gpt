use gpt_core::{train_doc, Moments, Scratch, BLOCK, BOS};
use p_gpt_interface::instruction::TrainStepArgs;
use p_gpt_interface::state::FLAG_WEIGHTS_READY;
use p_gpt_interface::{bump_ix, seeds, COMMUNITY_EVERY, DOC_STRIDE, LOSS_EMA_SHIFT, LOSS_RING_LEN};
use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use super::shared::{
    cast_mut, check_model, check_owned, check_pda, doc_record, docs_parts, model_parts,
};

/// The perpetual heartbeat: run `count` fused training steps (forward,
/// backward, Adam) on the next documents in the deterministic schedule.
///
/// Permissionless — on the ER the crank fires it every tick; on the base
/// layer anyone may push a step. The document order is a pure function of the
/// step counter, so replaying the transaction history reproduces the weights.
pub fn process_train_step(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = TrainStepArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;
    let count = args.count.clamp(1, 16);

    let [model, optimizer, scratch, dataset, community] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    check_owned(model)?;
    check_owned(optimizer)?;
    check_owned(scratch)?;
    check_owned(dataset)?;
    check_owned(community)?;

    let mut model_data = model.try_borrow_mut()?;
    let (header, weights) = model_parts(&mut model_data)?;
    check_model(header)?;
    if header.flags & FLAG_WEIGHTS_READY == 0 {
        return Err(ProgramError::UninitializedAccount);
    }
    // A fused step over an in-flight split step would mix its activations and
    // gradients into the micro state machine and diverge the model.
    if header.phase != p_gpt_interface::state::PHASE_PICK {
        return Err(ProgramError::InvalidAccountData);
    }
    check_pda(optimizer, seeds::OPTIMIZER, header.bumps[bump_ix::OPTIMIZER])?;
    check_pda(scratch, seeds::SCRATCH, header.bumps[bump_ix::SCRATCH])?;
    check_pda(dataset, seeds::DATASET, header.bumps[bump_ix::DATASET])?;
    check_pda(community, seeds::COMMUNITY, header.bumps[bump_ix::COMMUNITY])?;

    let mut optimizer_data = optimizer.try_borrow_mut()?;
    let moments: &mut Moments = cast_mut(&mut optimizer_data)?;
    let mut scratch_data = scratch.try_borrow_mut()?;
    let workspace: &mut Scratch = cast_mut(&mut scratch_data)?;
    let dataset_data = dataset.try_borrow()?;
    let (docs, doc_records) = docs_parts(&dataset_data)?;
    let community_data = community.try_borrow()?;
    let (extra, extra_records) = docs_parts(&community_data)?;

    if docs.count == 0 {
        return Err(ProgramError::UninitializedAccount);
    }

    for _ in 0..count {
        let step = header.step;

        // Deterministic curriculum: a community doc every Nth step (when any
        // exist), otherwise a stride walk over the pseudo-shuffled dataset.
        let record = if extra.count > 0 && step % COMMUNITY_EVERY == COMMUNITY_EVERY - 1 {
            doc_record(extra_records, ((step / COMMUNITY_EVERY) % extra.count) as usize)
        } else {
            doc_record(doc_records, ((step.wrapping_mul(DOC_STRIDE)) % docs.count) as usize)
        };

        let len = record[0] as usize;
        if len == 0 || len >= BLOCK || record[1..=len].iter().any(|t| *t >= BOS) {
            return Err(ProgramError::InvalidAccountData);
        }
        let mut tokens = [BOS; BLOCK + 2];
        tokens[1..=len].copy_from_slice(&record[1..=len]);
        let doc = &tokens[..len + 2];

        let loss = train_doc(
            weights,
            moments,
            workspace,
            doc,
            step,
            &mut header.pow_beta1,
            &mut header.pow_beta2,
        );

        header.step += 1;
        header.last_loss = loss;
        header.loss_ema = if step == 0 {
            loss
        } else {
            header.loss_ema + ((loss - header.loss_ema) >> LOSS_EMA_SHIFT)
        };
        header.loss_ring[(header.ring_pos % LOSS_RING_LEN as u64) as usize] = loss;
        header.ring_pos += 1;
    }

    #[cfg(feature = "logging")]
    pinocchio_log::log!("p-gpt: step {} loss_q32 {}", header.step, header.last_loss);

    Ok(())
}
