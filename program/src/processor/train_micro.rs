use gpt_core::{
    adam_range, backward_position, fmul, forward_pos_loss, zero_grads, AdamParams, Moments,
    Scratch, BLOCK, BOS, N_PARAMS,
};
use p_gpt_interface::state::{
    ADAM_CHUNK, FLAG_WEIGHTS_READY, PHASE_ADAM, PHASE_BACKWARD, PHASE_FORWARD, PHASE_PICK,
};
use p_gpt_interface::{bump_ix, seeds, COMMUNITY_EVERY, DOC_STRIDE, LOSS_EMA_SHIFT, LOSS_RING_LEN};
use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use super::shared::{
    cast_mut, check_model, check_owned, check_pda, doc_record, docs_parts, model_parts,
};

/// One micro-op of the split training path. Each op fits the ~200K CU budget
/// the ER crank gives a tick (and trivially fits the base layer's 1.4M CU),
/// so a full SGD step is a short sequence of permissionless transactions:
///
///   pick doc -> forward (n txs) -> backward (n txs) -> Adam (17 txs)
///
/// The model header carries the phase state machine; the arithmetic is
/// identical to the fused `TrainStep` — the parity test asserts both paths
/// produce bit-identical weights.
pub fn process_train_micro(accounts: &mut [AccountView], _data: &[u8]) -> ProgramResult {
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
    check_pda(optimizer, seeds::OPTIMIZER, header.bumps[bump_ix::OPTIMIZER])?;
    check_pda(scratch, seeds::SCRATCH, header.bumps[bump_ix::SCRATCH])?;
    check_pda(dataset, seeds::DATASET, header.bumps[bump_ix::DATASET])?;
    check_pda(community, seeds::COMMUNITY, header.bumps[bump_ix::COMMUNITY])?;

    let mut scratch_data = scratch.try_borrow_mut()?;
    let workspace: &mut Scratch = cast_mut(&mut scratch_data)?;

    // Positions in the in-flight doc (set by PHASE_PICK).
    let positions = |tokens_len: usize| BLOCK.min(tokens_len.saturating_sub(1));

    match header.phase {
        PHASE_PICK => {
            let dataset_data = dataset.try_borrow()?;
            let (docs, doc_records) = docs_parts(&dataset_data)?;
            let community_data = community.try_borrow()?;
            let (extra, extra_records) = docs_parts(&community_data)?;
            if docs.count == 0 {
                return Err(ProgramError::UninitializedAccount);
            }

            let step = header.step;
            let record = if extra.count > 0 && step % COMMUNITY_EVERY == COMMUNITY_EVERY - 1 {
                doc_record(extra_records, ((step / COMMUNITY_EVERY) % extra.count) as usize)
            } else {
                doc_record(doc_records, ((step.wrapping_mul(DOC_STRIDE)) % docs.count) as usize)
            };
            let len = record[0] as usize;
            if len == 0 || len >= BLOCK || record[1..=len].iter().any(|t| *t >= BOS) {
                return Err(ProgramError::InvalidAccountData);
            }

            header.doc = [BOS; 18];
            header.doc[1..=len].copy_from_slice(&record[1..=len]);
            header.doc_tokens_len = (len + 2) as u8;
            header.pending_loss = 0;

            zero_grads(workspace);
            header.phase = PHASE_FORWARD;
            header.phase_cursor = 0;
        }
        PHASE_FORWARD => {
            let n = positions(header.doc_tokens_len as usize);
            let t = header.phase_cursor as usize;
            if n == 0 || t >= n {
                return Err(ProgramError::InvalidAccountData);
            }
            header.pending_loss -=
                forward_pos_loss(weights, workspace, t, header.doc[t], header.doc[t + 1]);
            if t + 1 == n {
                // Mean over positions; backward starts from the top.
                header.pending_loss /= n as i64;
                header.phase = PHASE_BACKWARD;
                header.phase_cursor = (n - 1) as u8;
            } else {
                header.phase_cursor += 1;
            }
        }
        PHASE_BACKWARD => {
            let n = positions(header.doc_tokens_len as usize);
            let t = header.phase_cursor as usize;
            if n == 0 || t >= n {
                return Err(ProgramError::InvalidAccountData);
            }
            backward_position(weights, workspace, t, header.doc[t], header.doc[t + 1], n);
            if t == 0 {
                // Advance the bias-correction terms exactly once per step,
                // before the Adam chunks consume them.
                header.pow_beta1 = fmul(header.pow_beta1, gpt_core::adam::BETA1);
                header.pow_beta2 = fmul(header.pow_beta2, gpt_core::adam::BETA2);
                header.phase = PHASE_ADAM;
                header.phase_cursor = 0;
            } else {
                header.phase_cursor -= 1;
            }
        }
        PHASE_ADAM => {
            let mut optimizer_data = optimizer.try_borrow_mut()?;
            let moments: &mut Moments = cast_mut(&mut optimizer_data)?;

            let chunk = header.phase_cursor as usize;
            let start = chunk * ADAM_CHUNK;
            let end = N_PARAMS.min(start + ADAM_CHUNK);
            if start >= N_PARAMS {
                return Err(ProgramError::InvalidAccountData);
            }
            let params = AdamParams {
                lr_t: gpt_core::lr_at_step(header.step),
                pow_beta1: header.pow_beta1,
                pow_beta2: header.pow_beta2,
            };
            adam_range(
                weights.as_flat_mut(),
                moments,
                workspace.grads.as_flat(),
                start,
                end,
                params,
            );

            if end == N_PARAMS {
                // The step lands.
                let step = header.step;
                let loss = header.pending_loss;
                header.step += 1;
                header.last_loss = loss;
                header.loss_ema = if step == 0 {
                    loss
                } else {
                    header.loss_ema + ((loss - header.loss_ema) >> LOSS_EMA_SHIFT)
                };
                header.loss_ring[(header.ring_pos % LOSS_RING_LEN as u64) as usize] = loss;
                header.ring_pos += 1;
                header.phase = PHASE_PICK;
                header.phase_cursor = 0;

                #[cfg(feature = "logging")]
                pinocchio_log::log!("p-gpt: step {} loss_q32 {}", header.step, header.last_loss);
            } else {
                header.phase_cursor += 1;
            }
        }
        _ => return Err(ProgramError::InvalidAccountData),
    }

    Ok(())
}
