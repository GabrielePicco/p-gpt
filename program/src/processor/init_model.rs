use gpt_core::{Rng, ONE};
use p_gpt_interface::instruction::InitModelArgs;
use p_gpt_interface::state::{DocsHeader, GenLogHeader, ModelHeader, MODEL_MAGIC, STATE_VERSION};
use p_gpt_interface::{
    seeds, COMMUNITY_ACCOUNT_LEN, COMMUNITY_CAPACITY, DATASET_ACCOUNT_LEN, DATASET_CAPACITY,
    GENLOG_ACCOUNT_LEN, GENLOG_CAPACITY, MODEL_ACCOUNT_LEN, OPTIMIZER_ACCOUNT_LEN,
    SCRATCH_ACCOUNT_LEN,
};
use pinocchio::cpi::{Seed, Signer};
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult};
use pinocchio_system::instructions::CreateAccount;

use super::shared::{cast_mut, rent_exempt_minimum};

/// Create every program account and write the initial headers.
///
/// Weights are *not* initialized here (that is `InitWeights`, chunked); the
/// header records the PRNG seed so the whole model is derivable from it.
///
/// The first initializer becomes the authority for the singleton PDAs, so
/// deploy and initialize atomically (bundle the deploy and this instruction,
/// or initialize in the deployment transaction flow) — a front-runner could
/// otherwise claim the model.
pub fn process_init_model(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = InitModelArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;

    let [payer, model, optimizer, scratch, dataset, community, genlog, _system] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };

    if !payer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }

    let plan: [(&AccountView, &[u8], usize); 6] = [
        (model, seeds::MODEL, MODEL_ACCOUNT_LEN),
        (optimizer, seeds::OPTIMIZER, OPTIMIZER_ACCOUNT_LEN),
        (scratch, seeds::SCRATCH, SCRATCH_ACCOUNT_LEN),
        (dataset, seeds::DATASET, DATASET_ACCOUNT_LEN),
        (community, seeds::COMMUNITY, COMMUNITY_ACCOUNT_LEN),
        (genlog, seeds::GENLOG, GENLOG_ACCOUNT_LEN),
    ];

    let mut bumps = [0u8; 6];
    for (i, (account, seed, space)) in plan.iter().enumerate() {
        let (expected, bump) = Address::find_program_address(&[seed], &crate::ID);
        if account.address() != &expected {
            return Err(ProgramError::InvalidSeeds);
        }
        if !account.is_data_empty() {
            return Err(ProgramError::AccountAlreadyInitialized);
        }
        bumps[i] = bump;

        // The runtime caps data growth at 10,240 bytes per instruction, so
        // large accounts start small and reach `space` via repeated `Grow`.
        let initial = (*space).min(super::grow::MAX_GROW);
        let bump_seed = [bump];
        let signer_seeds = [Seed::from(*seed), Seed::from(&bump_seed)];
        CreateAccount {
            from: payer,
            to: account,
            lamports: rent_exempt_minimum(initial)?,
            space: initial as u64,
            owner: &crate::ID,
        }
        .invoke_signed(&[Signer::from(&signer_seeds)])?;
    }

    // Model header (the header fits in the initial 10KB; weights arrive
    // after `Grow` + `InitWeights`).
    {
        let mut data = model.try_borrow_mut()?;
        let header: &mut ModelHeader = cast_mut(&mut data)?;
        *header = ModelHeader {
            magic: MODEL_MAGIC,
            version: STATE_VERSION,
            bumps,
            flags: 0,
            _pad: [0; 4],
            authority: *payer.address().as_array(),
            seed: args.seed,
            rng_state: Rng::new(args.seed).0,
            init_cursor: 0,
            step: 0,
            pow_beta1: ONE,
            pow_beta2: ONE,
            loss_ema: 0,
            last_loss: 0,
            gen_count: 0,
            ring_pos: 0,
            loss_ring: [0; p_gpt_interface::LOSS_RING_LEN],
            phase: 0,
            phase_cursor: 0,
            doc_tokens_len: 0,
            _pad2: [0; 5],
            doc: [0; 18],
            _pad3: [0; 6],
            pending_loss: 0,
            _reserved: [0; 80],
        };
    }

    // Docs headers.
    {
        let mut data = dataset.try_borrow_mut()?;
        let header: &mut DocsHeader = cast_mut(&mut data)?;
        *header =
            DocsHeader { magic: MODEL_MAGIC, _pad: [0; 4], capacity: DATASET_CAPACITY, count: 0 };
    }
    {
        let mut data = community.try_borrow_mut()?;
        let header: &mut DocsHeader = cast_mut(&mut data)?;
        *header =
            DocsHeader { magic: MODEL_MAGIC, _pad: [0; 4], capacity: COMMUNITY_CAPACITY, count: 0 };
    }

    // Generation log header.
    {
        let mut data = genlog.try_borrow_mut()?;
        let header: &mut GenLogHeader = cast_mut(&mut data)?;
        *header = GenLogHeader {
            magic: MODEL_MAGIC,
            _pad: [0; 4],
            capacity: GENLOG_CAPACITY,
            total: 0,
            _reserved: [0; 8],
        };
    }

    Ok(())
}
