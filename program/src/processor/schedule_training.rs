use ephemeral_rollups_pinocchio::crank::{CrankInstruction, ScheduleCrankCpi};
use p_gpt_interface::instruction::{ix, ScheduleArgs};
use pinocchio::instruction::InstructionAccount;
use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use super::shared::check_owned;

/// Schedule the perpetual training crank on the ephemeral rollup: the ER
/// itself fires `TrainStep` every `interval_ms`, forever if `iterations`
/// is `u64::MAX`. No server anywhere.
pub fn process_schedule_training(accounts: &mut [AccountView], data: &[u8]) -> ProgramResult {
    let args = ScheduleArgs::parse(data).ok_or(ProgramError::InvalidInstructionData)?;

    let [payer, magic_program, model, optimizer, scratch, dataset, community] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    if !payer.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }
    check_owned(model)?;

    let metas = [
        InstructionAccount::writable(model.address()),
        InstructionAccount::writable(optimizer.address()),
        InstructionAccount::writable(scratch.address()),
        InstructionAccount::readonly(dataset.address()),
        InstructionAccount::readonly(community.address()),
    ];
    // Each tick fires one micro-op (pick / forward position / backward
    // position / Adam chunk). Tick transactions run the task instructions via
    // the magic program with the runtime's default budget (~200K CU per
    // instruction), which every micro-op fits. A full SGD step lands every
    // ~35 ticks. `steps_per_tick` is kept for future raised-limit runtimes.
    let _ = args.steps_per_tick;
    let ix_data = [ix::TRAIN_MICRO];
    let crank_ix = CrankInstruction::new(crate::ID, &metas, &ix_data);

    let instruction_accounts =
        [model.clone(), optimizer.clone(), scratch.clone(), dataset.clone(), community.clone()];

    let mut buf = [0u8; 1024];
    ScheduleCrankCpi::builder(payer.clone(), magic_program.clone())
        .task_id(args.task_id as i64)
        .execution_interval_millis(args.interval_ms as i64)
        .iterations(args.iterations as i64)
        .instruction_accounts(&instruction_accounts)
        .instructions(&[crank_ix])
        .build_and_invoke::<8>(&mut buf)
}
