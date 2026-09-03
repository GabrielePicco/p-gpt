use ephemeral_rollups_pinocchio::consts::EXTERNAL_UNDELEGATE_DISCRIMINATOR;
use pinocchio::{error::ProgramError, AccountView, Address, ProgramResult};

use crate::processor::*;
use p_gpt_interface::instruction::ix;

#[cfg(target_os = "solana")]
pinocchio::program_entrypoint!(process_instruction);
// Do not allocate memory: every buffer in this program is stack- or
// account-backed.
#[cfg(target_os = "solana")]
pinocchio::no_allocator!();
// All dependencies are no_std, so declare a rust runtime panic handler.
#[cfg(target_os = "solana")]
pinocchio::nostd_panic_handler!();

/// Process an instruction.
///
/// The hot instructions (`TrainStep`, fired by the crank every tick, and
/// `Generate`) are matched first.
#[inline(always)]
pub fn process_instruction(
    _program_id: &Address,
    accounts: &mut [AccountView],
    instruction_data: &[u8],
) -> ProgramResult {
    // Undelegation callback CPI'd by the delegation program.
    if instruction_data.len() >= 8 && instruction_data[..8] == EXTERNAL_UNDELEGATE_DISCRIMINATOR {
        return process_undelegate_callback(accounts, &instruction_data[8..]);
    }

    let (discriminator, data) =
        instruction_data.split_first().ok_or(ProgramError::InvalidInstructionData)?;

    match *discriminator {
        // 4 - TrainStep
        ix::TRAIN_STEP => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: TrainStep");

            process_train_step(accounts, data)
        }
        // 5 - TrainMicro
        ix::TRAIN_MICRO => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: TrainMicro");

            process_train_micro(accounts, data)
        }
        // 10 - Generate
        ix::GENERATE => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: Generate");

            process_generate(accounts, data)
        }
        // 0 - InitModel
        ix::INIT_MODEL => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: InitModel");

            process_init_model(accounts, data)
        }
        // 1 - InitWeights
        ix::INIT_WEIGHTS => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: InitWeights");

            process_init_weights(accounts, data)
        }
        // 2 - LoadDocs
        ix::LOAD_DOCS => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: LoadDocs");

            process_load_docs(accounts, data)
        }
        // 3 - Delegate
        ix::DELEGATE => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: Delegate");

            process_delegate(accounts, data)
        }
        // 7 - ScheduleTraining
        ix::SCHEDULE_TRAINING => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: ScheduleTraining");

            process_schedule_training(accounts, data)
        }
        // 8 - Checkpoint
        ix::CHECKPOINT => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: Checkpoint");

            process_checkpoint(accounts, data)
        }
        // 9 - Undelegate
        ix::UNDELEGATE => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: Undelegate");

            process_undelegate(accounts, data)
        }
        // 14 - InitShards
        ix::INIT_SHARDS => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: InitShards");

            process_init_shards(accounts, data)
        }
        // 13 - DelegatePrep
        ix::DELEGATE_PREP => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: DelegatePrep");

            process_delegate_prep(accounts, data)
        }
        // 12 - Grow
        ix::GROW => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: Grow");

            process_grow(accounts, data)
        }
        // 11 - Contribute
        ix::CONTRIBUTE => {
            #[cfg(feature = "logging")]
            pinocchio_log::log!("Instruction: Contribute");

            process_contribute(accounts, data)
        }
        _ => Err(ProgramError::InvalidInstructionData),
    }
}
