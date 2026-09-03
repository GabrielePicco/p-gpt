//! Instruction discriminators and argument layouts.
//!
//! Encoding: 1-byte discriminator followed by little-endian fixed-width
//! fields. Instruction data has no alignment guarantee, so args are decoded
//! field-by-field, never reinterpreted.

/// Instruction discriminators.
pub mod ix {
    /// Create and initialize all program accounts.
    /// Accounts: `[payer(s,w), model(w), optimizer(w), scratch(w), dataset(w),
    /// community(w), genlog(w), system]` — args: `InitModelArgs`.
    pub const INIT_MODEL: u8 = 0;
    /// PRNG-initialize the next chunk of weights.
    /// Accounts: `[model(w)]` — args: `InitWeightsArgs`.
    pub const INIT_WEIGHTS: u8 = 1;
    /// Append 16-byte doc records (packed in the instruction data).
    /// Accounts: `[authority(s), model, dataset(w)]`.
    pub const LOAD_DOCS: u8 = 2;
    /// Delegate one PDA to the ephemeral rollup. The payer must be the model
    /// authority.
    /// Accounts: `[payer(s,w), pda(w), owner_program, buffer(w),
    /// delegation_record(w), delegation_metadata(w), system,
    /// delegation_program, model]` — args: `DelegateArgs`.
    pub const DELEGATE: u8 = 3;
    /// Run fused training steps (forward + backward + Adam). Permissionless.
    /// Accounts: `[model(w), optimizer(w), scratch(w), dataset, community]` —
    /// args: `TrainStepArgs`.
    pub const TRAIN_STEP: u8 = 4;
    /// One micro-op of the split training path (fits in 1.4M CU): forward,
    /// a few backward positions, or an Adam chunk, driven by the phase state
    /// machine in the model header. Permissionless; same accounts as
    /// `TRAIN_STEP`; no args.
    pub const TRAIN_MICRO: u8 = 5;
    /// Schedule the perpetual training crank (ER only).
    /// Accounts: `[payer(s,w), magic_program, model(w), optimizer(w),
    /// scratch(w), dataset, community]` — args: `ScheduleArgs`.
    pub const SCHEDULE_TRAINING: u8 = 7;
    /// Sync the model image into the checkpoint shards and commit shards +
    /// genlog to the base layer (ER only). Rejected while Adam chunks are in
    /// flight (the weights would be torn). Permissionless.
    /// Accounts: `[payer(s,w), magic_context(w), magic_program, model,
    /// genlog(w), shard0(w)..shard3(w)]`.
    pub const CHECKPOINT: u8 = 8;
    /// Commit and undelegate the committable accounts (ER only). The payer
    /// must be the model authority; the large working accounts stay
    /// delegated (see the shard design notes in the README).
    /// Accounts: `[payer(s,w), magic_context(w), magic_program, model,
    /// community(w), genlog(w), shard0(w)..shard3(w)]`.
    pub const UNDELEGATE: u8 = 9;
    /// Sample a name from the model.
    /// Accounts: `[model(w), scratch(w), genlog(w)]` — args: `GenerateArgs`.
    pub const GENERATE: u8 = 10;
    /// Contribute a name to the community dataset.
    /// Accounts: `[contributor(s), community(w)]` — args: token bytes.
    pub const CONTRIBUTE: u8 = 11;
    /// Grow a program account toward its target size (runtime caps data
    /// growth at 10,240 bytes per instruction, so large accounts are created
    /// small and grown by repeated calls).
    /// Accounts: `[payer(s,w), pda(w), system]` — args: `GrowArgs`.
    pub const GROW: u8 = 12;
    /// Create / grow the delegate buffer for a large account before
    /// `DELEGATE` (10,240 bytes per call; rent returns on delegation). The
    /// payer must be the model authority.
    /// Accounts: `[payer(s,w), pda, buffer(w), system, model]` —
    /// args: `GrowArgs`.
    pub const DELEGATE_PREP: u8 = 13;
    /// Create the checkpoint shard accounts.
    /// Accounts: `[payer(s,w), shard0(w)..shard3(w), system]`.
    pub const INIT_SHARDS: u8 = 14;
}

/// Little-endian field reader over instruction data.
pub struct Reader<'a>(pub &'a [u8]);

impl<'a> Reader<'a> {
    pub fn u8(&mut self) -> Option<u8> {
        let (v, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(*v)
    }

    pub fn u32(&mut self) -> Option<u32> {
        let (v, rest) = self.0.split_first_chunk::<4>()?;
        self.0 = rest;
        Some(u32::from_le_bytes(*v))
    }

    pub fn u64(&mut self) -> Option<u64> {
        let (v, rest) = self.0.split_first_chunk::<8>()?;
        self.0 = rest;
        Some(u64::from_le_bytes(*v))
    }

    pub fn i64(&mut self) -> Option<i64> {
        self.u64().map(|v| v as i64)
    }

    pub fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (v, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(v)
    }

    pub fn rest(self) -> &'a [u8] {
        self.0
    }
}

pub struct InitModelArgs {
    pub seed: u64,
}

impl InitModelArgs {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let mut r = Reader(data);
        Some(Self { seed: r.u64()? })
    }
}

pub struct InitWeightsArgs {
    /// Max parameters to initialize in this call.
    pub count: u32,
}

impl InitWeightsArgs {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let mut r = Reader(data);
        Some(Self { count: r.u32()? })
    }
}

pub struct DelegateArgs {
    /// Which PDA to delegate: an index into `crate::bump_ix`.
    pub which: u8,
    /// Commit frequency hint for the ER, in ms.
    pub commit_frequency_ms: u32,
    /// The ER validator to delegate to; None lets the delegation program
    /// pick its default, which is only right on public clusters.
    pub validator: Option<[u8; 32]>,
}

impl DelegateArgs {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let mut r = Reader(data);
        let which = r.u8()?;
        let commit_frequency_ms = r.u32()?;
        let validator = match r.bytes(32) {
            Some(b) => Some(b.try_into().ok()?),
            None => None,
        };
        Some(Self { which, commit_frequency_ms, validator })
    }
}

pub struct TrainStepArgs {
    /// Number of fused steps to run in this transaction.
    pub count: u8,
}

impl TrainStepArgs {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let mut r = Reader(data);
        Some(Self { count: r.u8()? })
    }
}

pub struct ScheduleArgs {
    pub task_id: u64,
    pub interval_ms: u64,
    pub iterations: u64,
    pub steps_per_tick: u8,
}

impl ScheduleArgs {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let mut r = Reader(data);
        Some(Self {
            task_id: r.u64()?,
            interval_ms: r.u64()?,
            iterations: r.u64()?,
            steps_per_tick: r.u8()?,
        })
    }
}

pub struct GrowArgs {
    /// Which PDA to grow: an index into `crate::bump_ix`.
    pub which: u8,
}

impl GrowArgs {
    pub fn parse(data: &[u8]) -> Option<Self> {
        let mut r = Reader(data);
        Some(Self { which: r.u8()? })
    }
}

pub struct GenerateArgs<'a> {
    /// Sampling temperature in Q32.32, in (0, 1].
    pub temperature: i64,
    /// Client entropy, mixed with the slot and generation counter.
    pub seed: u64,
    /// Prefix token ids (0..26).
    pub prefix: &'a [u8],
}

impl<'a> GenerateArgs<'a> {
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        let mut r = Reader(data);
        let temperature = r.i64()?;
        let seed = r.u64()?;
        let prefix_len = r.u8()? as usize;
        let prefix = r.bytes(prefix_len)?;
        Some(Self { temperature, seed, prefix })
    }
}
