//! Functional tests against the compiled SBF program via mollusk.
//!
//! Run `cargo build-sbf` first (the test loads target/deploy/p_gpt_program.so),
//! then `cargo test -p p-gpt-program --test functional -- --nocapture` to see
//! per-instruction compute unit numbers.

use gpt_core::{Moments, Rng, Scratch, Weights, N_PARAMS, ONE};
use mollusk_svm::result::ProgramResult;
use mollusk_svm::Mollusk;
use p_gpt_interface::instruction::ix;
use p_gpt_interface::state::{GenLogHeader, GenRecord, ModelHeader, FLAG_WEIGHTS_READY};
use p_gpt_interface::{
    seeds, COMMUNITY_ACCOUNT_LEN, DATASET_ACCOUNT_LEN, DOC_RECORD_LEN, GENLOG_ACCOUNT_LEN,
    MODEL_ACCOUNT_LEN, OPTIMIZER_ACCOUNT_LEN, SCRATCH_ACCOUNT_LEN,
};
use solana_account::Account;
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;

const PROGRAM_ID: Pubkey = Pubkey::new_from_array(p_gpt_interface::PROGRAM_ID);
const SEED: u64 = 42;

struct Harness {
    mollusk: Mollusk,
    accounts: Vec<(Pubkey, Account)>,
    payer: Pubkey,
    model: Pubkey,
    optimizer: Pubkey,
    scratch: Pubkey,
    dataset: Pubkey,
    community: Pubkey,
    genlog: Pubkey,
}

impl Harness {
    fn new() -> Self {
        std::env::set_var("SBF_OUT_DIR", concat!(env!("CARGO_MANIFEST_DIR"), "/../target/deploy"));
        let mut mollusk = Mollusk::new(&PROGRAM_ID, "p_gpt_program");
        // Enough headroom to measure the real cost of a fused train step.
        mollusk.compute_budget.compute_unit_limit = 400_000_000;

        let payer = Pubkey::new_unique();
        let pda = |seed: &[u8]| Pubkey::find_program_address(&[seed], &PROGRAM_ID).0;
        let (model, optimizer, scratch, dataset, community, genlog) = (
            pda(seeds::MODEL),
            pda(seeds::OPTIMIZER),
            pda(seeds::SCRATCH),
            pda(seeds::DATASET),
            pda(seeds::COMMUNITY),
            pda(seeds::GENLOG),
        );

        let system_program = mollusk_svm::program::keyed_account_for_system_program();
        let accounts = vec![
            (payer, Account { lamports: 100_000_000_000, ..Account::default() }),
            (model, Account::default()),
            (optimizer, Account::default()),
            (scratch, Account::default()),
            (dataset, Account::default()),
            (community, Account::default()),
            (genlog, Account::default()),
            system_program,
        ];

        Self { mollusk, accounts, payer, model, optimizer, scratch, dataset, community, genlog }
    }

    /// Execute one instruction, thread resulting account state, return CU.
    fn run(&mut self, name: &str, instruction: Instruction) -> u64 {
        let result = self.mollusk.process_instruction(&instruction, &self.accounts);
        assert!(
            matches!(result.program_result, ProgramResult::Success),
            "{name} failed: {:?}",
            result.program_result
        );
        self.accounts = result.resulting_accounts.clone();
        result.compute_units_consumed
    }

    fn account(&self, key: &Pubkey) -> &Account {
        &self.accounts.iter().find(|(k, _)| k == key).unwrap().1
    }

    fn header(&self) -> ModelHeader {
        let data = &self.account(&self.model).data;
        unsafe { std::ptr::read(data.as_ptr() as *const ModelHeader) }
    }

    fn weights_bytes(&self) -> &[u8] {
        &self.account(&self.model).data[std::mem::size_of::<ModelHeader>()..]
    }

    fn init_model(&mut self) -> u64 {
        let mut data = vec![ix::INIT_MODEL];
        data.extend_from_slice(&SEED.to_le_bytes());
        let metas = vec![
            AccountMeta::new(self.payer, true),
            AccountMeta::new(self.model, false),
            AccountMeta::new(self.optimizer, false),
            AccountMeta::new(self.scratch, false),
            AccountMeta::new(self.dataset, false),
            AccountMeta::new(self.community, false),
            AccountMeta::new(self.genlog, false),
            AccountMeta::new_readonly(
                solana_pubkey::pubkey!("11111111111111111111111111111111"),
                false,
            ),
        ];
        self.run("init_model", Instruction::new_with_bytes(PROGRAM_ID, &data, metas))
    }

    /// Grow every program account to its target size (10KB per call).
    fn grow_all(&mut self) {
        let plan = [
            (0u8, self.model, MODEL_ACCOUNT_LEN),
            (1, self.optimizer, OPTIMIZER_ACCOUNT_LEN),
            (2, self.scratch, SCRATCH_ACCOUNT_LEN),
            (3, self.dataset, DATASET_ACCOUNT_LEN),
            (4, self.community, COMMUNITY_ACCOUNT_LEN),
            (5, self.genlog, GENLOG_ACCOUNT_LEN),
        ];
        for (which, pda, target) in plan {
            while self.account(&pda).data.len() < target {
                let data = vec![ix::GROW, which];
                let metas = vec![
                    AccountMeta::new(self.payer, true),
                    AccountMeta::new(pda, false),
                    AccountMeta::new_readonly(
                        solana_pubkey::pubkey!("11111111111111111111111111111111"),
                        false,
                    ),
                ];
                self.run("grow", Instruction::new_with_bytes(PROGRAM_ID, &data, metas));
            }
        }
    }

    fn init_weights(&mut self, count: u32) -> u64 {
        let mut data = vec![ix::INIT_WEIGHTS];
        data.extend_from_slice(&count.to_le_bytes());
        let metas = vec![AccountMeta::new(self.model, false)];
        self.run("init_weights", Instruction::new_with_bytes(PROGRAM_ID, &data, metas))
    }

    fn load_docs(&mut self, names: &[&str]) -> u64 {
        let mut data = vec![ix::LOAD_DOCS];
        for name in names {
            let mut record = [0u8; DOC_RECORD_LEN];
            record[0] = name.len() as u8;
            for (i, b) in name.bytes().enumerate() {
                record[1 + i] = b - b'a';
            }
            data.extend_from_slice(&record);
        }
        let metas = vec![
            AccountMeta::new_readonly(self.payer, true),
            AccountMeta::new(self.model, false),
            AccountMeta::new(self.dataset, false),
        ];
        self.run("load_docs", Instruction::new_with_bytes(PROGRAM_ID, &data, metas))
    }

    fn train_step(&mut self, count: u8) -> u64 {
        let data = vec![ix::TRAIN_STEP, count];
        let metas = vec![
            AccountMeta::new(self.model, false),
            AccountMeta::new(self.optimizer, false),
            AccountMeta::new(self.scratch, false),
            AccountMeta::new_readonly(self.dataset, false),
            AccountMeta::new_readonly(self.community, false),
        ];
        self.run("train_step", Instruction::new_with_bytes(PROGRAM_ID, &data, metas))
    }

    /// One split-path micro transaction; returns (phase_before, cu).
    fn train_micro(&mut self) -> (u8, u64) {
        let phase = self.header_field_phase();
        let data = vec![ix::TRAIN_MICRO];
        let metas = vec![
            AccountMeta::new(self.model, false),
            AccountMeta::new(self.optimizer, false),
            AccountMeta::new(self.scratch, false),
            AccountMeta::new_readonly(self.dataset, false),
            AccountMeta::new_readonly(self.community, false),
        ];
        let cu = self.run("train_micro", Instruction::new_with_bytes(PROGRAM_ID, &data, metas));
        (phase, cu)
    }

    fn header_field_phase(&self) -> u8 {
        // phase lives right after the loss ring: 128 + 256*8.
        self.account(&self.model).data[128 + 256 * 8]
    }

    fn generate(&mut self, prefix: &[u8], seed: u64) -> u64 {
        let mut data = vec![ix::GENERATE];
        data.extend_from_slice(&(ONE / 2).to_le_bytes());
        data.extend_from_slice(&seed.to_le_bytes());
        data.push(prefix.len() as u8);
        data.extend_from_slice(prefix);
        let metas = vec![
            AccountMeta::new(self.model, false),
            AccountMeta::new(self.scratch, false),
            AccountMeta::new(self.genlog, false),
        ];
        self.run("generate", Instruction::new_with_bytes(PROGRAM_ID, &data, metas))
    }

    fn contribute(&mut self, name: &str) -> u64 {
        let mut data = vec![ix::CONTRIBUTE];
        data.extend(name.bytes().map(|b| b - b'a'));
        let metas = vec![
            AccountMeta::new_readonly(self.payer, true),
            AccountMeta::new(self.community, false),
        ];
        self.run("contribute", Instruction::new_with_bytes(PROGRAM_ID, &data, metas))
    }
}

fn load_names(limit: usize) -> Vec<String> {
    let raw =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../reference/names.txt"))
            .expect("reference/names.txt");
    raw.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && l.len() <= 15 && l.bytes().all(|b| b.is_ascii_lowercase()))
        .take(limit)
        .collect()
}

#[test]
fn full_lifecycle_and_parity() {
    let mut h = Harness::new();

    // -- Init ---------------------------------------------------------------
    let cu_init = h.init_model();
    h.grow_all();
    assert_eq!(h.account(&h.model).data.len(), MODEL_ACCOUNT_LEN);
    assert_eq!(h.account(&h.optimizer).data.len(), OPTIMIZER_ACCOUNT_LEN);
    assert_eq!(h.account(&h.scratch).data.len(), SCRATCH_ACCOUNT_LEN);
    assert_eq!(h.account(&h.dataset).data.len(), DATASET_ACCOUNT_LEN);
    assert_eq!(h.account(&h.community).data.len(), COMMUNITY_ACCOUNT_LEN);
    assert_eq!(h.account(&h.genlog).data.len(), GENLOG_ACCOUNT_LEN);
    let header = h.header();
    assert_eq!(header.magic, *b"pGPT");
    assert_eq!(header.seed, SEED);
    assert_eq!(header.pow_beta1, ONE);

    // -- Weights (chunked) --------------------------------------------------
    let mut cu_init_weights = 0;
    while h.header().flags & FLAG_WEIGHTS_READY == 0 {
        cu_init_weights = cu_init_weights.max(h.init_weights(1024));
    }
    assert_eq!(h.header().init_cursor as usize, N_PARAMS);

    // -- Dataset ------------------------------------------------------------
    let names = load_names(64);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut cu_load = 0;
    for chunk in refs.chunks(48) {
        cu_load = cu_load.max(h.load_docs(chunk));
    }

    // -- Train --------------------------------------------------------------
    let steps = 10u64;
    let mut cu_train_max = 0;
    for _ in 0..steps {
        cu_train_max = cu_train_max.max(h.train_step(1));
    }
    let header = h.header();
    assert_eq!(header.step, steps);
    assert!(header.last_loss > 0);
    assert_eq!(header.ring_pos, steps);

    // -- Host parity: the exact same computation off-chain ------------------
    let mut w: Box<Weights> = unsafe { Box::new_zeroed().assume_init() };
    let mut mom: Box<Moments> = unsafe { Box::new_zeroed().assume_init() };
    let mut scr: Box<Scratch> = unsafe { Box::new_zeroed().assume_init() };
    let mut rng = Rng::new(SEED);
    w.init_range(0, N_PARAMS, &mut rng);
    let (mut pb1, mut pb2) = (ONE, ONE);
    for step in 0..steps {
        let idx = (step.wrapping_mul(p_gpt_interface::DOC_STRIDE) % refs.len() as u64) as usize;
        let name = refs[idx];
        let mut tokens = vec![gpt_core::BOS];
        tokens.extend(name.bytes().map(|b| b - b'a'));
        tokens.push(gpt_core::BOS);
        gpt_core::train_doc(&mut w, &mut mom, &mut scr, &tokens, step, &mut pb1, &mut pb2);
    }
    let host_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(w.as_flat().as_ptr() as *const u8, N_PARAMS * 8) };
    assert_eq!(
        host_bytes,
        h.weights_bytes(),
        "on-chain weights diverge from host replay — determinism broken"
    );

    // -- Generate -----------------------------------------------------------
    // Worst case: a 15-token prefix forces all 16 forward positions.
    let cu_gen_full = h.generate(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 7);
    let cu_gen = h.generate(&[], 7);
    let genlog = h.account(&h.genlog).data.clone();
    let log_header: GenLogHeader = unsafe { std::ptr::read(genlog.as_ptr() as *const _) };
    assert_eq!(log_header.total, 2);
    let record: GenRecord = unsafe {
        std::ptr::read(genlog.as_ptr().add(std::mem::size_of::<GenLogHeader>()) as *const _)
    };
    assert!(record.len as usize <= 16);

    // -- Contribute + community training ------------------------------------
    h.contribute("gabriele");
    // Step 16 (index 15) is a community step: 15 % 8 == 7.
    let mut cu_batch = 0;
    for _ in 0..6 {
        cu_batch = cu_batch.max(h.train_step(1));
    }
    assert_eq!(h.header().step, 16);

    // -- Batched steps ------------------------------------------------------
    let cu_train4 = h.train_step(4);
    assert_eq!(h.header().step, 20);

    println!("\n== p-gpt compute units ==");
    println!("init_model          {cu_init:>10}");
    println!("init_weights(1024)  {cu_init_weights:>10}");
    println!("load_docs(48)       {cu_load:>10}");
    println!("train_step(1) max   {cu_train_max:>10}");
    println!("train_step(4)       {cu_train4:>10}");
    println!("generate            {cu_gen:>10}");
    println!("generate (16 pos)   {cu_gen_full:>10}");
    assert!(cu_gen_full < 1_390_000, "worst-case generate exceeds a 1.4M CU transaction");
}

#[test]
fn train_step_fails_without_docs() {
    let mut h = Harness::new();
    h.init_model();
    h.grow_all();
    while h.header().flags & FLAG_WEIGHTS_READY == 0 {
        h.init_weights(2048);
    }
    let data = vec![ix::TRAIN_STEP, 1];
    let metas = vec![
        AccountMeta::new(h.model, false),
        AccountMeta::new(h.optimizer, false),
        AccountMeta::new(h.scratch, false),
        AccountMeta::new_readonly(h.dataset, false),
        AccountMeta::new_readonly(h.community, false),
    ];
    let ixn = Instruction::new_with_bytes(PROGRAM_ID, &data, metas);
    let result = h.mollusk.process_instruction(&ixn, &h.accounts);
    assert!(!matches!(result.program_result, ProgramResult::Success));
}

#[test]
fn rejects_bad_contribution() {
    let mut h = Harness::new();
    h.init_model();
    let mut data = vec![ix::CONTRIBUTE];
    data.push(26); // BOS is not a valid name token
    let metas =
        vec![AccountMeta::new_readonly(h.payer, true), AccountMeta::new(h.community, false)];
    let ixn = Instruction::new_with_bytes(PROGRAM_ID, &data, metas);
    let result = h.mollusk.process_instruction(&ixn, &h.accounts);
    assert!(!matches!(result.program_result, ProgramResult::Success));
}

#[test]
fn micro_path_matches_fused() {
    // Path A: fused steps.
    let mut fused = Harness::new();
    fused.init_model();
    fused.grow_all();
    while fused.header().flags & FLAG_WEIGHTS_READY == 0 {
        fused.init_weights(2048);
    }
    let names = load_names(64);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    for chunk in refs.chunks(48) {
        fused.load_docs(chunk);
    }
    for _ in 0..3 {
        fused.train_step(1);
    }

    // Path B: the same three steps as sub-1.4M CU micro transactions.
    let mut micro = Harness::new();
    micro.init_model();
    micro.grow_all();
    while micro.header().flags & FLAG_WEIGHTS_READY == 0 {
        micro.init_weights(2048);
    }
    for chunk in refs.chunks(48) {
        micro.load_docs(chunk);
    }
    let mut cu_by_phase = [0u64; 4];
    let mut txs = 0;
    while micro.header().step < 3 {
        let (phase, cu) = micro.train_micro();
        cu_by_phase[phase as usize] = cu_by_phase[phase as usize].max(cu);
        txs += 1;
        assert!(txs < 400, "micro path not converging to 3 steps");
    }

    assert_eq!(fused.header().step, micro.header().step);
    assert_eq!(fused.weights_bytes(), micro.weights_bytes(), "split path diverges from fused path");
    assert_eq!(fused.header().last_loss, micro.header().last_loss);

    println!("\n== micro path compute units (max per phase) ==");
    println!("pick      {:>9}", cu_by_phase[0]);
    println!("forward   {:>9}  (1 position/tx)", cu_by_phase[1]);
    println!("backward  {:>9}  (1 position/tx)", cu_by_phase[2]);
    println!("adam      {:>9}  (256 params/tx)", cu_by_phase[3]);
    println!("txs for 3 steps: {txs}");
    // Every micro-op must fit the ER crank tick budget: tick transactions get
    // the runtime default of 200K CU per top-level instruction (2 instructions
    // -> 400K total), and cannot request more.
    assert!(cu_by_phase.iter().all(|cu| *cu < 380_000), "micro op exceeds crank tick budget");
}

#[test]
fn negative_coverage_for_guards() {
    let mut h = Harness::new();
    h.init_model();
    h.grow_all();
    while h.header().flags & FLAG_WEIGHTS_READY == 0 {
        h.init_weights(2048);
    }
    let names = load_names(8);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    h.load_docs(&refs);

    // Start a split step (pick phase runs), leaving the state machine mid-step.
    h.train_micro();
    assert_ne!(h.header_field_phase(), 0);

    // A fused TrainStep over the in-flight split step must be rejected.
    let data = vec![ix::TRAIN_STEP, 1];
    let metas = vec![
        AccountMeta::new(h.model, false),
        AccountMeta::new(h.optimizer, false),
        AccountMeta::new(h.scratch, false),
        AccountMeta::new_readonly(h.dataset, false),
        AccountMeta::new_readonly(h.community, false),
    ];
    let ixn = Instruction::new_with_bytes(PROGRAM_ID, &data, metas);
    let result = h.mollusk.process_instruction(&ixn, &h.accounts);
    assert!(!matches!(result.program_result, ProgramResult::Success));

    // Contribute must reject a non-community program-owned account (the
    // dataset shares the magic and is attacker-substitutable otherwise).
    let mut data = vec![ix::CONTRIBUTE];
    data.extend([6u8, 0, 1]);
    let metas = vec![AccountMeta::new_readonly(h.payer, true), AccountMeta::new(h.dataset, false)];
    let ixn = Instruction::new_with_bytes(PROGRAM_ID, &data, metas);
    let result = h.mollusk.process_instruction(&ixn, &h.accounts);
    assert!(!matches!(result.program_result, ProgramResult::Success));

    // Generate must reject a temperature that would overflow the reciprocal.
    let mut data = vec![ix::GENERATE];
    data.extend_from_slice(&1i64.to_le_bytes()); // raw 1 => ~2e-10, far below 1/64
    data.extend_from_slice(&7u64.to_le_bytes());
    data.push(0);
    let metas = vec![
        AccountMeta::new(h.model, false),
        AccountMeta::new(h.scratch, false),
        AccountMeta::new(h.genlog, false),
    ];
    let ixn = Instruction::new_with_bytes(PROGRAM_ID, &data, metas);
    let result = h.mollusk.process_instruction(&ixn, &h.accounts);
    assert!(!matches!(result.program_result, ProgramResult::Success));
}
