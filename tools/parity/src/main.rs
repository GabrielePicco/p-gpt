//! Parity harness: proves the fixed-point core is a faithful microGPT.
//!
//!   cargo run -p parity --release -- gradcheck
//!   cargo run -p parity --release -- train 1000 [--f64]
//!   cargo run -p parity --release -- bitrepro
//!
//! Reads reference/names.txt (vendored makemore dataset).

mod ref64;

use gpt_core::{
    fdiv, fmul, train_doc, Fx, Moments, Rng, Scratch, Weights, BLOCK, BOS, N_PARAMS, ONE, VOCAB,
};
use ref64::{doc_tokens, Adam64, W64};

fn to_f64(x: Fx) -> f64 {
    x as f64 / ONE as f64
}

fn new_boxed<T>() -> Box<T> {
    // SAFETY: Weights/Moments/Scratch are plain arrays of i64; zeroed is valid.
    unsafe { Box::<T>::new_zeroed().assume_init() }
}

fn load_names() -> Vec<String> {
    let raw = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/names.txt"),
    )
    .expect("reference/names.txt missing");
    let mut names: Vec<String> = raw
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && l.len() <= 15 && l.bytes().all(|b| b.is_ascii_lowercase()))
        .collect();
    // Deterministic shuffle, seed 42 (Fisher-Yates over our own PRNG).
    let mut rng = Rng::new(42);
    for i in (1..names.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        names.swap(i, j);
    }
    names
}

fn init_weights(seed: u64) -> (Box<Weights>, Box<W64>) {
    let mut w = new_boxed::<Weights>();
    let mut rng = Rng::new(seed);
    w.init_range(0, N_PARAMS, &mut rng);
    let mut w64 = Box::new(W64::zeros());
    w64.set_flat(&w.as_flat().iter().map(|x| to_f64(*x)).collect::<Vec<_>>());
    (w, w64)
}

fn gradcheck() {
    let (w, w64) = init_weights(42);
    let tokens = doc_tokens("gabriele");

    // 1. Analytic f64 grads vs central finite differences.
    let (loss, g64) = ref64::train_fwd_bwd(&w64, &tokens);
    println!("f64 loss: {loss:.6} (expect ~ln(27) = {:.6} at init)", (VOCAB as f64).ln());
    let gflat = g64.flat();
    let eps = 1e-6;
    let mut max_rel = 0f64;
    let mut checked = 0;
    let mut rng = Rng::new(7);
    for _ in 0..300 {
        let i = (rng.next_u64() % N_PARAMS as u64) as usize;
        let mut wp = (*w64).clone();
        let mut flat = wp.flat();
        flat[i] += eps;
        wp.set_flat(&flat);
        let lp = ref64::loss_only(&wp, &tokens);
        flat[i] -= 2.0 * eps;
        wp.set_flat(&flat);
        let lm = ref64::loss_only(&wp, &tokens);
        let fd = (lp - lm) / (2.0 * eps);
        let an = gflat[i];
        let denom = fd.abs().max(an.abs()).max(1e-6);
        let rel = (fd - an).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        checked += 1;
    }
    println!("finite-diff check: {checked} params, max rel err {max_rel:.3e}");
    assert!(max_rel < 1e-3, "analytic gradient does not match finite differences");

    // 2. Fixed-point grads vs f64 grads.
    let mut s = new_boxed::<Scratch>();
    let loss_fx = gpt_core::train_fwd_bwd(&w, &mut s, &tokens);
    println!("fixed loss: {:.6} (f64 {loss:.6})", to_f64(loss_fx));
    assert!((to_f64(loss_fx) - loss).abs() < 1e-3);
    // On-chain gradients are loss-scaled by 2^LOSS_SCALE_SHIFT.
    let scale = (1u64 << gpt_core::LOSS_SCALE_SHIFT) as f64;
    let mut max_abs = 0f64;
    let mut max_at = 0usize;
    for i in 0..N_PARAMS {
        let d = (to_f64(s.grads.as_flat()[i]) / scale - gflat[i]).abs();
        if d > max_abs {
            max_abs = d;
            max_at = i;
        }
    }
    println!("fixed-vs-f64 grads: max abs err {max_abs:.3e} (param {max_at})");
    assert!(max_abs < 2e-3, "fixed-point gradients diverge from f64");
    println!("gradcheck PASSED");
}

fn train(steps: u64, run_f64: bool) {
    let names = load_names();
    println!("num docs: {}", names.len());
    let (mut w, mut w64) = init_weights(42);
    let mut mom = new_boxed::<Moments>();
    let mut scratch = new_boxed::<Scratch>();
    let (mut pow_b1, mut pow_b2) = (ONE, ONE);
    let mut adam64 = Adam64::new();

    let mut window_fx = 0f64;
    let mut window_64 = 0f64;
    let report = (steps / 10).max(1);
    let t0 = std::time::Instant::now();
    for step in 0..steps {
        let tokens = doc_tokens(&names[(step % names.len() as u64) as usize]);
        let loss_fx =
            train_doc(&mut w, &mut mom, &mut scratch, &tokens, step, &mut pow_b1, &mut pow_b2);
        window_fx += to_f64(loss_fx);
        if run_f64 {
            let (l64, g64) = ref64::train_fwd_bwd(&w64, &tokens);
            adam64.step(&mut w64, &g64, step);
            window_64 += l64;
        }
        if (step + 1) % report == 0 {
            if run_f64 {
                println!(
                    "step {:5} | loss fixed {:.4} | f64 {:.4}",
                    step + 1,
                    window_fx / report as f64,
                    window_64 / report as f64
                );
            } else {
                println!("step {:5} | loss {:.4}", step + 1, window_fx / report as f64);
            }
            window_fx = 0.0;
            window_64 = 0.0;
        }
    }
    println!("trained {steps} steps in {:.2?}", t0.elapsed());

    println!("--- inference (fixed-point, temperature 0.5) ---");
    let mut rng = Rng::new(1337);
    let temp = ONE / 2;
    for _ in 0..20 {
        let mut out = [0u8; BLOCK];
        let n = gpt_core::generate(&w, &mut scratch, &[], temp, &mut rng, &mut out);
        let name: String = out[..n].iter().map(|t| (b'a' + t) as char).collect();
        println!("  {name}");
    }
    let _ = (fdiv(ONE, ONE), fmul(ONE, ONE), BOS);
}

fn bitrepro() {
    let names = load_names();
    let mut hashes = Vec::new();
    for _ in 0..2 {
        let (mut w, _) = init_weights(42);
        let mut mom = new_boxed::<Moments>();
        let mut scratch = new_boxed::<Scratch>();
        let (mut pb1, mut pb2) = (ONE, ONE);
        for step in 0..200u64 {
            let tokens = doc_tokens(&names[(step % names.len() as u64) as usize]);
            train_doc(&mut w, &mut mom, &mut scratch, &tokens, step, &mut pb1, &mut pb2);
        }
        // FNV-1a over the weight bytes.
        let mut h: u64 = 0xcbf29ce484222325;
        for p in w.as_flat() {
            for b in p.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        hashes.push(h);
        println!("weights hash: {h:016x}");
    }
    assert_eq!(hashes[0], hashes[1], "training is not deterministic!");
    println!("bitrepro PASSED");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gradcheck") => gradcheck(),
        Some("train") => {
            let steps = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let run_f64 = args.iter().any(|a| a == "--f64");
            train(steps, run_f64);
        }
        Some("bitrepro") => bitrepro(),
        _ => {
            eprintln!("usage: parity <gradcheck|train [steps] [--f64]|bitrepro>");
            std::process::exit(2);
        }
    }
}
