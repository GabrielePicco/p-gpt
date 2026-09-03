# p-gpt — Perpetual on-chain GPT training & inference

> **Status: built.** This document is the original design plan; the
> implementation followed it with deviations forced by measured reality
> (micro-split training instead of a fused ER step, checkpoint shards for the
> >10KB commit limit, LUT/reciprocal math for the SBF cost model). See
> `README.md` for what actually shipped and the measured compute numbers.

> A pinocchio-based GPT program. Karpathy's microGPT (4,192 params), trained forever on a
> MagicBlock Ephemeral Rollup, checkpointed to Solana. Every gradient step is a transaction.
> The model's entire provenance — init seed, every SGD step, every sample — is replayable chain state.

## 0. The one-sentence pitch

**Solana L1 is where the model lives; the Ephemeral Rollup is where it learns.** A crank on the ER
runs `TrainStep` perpetually with no server anywhere; the weights PDA commits back to mainnet as
periodic checkpoints; anyone can call `Generate` against the live weights, on either layer.

---

## 1. The model (fixed by upstream, do not improvise)

From [karpathy's microgpt gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
(vendored at `reference/microgpt.py`):

| Hyperparam | Value |
|---|---|
| vocab_size | 27 (a–z + BOS) |
| n_layer / n_head / head_dim | 1 / 4 / 4 |
| n_embd / block_size | 16 / 16 |
| Norm / MLP / biases | RMSNorm / ReLU, 4× expansion / none |
| Params | **4,192** (wte 432, wpe 256, wq/wk/wv/wo 4×256, fc1 1024, fc2 1024, lm_head 432) |
| Optimizer | Adam (lr 0.01, β 0.85/0.99, bias-corrected), linear LR decay |
| Training | 1 doc (name) per step, avg-CE loss over ≤16 positions, 1000 steps upstream |
| Dataset | makemore `names.txt`, 32,033 names, chars a–z, ≤15 chars |

**Perpetual twist** (our only deviations, all in the direction of "runs forever"):
- LR schedule: linear warmdown over the first 1000 steps to a constant floor (e.g. 0.001), instead of decay to 0. Stored in the header, tunable by authority.
- Data order: pseudo-shuffle via `doc_idx = (step * STRIDE) % num_docs` with STRIDE co-prime to num_docs (no on-chain shuffle needed).
- A second, append-only community dataset account; the sampler alternates into it once non-empty.

## 2. Numerics — the load-bearing decision

**Fixed-point, integer-only.** Deterministic across SVM validators, cheap in SBF, and bit-exact
replayable — determinism is the whole crypto-native story ("verify the model by re-executing its life").

- Weights & activations: `i32` in **Q16.16** (range ±32K, resolution 1.5e-5 — generous for a net whose weights live in ±1).
- Accumulators & gradients: `i64` in **Q32.32** (matmul dot products, loss, grads).
- Adam moments m, v: `i64` Q32.32.
- Transcendentals in `gpt-core::math`, all integer:
  - `exp_q` — range-reduce to 2^k·exp(r), degree-4 poly on r ∈ [0, ln2) (softmax, CE loss).
  - `rsqrt_q` — Newton–Raphson, 3 iterations (RMSNorm scale, Adam √v̂, attn 1/√head_dim is a constant).
  - `log_q` — only needed for loss *reporting*, poly on mantissa.
- Softmax: subtract-max in integer domain, exp_q, fixed-point divide (i128 intermediate).
- Init: xorshift128+ PRNG seeded from an `InitModel` arg; Gaussian via CLT (sum of 12 uniforms − 6), scaled by 0.08 Q16.16. Cheap, deterministic, close enough for init.
- Backprop: **hand-derived analytic gradients** (no autograd graph): CE+softmax fuses to `probs − onehot`; RMSNorm, attention (softmax-jacobian-vector trick), ReLU mask, linears — all standard, all verified against the Python reference in M0.

**Fallback if fixed-point training diverges** (decided at the M0 gate, not later): SBF soft-float
`f32` end-to-end. ~5–20× CU cost but still deterministic (compiled softfloat) and trivially
parity-checkable against Python. The ER's configurable CU ceiling makes this survivable; on L1 it
would force the split-instruction path everywhere.

### CU budget (estimates — M0/M1 measures with mollusk before anything is frozen)

Per-position forward ≈ 3.6K MACs (+32·t attn). Avg name (~7 chars, 8 positions): forward ≈ 30K MACs,
backward ≈ 2×, Adam ≈ 4,192 × ~25 CU (rsqrt + divides). At ~5 CU per fixed-point MAC:

| Op | Est. CU | Fits 1.4M CU tx? |
|---|---|---|
| `TrainStep` (avg doc, fused fwd+bwd+Adam) | ~0.6–1.0M | yes |
| `TrainStep` (worst case, 16-char doc) | ~1.5–2.5M | **no → split path or ER raised limit** |
| `Generate` (16 tokens, forward only) | ~0.3–0.5M | yes, comfortably |
| `InitWeights` (per chunk of ~1K params) | ~0.1M | yes |

Design consequence: `TrainStep` is fused by default (ER runs with a raised per-tx CU ceiling — our
validator, our config), and a split pair `TrainFwdBwd` (grads → scratch) + `ApplyAdam(range)`
(chunked over param ranges) exists so the *same program* can demonstrably train on vanilla L1
within 1.4M CU/tx and the 12M CU/account/block write limit. The L1-trainable claim matters for the
showcase ("no cheating — the base layer can do it, the ER just does it 100× faster").

## 3. Repository layout (febo/p-token conventions)

```
neural-net-svm/
├── Cargo.toml                 # workspace; rustfmt.toml copied from p-token
├── reference/microgpt.py      # vendored upstream, pinned
├── gpt-core/                  # ★ no_std, zero-dep math core — the soul of the project
│   └── src/{lib,fixed,math,rng,model,backprop,adam}.rs
├── interface/                 # instruction layouts, discriminators, PDA derivation, state offsets
│   └── src/{lib,instruction,state,pda}.rs
├── program/                   # the on-chain program (crate-type cdylib, no_std)
│   └── src/
│       ├── entrypoint.rs      # program_entrypoint! + allocator + panic handler
│       ├── lib.rs
│       ├── processor/
│       │   ├── mod.rs
│       │   ├── init_model.rs      #  0
│       │   ├── init_weights.rs    #  1 (chunked)
│       │   ├── load_dataset.rs    #  2 (chunked)
│       │   ├── delegate.rs        #  3
│       │   ├── train_step.rs      #  4 ★
│       │   ├── train_fwd_bwd.rs   #  5 (L1 split path)
│       │   ├── apply_adam.rs      #  6 (L1 split path, param-range chunked)
│       │   ├── schedule_training.rs # 7 (ER crank)
│       │   ├── checkpoint.rs      #  8 (ER → commit intent)
│       │   ├── undelegate.rs      #  9 (commit_and_undelegate)
│       │   ├── generate.rs        # 10 ★
│       │   ├── contribute_doc.rs  # 11
│       │   └── shared/            # account validation, zero-copy loaders
│       └── state/                 # #[repr(C)] zero-copy structs (mirror interface offsets)
├── clients/ts/                # @…/p-gpt: PDA helpers, ix builders, account decoders, ER routing
├── tools/parity/              # host binary: gpt-core vs microgpt.py lockstep comparison
├── tools/dataset/             # tokenize names.txt → packed binary chunks
└── app/                       # the showcase UI (Next.js)
```

Style contract (lifted from p-token, enforced in review):
- `program_entrypoint!(process_instruction)` + `default_panic_handler!()`; u8 discriminator,
  flat `match`, hot instructions (`TrainStep`, `Generate`) first; `#[inline(always)]` processors;
  `#[cfg(feature = "logging")] msg!("Instruction: TrainStep")`.
- Allocator: start with `no_allocator!()`. `ephemeral-rollups-pinocchio`'s intent bundle has a
  `no_vec` path and `build_and_invoke(data_buf)` takes a caller-provided buffer; if the delegate
  CPI path turns out to require alloc (`delegation-actions` feature pulls `pinocchio/alloc`),
  switch to `default_allocator!()` and note it — correctness over purity, but try no-alloc first.
- State: `#[repr(C)]` structs over raw account bytes, explicit offsets in `interface`, no
  serde/borsh anywhere near the hot path. Weights are a flat `[i32; 4192]` at a fixed offset —
  the account *is* the tensor.
- `gpt-core` has **zero** pinocchio/solana deps → unit-tested, fuzzed, and benchmarked on host.

## 4. Accounts

All PDAs of `p-gpt`, seeds shown. Sizes exact except scratch (finalized in M1).

| Account | Seeds | Size | Delegated? | Contents |
|---|---|---|---|---|
| `Model` | `["model", id]` | ~17.2 KB | ✅ committed | header (version, authority, step u64, lr floor, loss EMA q32.32, last-loss, PRNG state, tokenizer table, delegation/status flags) + `[i32; 4192]` weights |
| `Optimizer` | `["opt", model]` | ~67 KB | ✅ committed rarely | `[i64; 4192]` m + `[i64; 4192]` v |
| `Scratch` | `["scratch", model]` | ~96 KB | ✅ never committed | per-position activations (x, norms, q/k/v, attn weights 4×16×16, mlp hidden+mask) ≈ 23 KB + `[i64; 4192]` grads ≈ 33 KB + headroom |
| `Dataset` | `["data", model, n]` | ~256 KB | ❌ (read-only clone) | packed tokenized names: `[len u8, tokens…]`, count in header |
| `Community` | `["community", model]` | 64 KB grow | ✅ committed | append-only user-contributed docs |
| `GenLog` | `["gen", model]` | 16 KB ring | ✅ committed | ring buffer of (step, sampled name) — the "babble feed" the UI streams |

Loss history for the chart: `Model.header` keeps loss EMA + last N losses in a small ring
(e.g. 256 × i32); full-resolution history comes from event logs (`sol_log_data`) which the UI
indexes from the ER websocket — no extra account writes.

## 5. Instructions

| # | Ix | Layer | What it does |
|---|---|---|---|
| 0 | `InitModel { id, seed, hyper }` | base | create Model/Optimizer/Scratch/GenLog PDAs, write header + tokenizer |
| 1 | `InitWeights { range }` | base | PRNG-init `weights[range]` (chunked to stay under CU); PRNG state persists in header so chunks are order-enforced |
| 2 | `LoadDataset { chunk_idx, bytes }` | base | append packed docs; authority-gated; ~900 B/tx |
| 3 | `Delegate` | base | `ephemeral_rollups_pinocchio::instruction::delegate_account(accounts, seeds, bump, DelegateConfig)` for Model, Optimizer, Scratch, Community, GenLog |
| 4 | `TrainStep { max_docs }` | **ER** | pick doc via `(step·STRIDE) % n` (alternating into Community), fused fwd+bwd+Adam in fixed point, ++step, update loss EMA + ring, `sol_log_data(step, loss)` |
| 5 | `TrainFwdBwd` | base | L1 split path: forward+backward, grads → Scratch |
| 6 | `ApplyAdam { param_range }` | base | L1 split path: Adam over a param slice; last slice ++step |
| 7 | `ScheduleTraining { task_id, interval_ms, iterations }` | **ER** | schedule crank: `TrainStep` every `interval_ms` (e.g. 100 ms), and a second slower crank calling `Checkpoint` every K steps. `iterations = u64::MAX` → **perpetual** |
| 8 | `Checkpoint` | **ER** | `MagicIntentBundleBuilder::new(payer, magic_context, magic_program).magic_fee_vault(vault).commit(&[model, gen_log])` — periodic mainnet checkpoint; Optimizer committed every ~10th time |
| 9 | `Undelegate` | **ER** | `commit_and_undelegate` everything (graduation / upgrade / re-delegation to refresh state) |
| 10 | `Generate { prefix, temp_q, seed }` | both | forward-only sampling from BOS+prefix up to 16 tokens; randomness = user seed ⊕ slot-hash (or ephemeral-VRF for the provable variant); result → GenLog ring + `sol_log_data` + return data |
| 11 | `ContributeDoc { name }` | **ER** | validate (a–z, 1–15 chars), append to Community — your name becomes training data in the next epoch |

**Perpetual mechanics** (the parts that make "forever" true):
- Commits beyond the 10 sponsored ones: `magic_fee_vault` PDA (derived from the validator in the
  delegation record, seeds `["magic-fee-vault", validator]`) + a delegated fee-payer PDA that
  signs the intent bundle via seeds; top it up from base layer with `lamportsDelegatedTransferIx`.
- Crank pays via that same delegated payer → no keypair on any server, no server at all.
- `Checkpoint` cadence: every 50 steps at 100 ms/step → a mainnet commit ≈ every 5 s of ER time;
  tune real cadence against fee-vault burn rate.

## 6. Verification story (what makes this more than a demo)

1. **Bit-exact parity** (`tools/parity`): a Python mirror of the fixed-point arithmetic runs
   lockstep with `gpt-core` for 1000 steps; every weight equal at every step, and the fixed-point
   loss curve tracks float microgpt.py within tolerance (expect final loss ≈ 2.0–2.2 like upstream).
2. **CU ground truth**: mollusk-svm benches per instruction, committed as `benches/compute_units.md`
   (p-token does exactly this) — the README table of real CU numbers *is* marketing material.
3. **Replay**: a `tools/parity` mode that replays the full on-chain history (init seed + ordered
   TrainSteps from ER/L1 history) and reproduces the committed weights hash. "Don't trust the
   checkpoint — recompute it."

## 7. The UI (`app/`) — "watch a mind grow"

Next.js + Tailwind + shadcn, dual-connection wiring straight from the ER TS SDK (router
`getDelegationStatus` → ER fqdn websocket; base-layer connection for checkpoints). Everything below
is rendered from **chain state only** — no backend, the ER websocket is the backend.

**Hero — the living brain.** The 4,192 weights as a WebGL heatmap (six labeled tensor tiles:
wte 27×16, wpe 16×16, the four attention mats, the MLP pair, lm_head), re-rendered on every
`accountSubscribe` push from the ER (~10 Hz with a 100 ms crank). Weights visibly organizing out
of noise is the single most hypnotic thing we can put above the fold. Overlaid: an odometer —
**"Gradient steps on-chain: 1,284,391"** — plus live loss sparkline and steps/sec.

**The babble feed.** GenLog streamed as a vertical ticker with step tags: step 12 `"xqzjw"` →
step 900 `"karis"` → step 40,000 `"maribella"`. A timeline scrubber over mainnet checkpoints
replays history — this is the demo-day money shot: *gibberish becoming names, live, with a tx
signature on every line*.

**Attention lens.** After each generation, read the Scratch account and render the 4 heads'
attention matrices over the generated name's characters (4 × 16×16 mini-heatmaps). Live
interpretability of an on-chain transformer — nobody has shown this.

**Prompt box.** Type a prefix ("gab…") → `Generate` tx to the ER (skipPreflight) → token-by-token
typewriter reveal ↔ explorer link. Toggle: "provably random" (VRF) vs instant. Then: *"contribute
your name to the dataset"* → `ContributeDoc`, and the UI highlights when the model next trains on it.

**Two-layer panel.** Left card: Solana mainnet — Model owner = delegation program, last checkpoint
sig + age, weights hash, "verify replay" button. Right card: Ephemeral Rollup — fqdn, current step,
loss, crank status, commits used / fee-vault balance. An animated pulse travels right→left on every
commit. This panel *is* the MagicBlock architecture diagram, rendered with live data.

**Checkpoint fossil record.** Grid of mainnet checkpoints; select two → weight-diff heatmap +
their sampled names side by side. "The model's git history."

## 8. Milestones

**M0 — the math (days 1–4).** `gpt-core` + `tools/parity`. Fixed-point fwd/bwd/Adam training on
host reaches upstream-comparable loss; bit-exact lockstep vs the Python fixed-point mirror.
**Gate: fixed-point converges → proceed; else flip to soft-float f32 and re-budget CU.**

**M1 — the program (days 5–9).** Pinocchio skeleton per §3, state + init/load/generate wired,
mollusk tests + CU benches. **Gate: fused TrainStep CU measured → freeze fused-vs-split default.**

**M2 — training on-chain (week 2).** TrainStep/TrainFwdBwd/ApplyAdam; full 1000-step run on
localnet (solana-test-validator + local ephemeral-validator); replay tool reproduces weight hash.

**M3 — perpetual on the ER (week 3).** Delegate, cranks, Checkpoint with fee vault, ContributeDoc;
devnet soak: 24 h unattended training, measure steps/sec, commit cadence, fee burn.

**M4 — the showcase (week 4).** UI, brand pass, mainnet deploy, README with real CU numbers,
demo video of the babble feed.

## 9. Risks & pre-committed mitigations

| Risk | Mitigation |
|---|---|
| Fixed-point Adam/softmax instability | M0 gate + f32 softfloat fallback; loss-scale grads (Q32.32 headroom is large) |
| TrainStep over CU ceiling | split path already designed; ER ceiling is our own config |
| Crank executing ~1M CU every 100 ms stresses ER scheduler | it's our validator — tune interval; worst case widen to 250 ms (still ~4 steps/s, 100× L1 block cadence) |
| `no_allocator!` vs SDK delegate CPI needing alloc | try no_vec/no-alloc path first; else default bump allocator, documented |
| Commit quota exhaustion mid-"perpetual" claim | fee vault + delegated payer from day one (M3), lamports top-up runbook |
| Scratch sizing wrong for attention backward | M1 computes exact layout; account has 3× headroom and resize is possible pre-delegation |
| Upstream gist drift | vendored + pinned in `reference/` |

## 10. What we get to say when it ships

- First gradient-descent **training** loop ever run inside a blockchain runtime — not inference, not zkML attestation: the chain itself does the backprop.
- A real GPT — Karpathy's microGPT, faithfully — living at a Solana address, learning **perpetually** with no server, crank-driven on an Ephemeral Rollup, checkpointed to mainnet.
- Fully verifiable: seed + transaction history ⇒ bit-exact weights. The model has a provenance proof.
- And it's ~2K lines of no_std pinocchio Rust in p-token style: small enough to read in an afternoon.
