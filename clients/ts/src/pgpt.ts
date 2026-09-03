// p-gpt client library: addresses, instruction builders, account decoders.
// Mirrors interface/src (the Rust source of truth) — offsets are documented
// there and asserted by the program's tests.

import { PublicKey, TransactionInstruction, SystemProgram } from "@solana/web3.js";

export const PROGRAM_ID = new PublicKey("6wPpJuYKKPbLYfYZpVeytPwxcq7TdGsgEHwyhYBangEC");
export const DELEGATION_PROGRAM_ID = new PublicKey("DELeGGvXpWV2fqJUhqcF5ZSYMS4JTLjteaAMARRSaeSh");
export const MAGIC_PROGRAM_ID = new PublicKey("Magic11111111111111111111111111111111111111");
export const MAGIC_CONTEXT_ID = new PublicKey("MagicContext1111111111111111111111111111111");

export const IX = {
  INIT_MODEL: 0,
  INIT_WEIGHTS: 1,
  LOAD_DOCS: 2,
  DELEGATE: 3,
  TRAIN_STEP: 4,
  TRAIN_MICRO: 5,
  SCHEDULE_TRAINING: 7,
  CHECKPOINT: 8,
  UNDELEGATE: 9,
  GENERATE: 10,
  CONTRIBUTE: 11,
  GROW: 12,
  DELEGATE_PREP: 13,
  INIT_SHARDS: 14,
} as const;

export const SEEDS = ["model", "opt", "scratch", "data", "community", "gen"] as const;
export type Which = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;
export const SHARD_COUNT = 4;

// Account sizes (kept in sync with interface/src/lib.rs).
export const N_PARAMS = 4192;
export const MODEL_HEADER_LEN = 2296;
export const LOSS_RING_LEN = 256;

export function pda(which: Which): PublicKey {
  if (which >= 6) {
    // Checkpoint shards: seeds ["shard", [k]].
    return PublicKey.findProgramAddressSync(
      [Buffer.from("shard"), Buffer.from([which - 6])],
      PROGRAM_ID,
    )[0];
  }
  return PublicKey.findProgramAddressSync(
    [Buffer.from(SEEDS[which as 0 | 1 | 2 | 3 | 4 | 5])],
    PROGRAM_ID,
  )[0];
}

export const shardPdas = () =>
  Array.from({ length: SHARD_COUNT }, (_, k) => pda((6 + k) as Which));

export const addrs = () => ({
  model: pda(0),
  optimizer: pda(1),
  scratch: pda(2),
  dataset: pda(3),
  community: pda(4),
  genlog: pda(5),
});

// -- encoding helpers --------------------------------------------------------

const u32 = (n: number) => {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n);
  return b;
};
const u64 = (n: bigint | number) => {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n));
  return b;
};
const i64 = (n: bigint) => {
  const b = Buffer.alloc(8);
  b.writeBigInt64LE(n);
  return b;
};

export const Q32_ONE = 1n << 32n;
export const toQ32 = (x: number) => BigInt(Math.round(x * 2 ** 32));
export const fromQ32 = (x: bigint) => Number(x) / 2 ** 32;

export function tokenize(name: string): Buffer {
  const out = Buffer.alloc(name.length);
  for (let i = 0; i < name.length; i++) {
    const c = name.charCodeAt(i) - 97;
    if (c < 0 || c > 25) throw new Error(`invalid char in name: ${name}`);
    out[i] = c;
  }
  return out;
}

export const detokenize = (tokens: Uint8Array, len: number) =>
  Array.from(tokens.slice(0, len), (t) => String.fromCharCode(97 + t)).join("");

// -- instruction builders ----------------------------------------------------

const w = (k: PublicKey) => ({ pubkey: k, isSigner: false, isWritable: true });
const r = (k: PublicKey) => ({ pubkey: k, isSigner: false, isWritable: false });
const ws = (k: PublicKey) => ({ pubkey: k, isSigner: true, isWritable: true });
const rs = (k: PublicKey) => ({ pubkey: k, isSigner: true, isWritable: false });

export function initModelIx(payer: PublicKey, seed: bigint): TransactionInstruction {
  const a = addrs();
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      ws(payer),
      w(a.model),
      w(a.optimizer),
      w(a.scratch),
      w(a.dataset),
      w(a.community),
      w(a.genlog),
      r(SystemProgram.programId),
    ],
    data: Buffer.concat([Buffer.from([IX.INIT_MODEL]), u64(seed)]),
  });
}

export function growIx(payer: PublicKey, which: Which): TransactionInstruction {
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [ws(payer), w(pda(which)), r(SystemProgram.programId)],
    data: Buffer.from([IX.GROW, which]),
  });
}

export function initWeightsIx(count: number): TransactionInstruction {
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [w(pda(0))],
    data: Buffer.concat([Buffer.from([IX.INIT_WEIGHTS]), u32(count)]),
  });
}

export function loadDocsIx(authority: PublicKey, names: string[]): TransactionInstruction {
  const records = names.map((name) => {
    const rec = Buffer.alloc(16);
    rec[0] = name.length;
    tokenize(name).copy(rec, 1);
    return rec;
  });
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [rs(authority), w(pda(0)), w(pda(3))],
    data: Buffer.concat([Buffer.from([IX.LOAD_DOCS]), ...records]),
  });
}

/// Delegation side accounts for one PDA (seeds mirror the on-chain SDK).
export function delegationAccounts(delegated: PublicKey) {
  const [buffer] = PublicKey.findProgramAddressSync(
    [Buffer.from("buffer"), delegated.toBuffer()],
    PROGRAM_ID,
  );
  const [record] = PublicKey.findProgramAddressSync(
    [Buffer.from("delegation"), delegated.toBuffer()],
    DELEGATION_PROGRAM_ID,
  );
  const [metadata] = PublicKey.findProgramAddressSync(
    [Buffer.from("delegation-metadata"), delegated.toBuffer()],
    DELEGATION_PROGRAM_ID,
  );
  return { buffer, record, metadata };
}

export function delegateIx(
  payer: PublicKey,
  which: Which,
  validator: PublicKey,
  commitFrequencyMs = 30_000,
): TransactionInstruction {
  const delegated = pda(which);
  const { buffer, record, metadata } = delegationAccounts(delegated);
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      ws(payer),
      w(delegated),
      r(PROGRAM_ID),
      w(buffer),
      w(record),
      w(metadata),
      r(SystemProgram.programId),
      r(DELEGATION_PROGRAM_ID),
      r(pda(0)), // model: authority check
    ],
    data: Buffer.concat([
      Buffer.from([IX.DELEGATE, which]),
      u32(commitFrequencyMs),
      validator.toBuffer(),
    ]),
  });
}

export function delegatePrepIx(payer: PublicKey, which: Which): TransactionInstruction {
  const delegated = pda(which);
  const { buffer } = delegationAccounts(delegated);
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      ws(payer),
      r(delegated),
      w(buffer),
      r(SystemProgram.programId),
      r(pda(0)), // model: authority check
    ],
    data: Buffer.from([IX.DELEGATE_PREP, which]),
  });
}

export function trainStepIx(count: number): TransactionInstruction {
  const a = addrs();
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [w(a.model), w(a.optimizer), w(a.scratch), r(a.dataset), r(a.community)],
    data: Buffer.from([IX.TRAIN_STEP, count]),
  });
}

export function trainMicroIx(): TransactionInstruction {
  const a = addrs();
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [w(a.model), w(a.optimizer), w(a.scratch), r(a.dataset), r(a.community)],
    data: Buffer.from([IX.TRAIN_MICRO]),
  });
}

export function scheduleTrainingIx(
  payer: PublicKey,
  taskId: bigint,
  intervalMs: bigint,
  iterations: bigint,
  stepsPerTick: number,
): TransactionInstruction {
  const a = addrs();
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      ws(payer),
      r(MAGIC_PROGRAM_ID),
      w(a.model),
      w(a.optimizer),
      w(a.scratch),
      r(a.dataset),
      r(a.community),
    ],
    data: Buffer.concat([
      Buffer.from([IX.SCHEDULE_TRAINING]),
      u64(taskId),
      u64(intervalMs),
      u64(iterations),
      Buffer.from([stepsPerTick]),
    ]),
  });
}

export function initShardsIx(payer: PublicKey): TransactionInstruction {
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [ws(payer), ...shardPdas().map(w), r(SystemProgram.programId)],
    data: Buffer.from([IX.INIT_SHARDS]),
  });
}

export function checkpointIx(payer: PublicKey): TransactionInstruction {
  const a = addrs();
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      ws(payer),
      w(MAGIC_CONTEXT_ID),
      r(MAGIC_PROGRAM_ID),
      r(a.model),
      w(a.genlog),
      ...shardPdas().map(w),
    ],
    data: Buffer.from([IX.CHECKPOINT]),
  });
}

export function undelegateIx(payer: PublicKey): TransactionInstruction {
  const a = addrs();
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      ws(payer),
      w(MAGIC_CONTEXT_ID),
      r(MAGIC_PROGRAM_ID),
      r(a.model),
      w(a.community),
      w(a.genlog),
      ...shardPdas().map(w),
    ],
    data: Buffer.from([IX.UNDELEGATE]),
  });
}

export function generateIx(prefix: string, temperature = 0.5, seed?: bigint): TransactionInstruction {
  const a = addrs();
  const prefixTokens = tokenize(prefix);
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [w(a.model), w(a.scratch), w(a.genlog)],
    data: Buffer.concat([
      Buffer.from([IX.GENERATE]),
      i64(toQ32(temperature)),
      u64(seed ?? BigInt(Math.floor(Math.random() * 2 ** 53))),
      Buffer.from([prefixTokens.length]),
      prefixTokens,
    ]),
  });
}

export function contributeIx(contributor: PublicKey, name: string): TransactionInstruction {
  return new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [rs(contributor), w(pda(4))],
    data: Buffer.concat([Buffer.from([IX.CONTRIBUTE]), tokenize(name)]),
  });
}

// -- decoders ----------------------------------------------------------------

export interface ModelHeader {
  version: number;
  flags: number;
  authority: PublicKey;
  seed: bigint;
  initCursor: bigint;
  step: bigint;
  lossEma: number;
  lastLoss: number;
  genCount: bigint;
  ringPos: bigint;
  lossRing: number[]; // chronological, oldest first
  /// Split-training phase (0 pick, 1 forward, 2 backward, 3 adam) + cursor.
  phase: number;
  phaseCursor: number;
  /// The name the in-flight training step is learning from.
  currentDoc: string;
}

export function decodeModelHeader(data: Buffer | Uint8Array): ModelHeader {
  const b = Buffer.from(data);
  if (b.length < MODEL_HEADER_LEN || b.toString("latin1", 0, 4) !== "pGPT" || b[4] !== 1) {
    throw new Error("not a p-gpt model account (v1)");
  }
  const ringPos = b.readBigUInt64LE(120);
  const raw: number[] = [];
  for (let i = 0; i < LOSS_RING_LEN; i++) {
    raw.push(Number(b.readBigInt64LE(128 + i * 8)) / 2 ** 32);
  }
  const filled = Number(ringPos < BigInt(LOSS_RING_LEN) ? ringPos : BigInt(LOSS_RING_LEN));
  const head = Number(ringPos % BigInt(LOSS_RING_LEN));
  const lossRing: number[] = [];
  for (let i = 0; i < filled; i++) {
    lossRing.push(raw[(head - filled + i + LOSS_RING_LEN) % LOSS_RING_LEN]);
  }
  return {
    version: b[4],
    flags: b[11],
    authority: new PublicKey(b.subarray(16, 48)),
    seed: b.readBigUInt64LE(48),
    initCursor: b.readBigUInt64LE(64),
    step: b.readBigUInt64LE(72),
    lossEma: Number(b.readBigInt64LE(96)) / 2 ** 32,
    lastLoss: Number(b.readBigInt64LE(104)) / 2 ** 32,
    genCount: b.readBigUInt64LE(112),
    ringPos,
    lossRing,
    phase: b[128 + LOSS_RING_LEN * 8],
    phaseCursor: b[128 + LOSS_RING_LEN * 8 + 1],
    currentDoc: (() => {
      // doc_tokens_len at +2, doc (BOS, chars.., BOS) at +8 after the ring.
      const base = 128 + LOSS_RING_LEN * 8;
      const len = b[base + 2];
      if (len < 2 || len > 18) return "";
      return detokenize(b.subarray(base + 8 + 1, base + 8 + len - 1), len - 2);
    })(),
  };
}

/// The 4,192 weights as float32-ish numbers (Q32.32 -> f64), in the canonical
/// flat order (wte, wpe, wq, wk, wv, wo, w1, w2, lm).
export function decodeWeights(data: Buffer | Uint8Array): Float64Array {
  const b = Buffer.from(data);
  const out = new Float64Array(N_PARAMS);
  for (let i = 0; i < N_PARAMS; i++) {
    out[i] = Number(b.readBigInt64LE(MODEL_HEADER_LEN + i * 8)) / 2 ** 32;
  }
  return out;
}

export interface GenEntry {
  step: bigint;
  name: string;
}

export function decodeGenLog(data: Buffer | Uint8Array): { total: bigint; entries: GenEntry[] } {
  const b = Buffer.from(data);
  const capacity = b.readBigUInt64LE(8);
  const total = b.readBigUInt64LE(16);
  const count = total < capacity ? Number(total) : Number(capacity);
  const entries: GenEntry[] = [];
  for (let i = 0; i < count; i++) {
    // chronological: oldest surviving entry first
    const slot = Number((total - BigInt(count) + BigInt(i)) % capacity);
    const off = 32 + slot * 32;
    const step = b.readBigUInt64LE(off);
    const len = b[off + 8];
    entries.push({ step, name: detokenize(b.subarray(off + 9, off + 25), len) });
  }
  return { total, entries };
}

export function decodeDocsCount(data: Buffer | Uint8Array): bigint {
  return Buffer.from(data).readBigUInt64LE(16);
}
