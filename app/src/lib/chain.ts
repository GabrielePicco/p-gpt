// Live chain wiring: dual connections (base layer + ephemeral rollup),
// account subscriptions, and transaction helpers. The page has no backend —
// the ER websocket is the backend.

import {
  ComputeBudgetProgram,
  Connection,
  Keypair,
  Transaction,
  TransactionInstruction,
} from "@solana/web3.js";
import {
  addrs,
  checkpointIx,
  contributeIx,
  decodeGenLog,
  decodeModelHeader,
  decodeWeights,
  generateIx,
  shardPdas,
  DELEGATION_PROGRAM_ID,
  GenEntry,
  ModelHeader,
} from "./pgpt";
import payerSecret from "./payer.json";

export const BASE_URL = import.meta.env.VITE_PGPT_BASE_URL ?? "http://127.0.0.1:7799";
export const ER_URL = import.meta.env.VITE_PGPT_ER_URL ?? "http://127.0.0.1:8899";

export const base = new Connection(BASE_URL, "confirmed");
export const er = new Connection(ER_URL, "confirmed");

// The localnet identity (public development keypair) signs from the browser —
// this is a local showcase, not a wallet integration.
export const payer = Keypair.fromSecretKey(new Uint8Array(payerSecret as number[]));

export interface LiveModel {
  header: ModelHeader;
  weights: Float64Array;
}

export interface BaseState {
  header: ModelHeader | null;
  delegated: boolean;
  updatedAt: number;
}

export function decodeLive(data: Buffer | Uint8Array): LiveModel {
  return { header: decodeModelHeader(data), weights: decodeWeights(data) };
}

/// Subscribe to the live model + genlog on the ER; poll as fallback.
export function watchEr(
  onModel: (m: LiveModel) => void,
  onGen: (entries: GenEntry[], total: bigint) => void,
): () => void {
  const a = addrs();
  let closed = false;

  const pump = async () => {
    try {
      const info = await er.getAccountInfo(a.model);
      if (info) onModel(decodeLive(info.data));
      const gen = await er.getAccountInfo(a.genlog);
      if (gen) {
        const { entries, total } = decodeGenLog(gen.data);
        onGen(entries, total);
      }
    } catch {
      /* ER not up yet */
    }
  };
  void pump();

  const modelSub = er.onAccountChange(a.model, (info) => {
    try {
      onModel(decodeLive(info.data));
    } catch {}
  });
  const genSub = er.onAccountChange(a.genlog, (info) => {
    try {
      const { entries, total } = decodeGenLog(info.data);
      onGen(entries, total);
    } catch {}
  });
  const poll = setInterval(pump, 5000);

  return () => {
    closed = true;
    clearInterval(poll);
    void er.removeAccountChangeListener(modelSub).catch(() => {});
    void er.removeAccountChangeListener(genSub).catch(() => {});
    void closed;
  };
}

/// Poll the base layer: delegation status + the checkpointed image (shards).
export function watchBase(onState: (s: BaseState) => void): () => void {
  const a = addrs();
  const pump = async () => {
    try {
      const model = await base.getAccountInfo(a.model);
      const delegated = model?.owner.equals(DELEGATION_PROGRAM_ID) ?? false;
      let header: ModelHeader | null = null;
      if (delegated) {
        const shards = await Promise.all(shardPdas().map((s) => base.getAccountInfo(s)));
        if (shards.every((s) => s)) {
          try {
            header = decodeModelHeader(Buffer.concat(shards.map((s) => s!.data)));
          } catch {}
        }
      } else if (model) {
        try {
          header = decodeModelHeader(model.data);
        } catch {}
      }
      onState({ header, delegated, updatedAt: Date.now() });
    } catch {
      /* base not up yet */
    }
  };
  void pump();
  const poll = setInterval(pump, 4000);
  return () => clearInterval(poll);
}

async function send(conn: Connection, ixs: TransactionInstruction[]): Promise<string> {
  const tx = new Transaction()
    .add(ComputeBudgetProgram.setComputeUnitLimit({ units: 1_400_000 }))
    .add(...ixs);
  tx.feePayer = payer.publicKey;
  tx.recentBlockhash = (await conn.getLatestBlockhash()).blockhash;
  tx.sign(payer);
  const sig = await conn.sendRawTransaction(tx.serialize(), { skipPreflight: true });
  await conn.confirmTransaction(sig, "confirmed");
  return sig;
}

async function mutatingConn(): Promise<Connection> {
  const info = await base.getAccountInfo(addrs().model);
  return info?.owner.equals(DELEGATION_PROGRAM_ID) ? er : base;
}

export async function sendGenerate(prefix: string, temperature: number): Promise<string> {
  return send(await mutatingConn(), [generateIx(prefix, temperature)]);
}

export async function sendContribute(name: string): Promise<string> {
  const conn = (await base.getAccountInfo(addrs().community))?.owner.equals(
    DELEGATION_PROGRAM_ID,
  )
    ? er
    : base;
  return send(conn, [contributeIx(payer.publicKey, name)]);
}

export async function sendCheckpoint(): Promise<string> {
  // The program refuses to snapshot mid-Adam; retry through update chunks.
  for (let attempt = 0; ; attempt++) {
    try {
      return await send(er, [checkpointIx(payer.publicKey)]);
    } catch (e) {
      if (attempt >= 20) throw e;
      await new Promise((r) => setTimeout(r, 250));
    }
  }
}

// -- Attention lens ----------------------------------------------------------

/// Scratch layout offsets (mirrors gpt_core::Scratch; verified by the
/// program's tests). The generation workspace is the second half.
const SCRATCH_HALF = 86_528;
const ATT_OFFSET = 20_480; // after the 10 BLOCK x N_EMBD activation planes
export const BLOCK = 16;
export const N_HEAD = 4;

/// Attention weights of the generation workspace: att[t][head][s], row-major.
export async function fetchAttention(): Promise<Float64Array | null> {
  const conn = await mutatingConn();
  const info = await conn.getAccountInfo(addrs().scratch);
  if (!info || info.data.length < 2 * SCRATCH_HALF) return null;
  const out = new Float64Array(BLOCK * N_HEAD * BLOCK);
  const buf = Buffer.from(info.data);
  const start = SCRATCH_HALF + ATT_OFFSET;
  for (let i = 0; i < out.length; i++) {
    out[i] = Number(buf.readBigInt64LE(start + i * 8)) / 2 ** 32;
  }
  return out;
}
