// p-gpt CLI — drives the full perpetual-training lifecycle on a localnet
// (or any base + ER pair).
//
//   npm run cli -- setup [names]     create accounts, init weights, load dataset
//   npm run cli -- delegate          delegate model/optimizer/scratch/community/genlog to the ER
//   npm run cli -- train <n>         push n manual train steps (works on base pre-delegation, ER after)
//   npm run cli -- schedule [ms] [steps_per_tick]   start the perpetual crank on the ER
//   npm run cli -- status            model header from base + ER
//   npm run cli -- watch             live loss / babble stream from the ER
//   npm run cli -- generate [prefix] sample a name (ER when delegated)
//   npm run cli -- contribute <name> add a name to the community dataset
//   npm run cli -- checkpoint        commit model+genlog state to base
//   npm run cli -- undelegate        commit and undelegate everything
//   npm run cli -- babble            print the generation log

import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  TransactionInstruction,
  ComputeBudgetProgram,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as url from "node:url";
import {
  addrs,
  checkpointIx,
  contributeIx,
  decodeDocsCount,
  decodeGenLog,
  decodeModelHeader,
  delegateIx,
  delegatePrepIx,
  delegationAccounts,
  generateIx,
  growIx,
  initModelIx,
  initShardsIx,
  initWeightsIx,
  loadDocsIx,
  scheduleTrainingIx,
  trainMicroIx,
  undelegateIx,
  DELEGATION_PROGRAM_ID,
  shardPdas,
  N_PARAMS,
  Which,
} from "./pgpt.js";

const BASE_URL = process.env.PGPT_BASE_URL ?? "http://127.0.0.1:7799";
const ER_URL = process.env.PGPT_ER_URL ?? "http://127.0.0.1:8899";
const SEED = BigInt(process.env.PGPT_SEED ?? "42");

const here = path.dirname(url.fileURLToPath(import.meta.url));
// The localnet validator identity: funded with u64::MAX/2 at genesis on both
// the base layer and the ER, so it pays for everything locally (the airdrop
// faucet is disabled in current validator builds). Override with PGPT_PAYER.
const KEYPAIR_PATH =
  process.env.PGPT_PAYER ?? path.join(here, "../../../localnet/payer.json");

const base = new Connection(BASE_URL, "confirmed");
const er = new Connection(ER_URL, "confirmed");

function payer(): Keypair {
  return Keypair.fromSecretKey(new Uint8Array(JSON.parse(fs.readFileSync(KEYPAIR_PATH, "utf8"))));
}

async function ensureFunded(kp: Keypair) {
  const balance = await base.getBalance(kp.publicKey);
  if (balance >= 10e9) return;
  const sig = await base.requestAirdrop(kp.publicKey, 1000e9).catch(() => null);
  if (sig) {
    await base.confirmTransaction(sig, "confirmed");
    return;
  }
  throw new Error(
    `payer ${kp.publicKey.toBase58()} has only ${balance} lamports on base and the faucet is unavailable`,
  );
}

async function send(
  conn: Connection,
  kp: Keypair,
  ixs: TransactionInstruction[],
  computeUnits?: number,
): Promise<string> {
  const tx = new Transaction();
  if (computeUnits) {
    tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: computeUnits }));
  }
  for (const ix of ixs) tx.add(ix);
  return sendAndConfirmTransaction(conn, tx, [kp], {
    skipPreflight: true,
    commitment: "confirmed",
  });
}

/// Is this PDA currently delegated (base-layer owner = delegation program)?
async function isDelegated(account: PublicKey): Promise<boolean> {
  const info = await base.getAccountInfo(account);
  return info?.owner.equals(DELEGATION_PROGRAM_ID) ?? false;
}

/// The connection that can currently mutate the model.
async function trainingConn(): Promise<Connection> {
  return (await isDelegated(addrs().model)) ? er : base;
}

function loadNames(limit: number): string[] {
  const raw = fs.readFileSync(path.join(here, "../../../reference/names.txt"), "utf8");
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && l.length <= 15 && /^[a-z]+$/.test(l))
    .slice(0, limit);
}

async function setup(nameCount: number) {
  const kp = payer();
  await ensureFunded(kp);
  const a = addrs();

  if (!(await base.getAccountInfo(a.model))) {
    console.log("init model:", await send(base, kp, [initModelIx(kp.publicKey, SEED)]));
  }
  if (!(await base.getAccountInfo(shardPdas()[0]))) {
    console.log("init shards:", await send(base, kp, [initShardsIx(kp.publicKey)]));
  }

  // Grow every account to its full size (10KB per call, idempotent).
  for (let which = 0 as Which; which <= 5; which++) {
    let previous = -1;
    for (;;) {
      const info = await base.getAccountInfo(Object.values(a)[which]);
      const len = info?.data.length ?? 0;
      if (len === previous) break;
      previous = len;
      await send(base, kp, [growIx(kp.publicKey, which as Which)]);
    }
    console.log(`grow[${which}] -> ${previous} bytes`);
  }

  // Weights.
  for (;;) {
    const info = await base.getAccountInfo(a.model);
    const header = decodeModelHeader(info!.data);
    if (header.flags & 1) break;
    await send(base, kp, [initWeightsIx(1024)], 1_400_000);
    console.log(`init weights: ${header.initCursor}/${N_PARAMS}`);
  }
  console.log("weights ready");

  // Dataset.
  const names = loadNames(nameCount);
  const loaded = Number(decodeDocsCount((await base.getAccountInfo(a.dataset))!.data));
  const chunk = 48;
  for (let i = loaded; i < names.length; i += chunk) {
    await send(base, kp, [loadDocsIx(kp.publicKey, names.slice(i, i + chunk))]);
    if ((i / chunk) % 10 === 0) console.log(`docs ${Math.min(i + chunk, names.length)}/${names.length}`);
  }
  console.log(`dataset loaded: ${names.length} names`);
}

async function delegate() {
  const kp = payer();
  await ensureFunded(kp);
  // Everything but the (read-only) dataset; 6..9 are the checkpoint shards.
  for (const which of [0, 1, 2, 4, 5, 6, 7, 8, 9] as Which[]) {
    const account = which >= 6 ? shardPdas()[which - 6] : Object.values(addrs())[which];
    if (await isDelegated(account)) {
      console.log(`delegate[${which}]: already delegated`);
      continue;
    }
    // Pre-create the delegate buffer (10KB per tx) for large accounts only;
    // small accounts go through the SDK path which creates its own buffer.
    const target = (await base.getAccountInfo(account))!.data.length;
    if (target > 10_240) {
      const { buffer } = delegationAccounts(account);
      for (;;) {
        const len = (await base.getAccountInfo(buffer))?.data.length ?? 0;
        if (len >= target) break;
        await send(base, kp, [delegatePrepIx(kp.publicKey, which)]);
      }
    }
    // Delegate specifically to the local ER's validator identity.
    console.log(
      `delegate[${which}]:`,
      await send(base, kp, [delegateIx(kp.publicKey, which, kp.publicKey)], 1_400_000),
    );
  }
}

async function train(steps: number) {
  const kp = payer();
  await ensureFunded(kp);
  const conn = await trainingConn();
  // Split path: each SGD step is a short sequence of sub-1.4M CU
  // transactions (forward, backward x4-positions, adam x8-chunks).
  const start = decodeModelHeader((await conn.getAccountInfo(addrs().model))!.data).step;
  const target = start + BigInt(steps);
  let h = null;
  do {
    await send(conn, kp, [trainMicroIx()], 1_400_000);
    h = decodeModelHeader((await conn.getAccountInfo(addrs().model))!.data);
  } while (h.step < target);
  console.log(`step ${h.step} | loss ${h.lastLoss.toFixed(4)} | ema ${h.lossEma.toFixed(4)}`);
}

async function schedule(intervalMs: number, stepsPerTick: number) {
  const kp = payer();
  await ensureFunded(kp);
  const taskId = BigInt(Date.now());
  const sig = await send(er, kp, [
    scheduleTrainingIx(kp.publicKey, taskId, BigInt(intervalMs), BigInt("9223372036854775807"), stepsPerTick),
  ]);
  console.log(`perpetual training crank scheduled (task ${taskId}, every ${intervalMs}ms): ${sig}`);
}

async function status() {
  for (const [label, conn] of [
    ["base", base],
    ["  er", er],
  ] as const) {
    let info = await conn.getAccountInfo(addrs().model).catch(() => null);
    if (!info) {
      console.log(`${label}: no model account`);
      continue;
    }
    const owner = info.owner.equals(DELEGATION_PROGRAM_ID) ? "delegated" : info.owner.toBase58();
    if (label === "base" && info.owner.equals(DELEGATION_PROGRAM_ID)) {
      // While delegated, the checkpointed image lives in the shards.
      const shards = await Promise.all(shardPdas().map((s) => conn.getAccountInfo(s)));
      if (shards.every((s) => s)) {
        info = { ...info, data: Buffer.concat(shards.map((s) => s!.data)) } as any;
      }
    }
    try {
      const h = decodeModelHeader(info.data);
      console.log(
        `${label}: step ${h.step} | loss ${h.lastLoss.toFixed(4)} | ema ${h.lossEma.toFixed(4)} | gens ${h.genCount} | owner ${owner}`,
      );
    } catch {
      console.log(`${label}: owner ${owner} (undecodable — buffered?)`);
    }
  }
}

async function watch() {
  console.log("watching model on the ER (ctrl-c to stop)...");
  let lastStep = -1n;
  er.onAccountChange(addrs().model, (info) => {
    try {
      const h = decodeModelHeader(info.data);
      if (h.step !== lastStep) {
        lastStep = h.step;
        console.log(`step ${h.step} | loss ${h.lastLoss.toFixed(4)} | ema ${h.lossEma.toFixed(4)}`);
      }
    } catch {}
  });
  er.onAccountChange(addrs().genlog, (info) => {
    try {
      const { entries } = decodeGenLog(info.data);
      const last = entries[entries.length - 1];
      if (last) console.log(`   ✨ "${last.name}" (step ${last.step})`);
    } catch {}
  });
  await new Promise(() => {});
}

async function generate(prefix: string) {
  const kp = payer();
  await ensureFunded(kp);
  const conn = await trainingConn();
  await send(conn, kp, [generateIx(prefix)], 2_000_000);
  const { entries } = decodeGenLog((await conn.getAccountInfo(addrs().genlog))!.data);
  const last = entries[entries.length - 1];
  console.log(`"${last?.name}" (step ${last?.step})`);
}

async function contribute(name: string) {
  const kp = payer();
  await ensureFunded(kp);
  const conn = (await isDelegated(addrs().community)) ? er : base;
  console.log("contribute:", await send(conn, kp, [contributeIx(kp.publicKey, name)]));
}

async function checkpoint() {
  const kp = payer();
  await ensureFunded(kp);
  // The program refuses to snapshot mid-Adam (weights would be torn), so a
  // few retries ride out the crank's update chunks.
  for (let attempt = 0; ; attempt++) {
    try {
      console.log("checkpoint:", await send(er, kp, [checkpointIx(kp.publicKey)]));
      return;
    } catch (e) {
      if (attempt >= 20) throw e;
      await new Promise((r) => setTimeout(r, 250));
    }
  }
}

async function undelegate() {
  const kp = payer();
  await ensureFunded(kp);
  console.log("undelegate:", await send(er, kp, [undelegateIx(kp.publicKey)]));
}

async function babble() {
  const conn = await trainingConn();
  const { total, entries } = decodeGenLog((await conn.getAccountInfo(addrs().genlog))!.data);
  console.log(`${total} generations:`);
  for (const e of entries.slice(-30)) console.log(`  step ${e.step}: ${e.name}`);
}

const [cmd, ...args] = process.argv.slice(2);
const run: Record<string, () => Promise<void>> = {
  setup: () => setup(Number(args[0] ?? 4096)),
  delegate,
  train: () => train(Number(args[0] ?? 1)),
  schedule: () => schedule(Number(args[0] ?? 250), Number(args[1] ?? 1)),
  status,
  watch,
  generate: () => generate(args[0] ?? ""),
  contribute: () => contribute(args[0]!),
  checkpoint,
  undelegate,
  babble,
};

if (!cmd || !(cmd in run)) {
  console.error("usage: cli <setup|delegate|train|schedule|status|watch|generate|contribute|checkpoint|undelegate|babble>");
  process.exit(2);
}
run[cmd]().then(() => process.exit(0), (e) => {
  console.error(e);
  process.exit(1);
});
