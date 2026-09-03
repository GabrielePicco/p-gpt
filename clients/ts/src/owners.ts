import { Connection } from "@solana/web3.js";
import { addrs, shardPdas, DELEGATION_PROGRAM_ID } from "./pgpt.js";
async function main() {
  const base = new Connection("http://127.0.0.1:7799", "confirmed");
  for (const [n, k] of [...Object.entries(addrs()), ...shardPdas().map((s, i) => [`shard${i}`, s] as const)]) {
    const i = await base.getAccountInfo(k);
    console.log(n.padEnd(10), i ? (i.owner.equals(DELEGATION_PROGRAM_ID) ? "delegated" : i.owner.toBase58().slice(0, 8)) : "none");
  }
}
main();
