#!/usr/bin/env bash
# Boot the p-gpt localnet:
#   - base layer : solana-test-validator on :7799 (ws :7800) with the
#                  delegation stack + p-gpt preloaded
#   - rollup     : ephemeral-validator on :8899, delegating from the base
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ../target/deploy/p_gpt_program.so ]; then
    echo "building p_gpt_program.so..."
    (cd .. && cargo build-sbf --manifest-path program/Cargo.toml)
fi

mkdir -p .run
rm -rf .run/base-ledger .run/er

cleanup() {
    kill "${BASE_PID:-}" "${ER_PID:-}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_rpc() {
    for _ in $(seq 1 120); do
        if curl -sf -X POST -H 'content-type: application/json' \
            -d '{"jsonrpc":"2.0","id":1,"method":"getVersion"}' "$1" 2>/dev/null | grep -q result; then
            return 0
        fi
        sleep 0.5
    done
    echo "warning: timed out waiting for $1 (continuing)" >&2
    return 0
}

echo "starting base layer on :7799..."
solana-test-validator \
    --ledger .run/base-ledger \
    --rpc-port 7799 \
    --reset \
    --quiet \
    --ticks-per-slot 8 \
    --bpf-program DELeGGvXpWV2fqJUhqcF5ZSYMS4JTLjteaAMARRSaeSh elfs/dlp.so \
    --bpf-program DmnRGfyyftzacFb1XadYhWF6vWqXwtQk5tbr6XgR3BA1 elfs/mdp.so \
    --bpf-program ComtrB2KEaWgXsW1dhr1xYL4Ht4Bjj3gXnnL6KMdABq elfs/magicblock_committor_program.so \
    --bpf-program 6wPpJuYKKPbLYfYZpVeytPwxcq7TdGsgEHwyhYBangEC ../target/deploy/p_gpt_program.so \
    --bpf-program noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV elfs/noop.so \
    --url https://api.devnet.solana.com \
    --clone 7JrkjmZPprHwtuvtuGTXp9hwfGYFAQLnLeFM52kqAgXg \
    --clone EpJnX7ueXk7fKojBymqmVuCuwyhDQsYcLVL1XMsBbvDX \
    --clone 8wdZfgo66d6hMMRrcYBDW2K9WtDe2mEG47ynhnYxrAFp \
    >.run/base.log 2>&1 &
BASE_PID=$!
wait_rpc http://127.0.0.1:7799
echo "base layer up."

# Fund the shared localnet identity (payer + ER validator identity, so the
# rollup can pay for commits on the base layer).
solana airdrop 1000 mAGicPQYBMvcYveUZA5F5UNNwyHvfYh5xkLS2Fr1mev -u http://127.0.0.1:7799 >/dev/null
echo "funded mAGicPQYBMvcYveUZA5F5UNNwyHvfYh5xkLS2Fr1mev with 1000 SOL"

echo "starting ephemeral rollup on :8899..."
RUST_LOG=info,magicblock_committor_service=debug,magicblock_task_scheduler=info ephemeral-validator er.toml --no-tui --storage .run/er >.run/er.log 2>&1 &
ER_PID=$!
wait_rpc http://127.0.0.1:8899
echo "ephemeral rollup up."
echo
echo "  base layer : http://127.0.0.1:7799  (logs: localnet/.run/base.log)"
echo "  rollup     : http://127.0.0.1:8899  (logs: localnet/.run/er.log)"
echo
wait
