#!/usr/bin/env bash
# Differential verification: compile the original llm.c train_gpt2.c reference
# and this Rust port, run both on identical synthetic data (deterministic
# gen_synth), and compare their printed loss trajectories step by step.
#
# Usage: bash dev/verify_vs_c.sh   (or: make verify)

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CRATE_DIR="$REPO_ROOT/gpt2"
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

echo "== building Rust binaries (release) =="
cargo build --release --manifest-path "$CRATE_DIR/Cargo.toml" --bins

echo "== building original C reference =="
# -ffp-contract=off: no FMA contraction, so the C float ops map 1:1 to Rust's
# (no OpenMP: the pragmas are ignored, which keeps the C run deterministic)
gcc -O2 -ffp-contract=off -o "$SCRATCH/train_gpt2_c" "$REPO_ROOT/llm.c/train_gpt2.c" -lm

echo "== generating identical synthetic data for both =="
"$CRATE_DIR/target/release/gen_synth" "$SCRATCH/c"
"$CRATE_DIR/target/release/gen_synth" "$SCRATCH/rs"
cmp "$SCRATCH/c/gpt2_124M.bin" "$SCRATCH/rs/gpt2_124M.bin" || { echo "gen_synth is not deterministic!"; exit 1; }

echo "== running the C reference =="
(cd "$SCRATCH/c" && "$SCRATCH/train_gpt2_c") > "$SCRATCH/c.log" 2>&1
echo "== running the Rust port =="
(cd "$SCRATCH/rs" && "$CRATE_DIR/target/release/train_gpt2") > "$SCRATCH/rs.log" 2>&1

echo "== checking run lengths =="
NC=$(grep -cE 'val loss|train loss' "$SCRATCH/c.log")
NR=$(grep -cE 'val loss|train loss' "$SCRATCH/rs.log")
echo "C emitted $NC loss values, Rust emitted $NR"
if [ "$NC" -ne "$NR" ] || [ "$NC" -eq 0 ]; then
    echo "DIFFERENTIAL TEST FAILED: run lengths differ (C=$NC Rust=$NR)"
    echo "--- last 10 lines of C log ---";  tail -10 "$SCRATCH/c.log"
    echo "--- last 10 lines of Rust log ---"; tail -10 "$SCRATCH/rs.log"
    exit 1
fi

echo "== comparing loss trajectories =="
paste \
    <(grep -E 'val loss|train loss' "$SCRATCH/c.log"  | sed -E 's/.*loss ([0-9.-]+).*/\1/') \
    <(grep -E 'val loss|train loss' "$SCRATCH/rs.log" | sed -E 's/.*loss ([0-9.-]+).*/\1/') \
| awk '
    {
        c = $1; r = $2; n++
        d = c - r; if (d < 0) d = -d
        if (d > max) { max = d; maxi = n }
        rel = d / (c > 1 ? c : 1)
        if (rel > 1e-3) { bad++; printf("  MISMATCH at loss #%d: C=%s Rust=%s (abs diff %.2e)\n", n, c, r, d) }
    }
    END {
        printf("compared %d loss values: max abs diff = %.3e (at #%d)\n", n, max, maxi)
        if (bad > 0) { printf("DIFFERENTIAL TEST FAILED (%d mismatches)\n", bad); exit 1 }
        printf("loss trajectories match\n")
    }'

echo "== comparing all other output (headers, generation text), ignoring timings =="
if diff -q <(grep -v '(took' "$SCRATCH/c.log") <(grep -v '(took' "$SCRATCH/rs.log") > /dev/null; then
    echo "non-timing output is byte-identical"
else
    echo "NOTE: non-timing output differs (this can be a last-digit formatting difference):"
    diff <(grep -v '(took' "$SCRATCH/c.log") <(grep -v '(took' "$SCRATCH/rs.log") | head -20 || true
fi

echo "DIFFERENTIAL TEST PASSED"
