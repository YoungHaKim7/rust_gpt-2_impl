# gpt2 — llm.c in Rust

A faithful Rust port of [karpathy/llm.c](https://github.com/karpathy/llm.c)'s
pure-CPU reference: **GPT-2 training in simple, readable code** — no PyTorch,
no autograd, just the forward/backward kernels and an AdamW update. This crate
tracks `../llm.c/train_gpt2.c` (the original) 1:1: same function names, same
loop structure, same comments, same binary file formats, same printed output.

## Source mapping

| llm.c (original)                | this port                                |
| ------------------------------- | ---------------------------------------- |
| `train_gpt2.c` (layers + model) | `src/lib.rs`                             |
| `train_gpt2.c` (sampler, main)  | `src/bin/train_gpt2.rs`                  |
| `test_gpt2.c`                   | `src/bin/test_gpt2.rs`                   |
| `llmc/utils.h`                  | `src/llmc/utils.rs`                      |
| `llmc/tokenizer.h`              | `src/llmc/tokenizer.rs`                  |
| `llmc/dataloader.h`             | `src/llmc/dataloader.rs`                 |
| `llmc/rand.h` (mt19937)         | `src/llmc/rand.rs`                       |

Not ported (out of scope for the pure-CPU reference, which doesn't use them):
the CUDA kernels (`*.cu`, `llmc/*.cuh`), the HellaSwag `EvalLoader`,
multi-GPU/distributed training, and the `train_gpt2.cu` CLI flags.

## Quick start

```bash
make            # builds train_gpt2, test_gpt2, gen_synth (release)
make test       # unit tests
make verify     # differential test against the original C (see below)
```

To actually train you need three files in the working directory (same paths as
llm.c): `gpt2_124M.bin`, `gpt2_tokenizer.bin`, and token data under
`dev/data/...`. Ways to get them:

1. **Real GPT-2 124M** (on a machine with PyTorch + internet):
   ```bash
   pip install -r ../llm.c/requirements.txt
   python ../llm.c/train_gpt2.py   # downloads GPT-2, writes gpt2_124M.bin, gpt2_tokenizer.bin,
                                   # and gpt2_124M_debug_state.bin (reference values for test_gpt2)
   python ../llm.c/dev/data/tinyshakespeare/tinyshakespeare.py  # token data
   ./target/release/train_gpt2     # finetune; expect it to match the C version's numbers
   ./target/release/test_gpt2      # checks logits/losses/gradients against PyTorch's
   ```
2. **Synthetic data** (no downloads; for testing only):
   ```bash
   ./target/release/gen_synth <dir> && cd <dir> && <path-to>/train_gpt2
   ```
   A tiny random-init model (867k params) + random tokens; loss hovers around
   ln(512) ≈ 6.24. Enough to exercise every line of the code.

## Verification

- `make test` — self-contained unit tests:
  - mt19937 reproduces the exact PyTorch-documented sequence (seed 137 →
    4053805790, 2173880614, …)
  - `matmul_forward` (unrolled) == `matmul_forward_naive`, including the
    `B*T % 8 != 0` fallback path
  - **finite-difference gradient check** of the entire model: central
    differences of the loss w.r.t. ~100 parameters vs. `gpt2_backward`, which
    exercises every backward kernel end to end
  - tokenizer / dataloader round-trips (shift-by-one targets, epoch wrap)
- `make verify` (`dev/verify_vs_c.sh`) — the strongest check: compiles the
  original `../llm.c/train_gpt2.c` with gcc, generates identical synthetic
  data for both, runs both trainers, and compares. **The Rust port reproduces
  the C loss trajectory bit-for-bit** (46/46 loss values identical, and even
  the sampled generation text is byte-identical).

## How the port stays faithful *and* safe

- llm.c allocates one big block per tensor-group and points ~40 raw pointers
  into it. The port keeps the exact same layout (`fill_in_*_sizes`,
  `malloc_and_point_*`), but tensors are `(start, len)` views (`TensorView`)
  into a `Vec<f32>`, materialized as slices by one safe helper,
  `split_disjoint`, which carves N disjoint ranges out of one buffer with
  `split_at_mut`. No raw pointers anywhere.
- Every `#pragma omp` in the C maps to a rayon parallel iterator at the same
  spot. Parallelism is over disjoint chunks with unchanged per-element
  summation order, so results are deterministic and bit-identical to the
  single-threaded C. (Attention forward parallelizes over `b` rather than
  OpenMP's `collapse(3)` over `(b,t,h)` — coarser but equivalent; matmul, the
  hot path, parallelizes exactly like the C loops.)
- Rust's float math is IEEE-strict by default, which is what the C reference
  needs anyway (llm.c carries `#pragma float_control(precise)` workarounds for
  `-Ofast`; here there is nothing to work around).
- Binary formats are byte-compatible (little-endian): model header magic
  20240326 v3, tokenizer 20240328 v1/v2, data shards 20240520 v1 with uint16
  tokens, debug state 20240327 v2. Checkpoints produced/consumed by llm.c work
  as-is.

## Deviations from the C code (deliberate, small)

- `test_gpt2` exits with code 1 on failure (the C one always returns 0) so it
  can gate CI.
- Error paths print the same messages but without `__FILE__:__LINE__`; unused
  C helpers (`find_max_step`, `ends_with_bin`, `EvalLoader`) are omitted.
- `fclose`/`free`-style cleanup is implicit (Drop).

## Performance

`cargo build --release` (default codegen) is intentionally conservative to
keep results bit-identical to the reference. If you want to trade exactness
for speed, `RUSTFLAGS="-C target-cpu=native" cargo build --release` is safe to
try — on this test setup the default build trains at the same order of
magnitude as `gcc -O2` on the C code.
