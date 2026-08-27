# Run


```bash
$ make
cargo build --release --bin train_gpt2
    Finished `release` profile [optimized] target(s) in 0.01s
cargo build --release --bin test_gpt2
    Finished `release` profile [optimized] target(s) in 0.01s
cargo build --release --bin gen_synth
    Finished `release` profile [optimized] target(s) in 0.01s
```

- `make test`

```bash
$ make test
cargo test --release
   Compiling gpt2 v0.1.0 (/home/gygy/my_projects/C_Lang/rust_gpt-2_impl/gpt2)
    Finished `release` profile [optimized] target(s) in 0.06s
     Running unittests src/lib.rs (target/release/deps/gpt2-efad2e91cc470f06)

running 2 tests
test llmc::rand::tests::mt19937_matches_torch_reference ... ok
test llmc::rand::tests::random_permutation_is_deterministic_and_bijective ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/gen_synth.rs (target/release/deps/gen_synth-f0600908f607900d)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/test_gpt2.rs (target/release/deps/test_gpt2-313dd43eef96e67e)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/train_gpt2.rs (target/release/deps/train_gpt2-9f7e5ed6268c6256)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/unit.rs (target/release/deps/unit-e7e44cb6ded71bf8)

running 4 tests
test dataloader_serves_shifted_batches_and_wraps ... ok
test tokenizer_roundtrip ... ok
test matmul_forward_tiled_matches_naive ... ok
test finite_difference_gradient_check ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s

   Doc-tests gpt2

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

- `make verify`
```bash

$ make verify
bash dev/verify_vs_c.sh
== building Rust binaries (release) ==
    Finished `release` profile [optimized] target(s) in 0.01s
== building original C reference ==
== generating identical synthetic data for both ==
wrote /tmp/tmp.l7vNC75dKy/c/gpt2_124M.bin (867072 params)
wrote /tmp/tmp.l7vNC75dKy/c/gpt2_tokenizer.bin
wrote 5120 train + 2048 val tokens under /tmp/tmp.l7vNC75dKy/c/dev/data/tinyshakespeare
wrote /tmp/tmp.l7vNC75dKy/rs/gpt2_124M.bin (867072 params)
wrote /tmp/tmp.l7vNC75dKy/rs/gpt2_tokenizer.bin
wrote 5120 train + 2048 val tokens under /tmp/tmp.l7vNC75dKy/rs/dev/data/tinyshakespeare
== running the C reference ==
== running the Rust port ==
== checking run lengths ==
C emitted 46 loss values, Rust emitted 46
== comparing loss trajectories ==
compared 46 loss values: max abs diff = 0.000e+00 (at #0)
loss trajectories match
== comparing all other output (headers, generation text), ignoring timings ==
non-timing output is byte-identical
DIFFERENTIAL TEST PASSED
```

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
