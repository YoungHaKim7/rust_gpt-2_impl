Done. The complete Rust port is delivered and verified against the original.

## What was built — `gpt2/` (~3,000 lines, new crate at repo root)

A 1:1 port of `llm.c`'s pure-CPU reference, keeping the original's names, loop structure, comments, binary formats, and output:

| llm.c (original)                                              | Port                    |
| ------------------------------------------------------------- | ----------------------- |
| `train_gpt2.c` — 13 layer kernels fwd/bwd, GPT-2 model, AdamW | `src/lib.rs`            |
| `train_gpt2.c` — sampler + training `main()`                  | `src/bin/train_gpt2.rs` |
| `test_gpt2.c`                                                 | `src/bin/test_gpt2.rs`  |
| `llmc/{utils,tokenizer,dataloader,rand}.h`                    | `src/llmc/*.rs`         |

- llm.c's "one big allocation + ~40 raw pointers" design is preserved exactly (`fill_in_*_sizes`, `malloc_and_point_*`), but with safe `(start, len)` tensor views carved out by one `split_disjoint` helper — **100% safe Rust, zero raw pointers** (the vendored `llm.rs` fork's approach was all-unsafe and based on an older llm.c).
- Every `#pragma omp` maps to a rayon iterator at the same spot; per-element summation order is unchanged, so results stay deterministic.
- Checkpoint/tokenizer/data-shard formats are byte-compatible, so real llm.c files work as-is.

## Verification results

- **`make verify`** (differential vs the compiled original C, identical synthetic data): **46/46 loss values bit-identical (max abs diff = 0.0)**, and all non-timing output — including the autoregressively sampled generation text — is **byte-identical**.
- **`cargo test`** (6 tests): mt19937 reproduces PyTorch's documented sequence exactly; `matmul_forward` (unrolled) == naive incl. the `B*T%8` fallback; a finite-difference gradient check of the full loss exercises every backward kernel end to end; tokenizer/dataloader round-trips.
- `cargo clippy --all-targets`: 0 warnings.
- Performance: **39.6 vs 202.2 ms/step** — ~5× the single-threaded C build (rayon on 20 cores).

To train on real GPT-2 124M, run `llm.c/train_gpt2.py` once on a torch machine to produce `gpt2_124M.bin` etc. (documented in `gpt2/README.md`); `test_gpt2` then validates against PyTorch's reference logits/losses/gradients. Omitted by design (documented): CUDA kernels, HellaSwag EvalLoader, distributed training — none used by the CPU reference you pointed at.
