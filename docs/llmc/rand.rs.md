# `llmc/rand.rs`

- Thanks for sharing `llmc/mod.rs`, `dataloader.rs`, and `rand.rs`. These files form a nice layered structure for the Rust port of **llm.c's `llmc` support code**.

## Overall module structure

Your `llmc/mod.rs` acts as the module entry point:

```text
llmc/
├── mod.rs          ← Declares the submodules
├── dataloader.rs   ← Reads training-token data shards
├── rand.rs         ← PyTorch-compatible MT19937 RNG
├── tokenizer.rs    ← Token ↔ text conversion
└── utils.rs        ← File and utility helpers
```

```rust
pub mod dataloader;
pub mod rand;
pub mod tokenizer;
pub mod utils;
```

Because the modules are declared with `pub mod`, other parts of the crate can use them, for example:

```rust
use crate::llmc::dataloader::DataLoader;
use crate::llmc::rand::Mt19937State;
```

---

# `rand.rs`: the deterministic random-number foundation

The most important point about this implementation is that it is **not just any random number generator**. It is designed to reproduce the behavior of the original C implementation and PyTorch-compatible results.

The central state is:

```rust
pub struct Mt19937State {
    left_: i32,
    next_: usize,
    state_: [u32; MERSENNE_STATE_N],
    MATRIX_A: [u32; 2],
}
```

Conceptually:

```text
Mt19937State
│
├── state_[624]  ← Internal Mersenne Twister state
├── next_        ← Which random value comes next
├── left_        ← How many values remain before regenerating state
└── MATRIX_A     ← Constants used during state regeneration
```

### `manual_seed`

```rust
manual_seed(&mut state, 137);
```

Initializes the generator to a deterministic state.

That means:

```text
same seed
    ↓
same internal state
    ↓
same sequence of random numbers
    ↓
reproducible training/shuffling
```

This is particularly important for ML training and verification tests.

---

## How `random_permutation` works

This function is directly relevant to `DataLoader`:

```rust
pub fn random_permutation(data: &mut [i32], state: &mut Mt19937State) {
    let numel = data.len();

    for i in (1..numel).rev() {
        let j = (randint32(state) as usize) % (i + 1);
        data.swap(i, j);
    }
}
```

This is the **Fisher–Yates shuffle**.

For example:

```text
Before:
[0, 1, 2, 3, 4]

After:
[3, 0, 4, 1, 2]
```

The important advantage is that it creates a permutation without duplicating or losing elements.

---

# How `dataloader.rs` uses `rand.rs`

The two modules are connected here:

```rust
use super::rand::{
    init_identity_permutation,
    manual_seed,
    random_permutation,
    Mt19937State,
};
```

When shuffling is enabled, the loader maintains two separate permutations:

```rust
pub shard_indices: Vec<i32>,
pub intra_shard_indices: Vec<i32>,
```

They represent two levels of shuffling:

```text
Dataset
│
├── Shard 0
├── Shard 1
├── Shard 2
└── Shard 3

       ↓ shuffle

├── Shard 2
├── Shard 0
├── Shard 3
└── Shard 1
```

Then, inside each shard:

```text
Samples: [0, 1, 2, 3, 4, 5]

       ↓ shuffle

Samples: [4, 1, 5, 0, 3, 2]
```

This is why the loader needs both:

* `shard_indices` → randomizes **which shard is visited**
* `intra_shard_indices` → randomizes **which batch/sample is read within a shard**

---

# `DataLoader` and distributed training

The most interesting part of `dataloader.rs` is that it supports multiple training processes.

Suppose:

```text
B = 2       batch size
T = 4       sequence length
num_processes = 2
```

Each process consumes:

```text
B × T = 8 tokens
```

So all processes together consume:

```text
2 processes × 8 tokens = 16 tokens
```

The code calculates:

```rust
loader.total_batch_size_bytes =
    (num_processes as usize * (B * T)) * 2;
```

The `* 2` exists because each token is a `u16`:

```text
u16 = 2 bytes
```

Each process then gets a different offset:

```rust
loader.local_batch_offset_bytes =
    process_rank as usize * B * T * 2;
```

Conceptually:

```text
Training file
─────────────────────────────────────────────

Process 0: [ batch data ]
Process 1:              [ batch data ]

Next global batch:
Process 0:                         [ batch ]
Process 1:                                   [ batch ]
```

This allows multiple workers to read from the same dataset without training on exactly the same tokens.

---

# Why the loader reads `B * T + 1` tokens

This part is essential for language-model training:

```rust
self.buffer = read_u16s(tokens_file, B * T + 1);
```

Imagine the file contains:

```text
[10, 20, 30, 40, 50]
```

The model input is shifted one token to the left relative to the target:

```text
Inputs:   [10, 20, 30, 40]
Targets:  [20, 30, 40, 50]
```

The code does exactly that:

```rust
for i in 0..B * T {
    self.inputs[i] = self.buffer[i] as i32;
    self.targets[i] = self.buffer[i + 1] as i32;
}
```

Visually:

```text
buffer:   token₀  token₁  token₂  token₃  token₄
            │       │       │       │
inputs:     └───────┴───────┴───────┴──→

targets:           └───────┴───────┴───────┴──→
```

This is the standard **next-token prediction** objective used by GPT-style language models.

---

# The complete relationship between the modules

At a high level, the execution flow looks like this:

```text
                 llmc/mod.rs
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     dataloader     rand       tokenizer
          │           │
          │ uses      │ provides
          └──────────►│
                      │
                      ▼
              Mt19937 random state
                      │
                      ▼
              Shuffle shards/samples
                      │
                      ▼
              Read training tokens
                      │
                      ▼
              inputs + targets
                      │
                      ▼
                Transformer
```

## In short

| File            | Primary responsibility                                         |
| --------------- | -------------------------------------------------------------- |
| `mod.rs`        | Organizes and exposes the `llmc` modules                       |
| `rand.rs`       | Deterministic, PyTorch-compatible random numbers and shuffling |
| `dataloader.rs` | Reads token shards and produces training batches               |
| `tokenizer.rs`  | Converts between tokens and text                               |
| `utils.rs`      | File I/O and common helper functions                           |

The design is particularly good for a **1:1 C-to-Rust port**: `DataLoader` owns its buffers and RNG state, Rust's `Drop` automatically closes the `File` and frees vectors, and the deterministic RNG makes it possible to compare Rust results directly against the original C implementation.

One small Rust-style observation: names such as `B`, `T`, `MATRIX_A`, and functions ending in `_` intentionally preserve the original C naming. Your `#![allow(non_snake_case)]` makes sense for a faithful port, although a more idiomatic Rust redesign would eventually rename them.
