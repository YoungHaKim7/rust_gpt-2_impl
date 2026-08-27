# `llmc/dataloader.rs`

- This `llmc/dataloader.rs` is a Rust implementation of a **data loader for training a language model**. Its job is to read tokenized training data from binary files ("shards") and repeatedly produce batches of **input tokens and target tokens** for the model.

The overall flow is:

```text
Training data files (shards)
          │
          ▼
     DataLoader
          │
          ├── Select shard
          ├── Select batch
          ├── Read B × T + 1 tokens
          │
          ▼
   inputs:  x₀ x₁ x₂ ... xₙ
   targets: x₁ x₂ x₃ ... xₙ₊₁
```

# 1. What are `B` and `T`?

```rust
pub B: usize,
pub T: usize,
```

These are standard LLM training dimensions:

* **B (Batch size)**: number of sequences in a batch
* **T (Sequence length)**: number of tokens per sequence

Therefore, one batch contains:

```text
B × T tokens
```

For example:

```text
B = 4
T = 8

Input batch = 4 × 8 = 32 tokens
```

The loader actually reads:

```rust
B * T + 1
```

tokens because the targets are shifted by one token.

---

# 2. Why read `B * T + 1` tokens?

This is one of the most important parts of the data loader:

```rust
self.buffer = read_u16s(tokens_file, B * T + 1);

for i in 0..B * T {
    self.inputs[i] = self.buffer[i] as i32;
    self.targets[i] = self.buffer[i + 1] as i32;
}
```

Suppose the data contains:

```text
[10, 20, 30, 40, 50]
```

The loader produces:

```text
inputs:  [10, 20, 30, 40]
targets: [20, 30, 40, 50]
```

In other words, the model is trained to predict:

> **Given the current token, predict the next token.**

This is the fundamental training task of GPT-style autoregressive language models.

---

# 3. What is a "shard"?

A training dataset can be too large to store conveniently in one file, so it is divided into multiple pieces:

```text
data/
├── train_000.bin
├── train_001.bin
├── train_002.bin
└── train_003.bin
```

These individual files are called **shards**.

The loader stores them here:

```rust
pub shard_paths: Vec<PathBuf>,
```

and tracks which one it is currently reading:

```rust
pub current_shard_idx: usize,
```

When one shard is exhausted:

```text
Shard 0 ──► Shard 1 ──► Shard 2 ──► ...
                                  │
                                  ▼
                           Start new epoch
```

This behavior is implemented by:

```rust
fn dataloader_advance_(&mut self)
```

---

# 4. The `DataLoader` structure

The struct contains four main categories of information.

### Distributed training

```rust
pub process_rank: i32,
pub num_processes: i32,
```

These allow multiple processes or workers to train simultaneously.

For example:

```text
Dataset
   │
   ├── Worker 0 → Batch A
   ├── Worker 1 → Batch B
   ├── Worker 2 → Batch C
   └── Worker 3 → Batch D
```

Each worker has a unique `process_rank`.

---

### Current position

```rust
pub current_shard_idx: usize,
pub current_sample_idx: usize,
```

These answer:

* Which file am I reading?
* Which batch/sample inside that file am I reading?

---

### File and data buffers

```rust
pub tokens_file: Option<File>,
pub buffer: Vec<u16>,
pub inputs: Vec<i32>,
pub targets: Vec<i32>,
```

The data goes through this pipeline:

```text
Binary file
    ↓
Vec<u16> buffer
    ↓
inputs  Vec<i32>
targets Vec<i32>
```

The source file stores tokens as `u16`, which saves disk space. They are converted to `i32` for use by the training code.

---

### Shuffling

```rust
pub shuffle_rng: Mt19937State,
pub should_shuffle: bool,
pub shard_indices: Vec<i32>,
pub intra_shard_indices: Vec<i32>,
```

There are **two levels of shuffling**:

1. Shuffle the order of shards
2. Shuffle the order of samples inside each shard

For example:

```text
Original shards:
[A, B, C]

After shuffling:
[C, A, B]
```

And within shard A:

```text
Original:
[0, 1, 2, 3]

Shuffled:
[2, 0, 3, 1]
```

This helps prevent the model from learning artifacts caused by the original ordering of the dataset.

---

# 5. `dataloader_load_shard_()` — opening and validating a shard

```rust
fn dataloader_load_shard_(&mut self, shard_index: usize) -> i64
```

This function loads a new data file.

## Step 1: Select the shard

```rust
let shard_index = if self.should_shuffle {
    self.shard_indices[shard_index] as usize
} else {
    shard_index
};
```

If shuffling is enabled, the logical shard number is mapped through the shuffled permutation.

---

## Step 2: Open the file

```rust
self.tokens_file = Some(fopen_check(&filename, "rb"));
```

The `Option<File>` is replaced with a new file handle.

The previous `File`, if any, is automatically dropped and closed by Rust.

This is a nice example of Rust's RAII-style resource management.

---

## Step 3: Validate the header

```rust
let header = read_i32s(tokens_file, HEADER_SIZE);
```

Each data shard begins with a header containing metadata.

The loader checks:

```rust
if header[0] != 20240520
```

This is a **magic number**, used to verify that the file is actually in the expected format.

Then:

```rust
if header[1] != 1
```

checks the file format version.

And:

```rust
let ntok = header[2] as i64;
```

gets the number of tokens stored in the file.

---

# 6. Why check the file size?

```rust
let expected_file_size =
    (HEADER_SIZE * 4 + ntok as usize * 2) as i64;
```

The format is expected to contain:

```text
Header: HEADER_SIZE × 4 bytes
Tokens: ntok × 2 bytes
```

because:

* each header value is an `i32` → 4 bytes
* each token is a `u16` → 2 bytes

The loader verifies:

```rust
if self.file_size_bytes != expected_file_size {
    println!("Error: file size is not as expected");
    exit(1);
}
```

This protects against:

* corrupted files
* incomplete downloads
* incompatible dataset formats

---

# 7. Distributed training: the most interesting calculation

These two fields are particularly important:

```rust
pub total_batch_size_bytes: usize,
pub local_batch_offset_bytes: usize,
```

They are initialized as:

```rust
loader.total_batch_size_bytes =
    (num_processes as usize * (B * T)) * 2;

loader.local_batch_offset_bytes =
    process_rank as usize * B * T * 2;
```

Imagine:

```text
B = 2
T = 4
num_processes = 2
```

Each process needs:

```text
B × T = 8 tokens
```

Each token is 2 bytes:

```text
8 × 2 = 16 bytes per worker
```

So the layout might be:

```text
Dataset tokens:

| Worker 0 batch | Worker 1 batch | Worker 0 batch | Worker 1 batch |
|---- 16 bytes --|---- 16 bytes --|---- 16 bytes --|---- 16 bytes --|
```

Worker 0 starts at offset:

```text
0 bytes
```

Worker 1 starts at:

```text
16 bytes
```

After both workers consume their data, the next global batch starts after:

```text
32 bytes
```

This ensures **different workers don't accidentally train on the same tokens**.

---

# 8. `init()` — constructing the DataLoader

The public constructor is:

```rust
pub fn init(
    filename_pattern: &str,
    B: usize,
    T: usize,
    process_rank: i32,
    num_processes: i32,
    should_shuffle: i32,
) -> DataLoader
```

It performs several jobs:

### ① Creates an empty loader

```rust
let mut loader = DataLoader {
    ...
};
```

### ② Finds all matching data files

```rust
glob::glob(filename_pattern)
```

For example:

```text
data/*.bin
```

might find:

```text
data/001.bin
data/002.bin
data/003.bin
```

### ③ Initializes random shuffling

```rust
manual_seed(
    &mut loader.shuffle_rng,
    (42 + process_rank) as u32,
);
```

Each process gets a different seed.

### ④ Validates every shard

```rust
for shard_index in 0..loader.shard_paths.len() {
    let shard_ntok = loader.dataloader_load_shard_(shard_index);
    ...
}
```

This checks all files before training starts.

### ⑤ Allocates reusable buffers

```rust
loader.buffer = vec![0u16; B * T + 1];
loader.inputs = vec![0i32; B * T];
loader.targets = vec![0i32; B * T];
```

The buffers are allocated once and reused for every batch. This is important for performance because training loops run millions of times.

### ⑥ Calls `reset()`

```rust
loader.reset();
```

The loader is now ready to provide batches.

---

# 9. `next_batch()` — the main function used during training

The training loop will repeatedly call:

```rust
loader.next_batch();
```

Internally:

```rust
pub fn next_batch(&mut self) {
    if self.current_sample_idx >= self.shard_num_samples {
        self.dataloader_advance_();
    }

    self.dataloader_load_batch();
    self.current_sample_idx += 1;
}
```

Conceptually:

```text
┌──────────────────────────────┐
│        next_batch()          │
└──────────────┬───────────────┘
               │
               ▼
      End of current shard?
          │             │
         Yes            No
          │             │
          ▼             │
   Load next shard       │
          │             │
          └──────┬──────┘
                 ▼
          Read B×T+1 tokens
                 │
                 ▼
       Create inputs + targets
                 │
                 ▼
            Return
```

After calling it, the training code can access:

```rust
loader.inputs
loader.targets
```

For example:

```rust
loader.next_batch();

let x = &loader.inputs;
let y = &loader.targets;

// Train the model using x → y
```

---

# 10. What does `resume()` do?

```rust
pub fn resume(
    &mut self,
    current_shard_idx: usize,
    current_sample_idx: usize,
)
```

This is useful when training is interrupted and later resumed.

Suppose training stopped here:

```text
Shard: 17
Batch: 2,543
```

A checkpoint can save those positions. Later:

```rust
loader.resume(17, 2543);
```

allows training to continue from approximately the same point in the dataset.

---

# 11. Why is there no `dataloader_free()`?

The final comment is an important difference between C and Rust:

```rust
// dataloader_free() has no Rust equivalent: everything is freed when dropped
```

In C, you might need:

```c
dataloader_free(&loader);
```

to manually release:

* allocated memory
* file handles

In Rust:

```rust
{
    let loader = DataLoader::init(...);
    // use loader
} // ← loader automatically drops here
```

Rust automatically drops:

```text
DataLoader
 ├── Vec<PathBuf>        → freed
 ├── Option<File>        → file closed
 ├── Vec<u16>            → freed
 ├── Vec<i32>            → freed
 └── other owned values  → dropped
```

This is Rust's ownership and `Drop` system working automatically.

---

## Summary

The most important idea is that `DataLoader` converts a large collection of binary token files into a continuous stream of training batches:

```text
              ┌─────────────────┐
              │  Dataset Shards │
              │ .bin .bin .bin  │
              └────────┬────────┘
                       ▼
                 DataLoader
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
        Select shard          Shuffle
             │                    │
             └─────────┬──────────┘
                       ▼
                 Read tokens
                       │
                       ▼
              B × T + 1 tokens
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Inputs            Targets
          token[i]        token[i + 1]
              │                 │
              └────────┬────────┘
                       ▼
                  GPT Training
```

The design is deliberately **low-level and efficient**, closely matching the original C implementation while using Rust's `Vec`, `File`, `Option`, ownership, and automatic resource cleanup. This makes it a good example of how a performance-oriented C data-loading system can be translated fairly directly into safe Rust.
