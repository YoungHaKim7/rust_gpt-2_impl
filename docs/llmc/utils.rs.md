# `llmc/utils.rs`

- Thanks for sharing the `llmc` module. This is a **clean Rust port of the shared infrastructure from `llm.c`**. The five files work together as the foundation underneath GPT-2 training and generation.

## Overall module structure

```text
llmc/
├── mod.rs          ← Module organization
├── dataloader.rs   ← Reads training tokens from binary data shards
├── rand.rs         ← PyTorch-compatible MT19937 random number generator
├── tokenizer.rs    ← Converts token IDs back into text
└── utils.rs        ← Binary file I/O and validation utilities
```

The dependency relationships look approximately like this:

```text
                    llmc/mod.rs
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    dataloader.rs     tokenizer.rs    rand.rs
          │               │
          ▼               ▼
       utils.rs ◄─────────┘
```

More specifically:

```text
dataloader.rs ──uses──► rand.rs
       │
       └──────────────► utils.rs

tokenizer.rs ─────────► utils.rs
```

---

# 1. `mod.rs`: the module's public interface

```rust
pub mod dataloader;
pub mod rand;
pub mod tokenizer;
pub mod utils;
```

This tells Rust:

> "The `llmc` module consists of these four submodules, and they are publicly accessible."

For example, code elsewhere in the crate can write:

```rust
use crate::llmc::dataloader::DataLoader;
use crate::llmc::tokenizer::Tokenizer;
use crate::llmc::rand::Mt19937State;
```

This file is essentially the **table of contents** for the `llmc` infrastructure.

---

# 2. `DataLoader`: feeding training data to the model

The most important idea in `dataloader.rs` is that a language model is trained on a continuous sequence of tokens.

For example:

```text
[15496, 995, 11, 703, 389, ...]
```

The loader turns these into two sequences:

```text
Input:   [15496, 995, 11, 703]
Target:    [995,  11, 703, 389]
```

The target is simply the input shifted one token to the left.

That is exactly why the loader reads:

```rust
B * T + 1
```

tokens:

```rust
self.buffer = read_u16s(tokens_file, B * T + 1);
```

and then does:

```rust
for i in 0..B * T {
    self.inputs[i] = self.buffer[i] as i32;
    self.targets[i] = self.buffer[i + 1] as i32;
}
```

### Example

If:

```text
buffer = [10, 20, 30, 40, 50]
```

then:

```text
inputs  = [10, 20, 30, 40]
targets = [20, 30, 40, 50]
```

The model learns:

> Given token `10`, predict `20`; given the previous context, predict the next token.

---

## What do `B` and `T` mean?

```rust
pub B: usize,
pub T: usize,
```

Conventionally:

* **B** = batch size
* **T** = sequence length

So if:

```text
B = 4
T = 8
```

one training batch contains:

```text
4 × 8 = 32 tokens
```

The current implementation stores them as flattened arrays:

```rust
inputs:  Vec<i32>  // length B * T
targets: Vec<i32>  // length B * T
```

Conceptually, however, they represent:

```text
[B][T]
```

---

## Distributed training support

These fields are particularly interesting:

```rust
pub process_rank: i32,
pub num_processes: i32,
```

Suppose you have four training processes:

```text
Process 0
Process 1
Process 2
Process 3
```

You don't want every process to train on the same tokens. Each process receives a different section of the global batch.

This calculation:

```rust
loader.total_batch_size_bytes =
    (num_processes as usize * (B * T)) * 2;
```

accounts for:

* all processes
* `B × T` tokens per process
* 2 bytes per `u16` token

And:

```rust
loader.local_batch_offset_bytes =
    process_rank as usize * B * T * 2;
```

moves each process to its own section.

Conceptually:

```text
Data shard
│
├── Batch for process 0
├── Batch for process 1
├── Batch for process 2
└── Batch for process 3
```

This is a nice example of how the data loader supports distributed training **without each worker needing a separate dataset file**.

---

# 3. Shuffling

The loader supports two levels of randomization.

### Shuffle the shards

```rust
random_permutation(&mut self.shard_indices, &mut self.shuffle_rng);
```

For example:

```text
Before: [shard0, shard1, shard2]
After:  [shard2, shard0, shard1]
```

### Shuffle samples within a shard

```rust
self.prepare_intra_shard_indices_();
```

This creates:

```text
[0, 1, 2, 3, 4, ...]
```

and randomly permutes it.

An important design detail is that the loader **doesn't physically shuffle the data file**. Instead, it shuffles an array of indices:

```rust
intra_shard_indices[idx]
```

This is much more efficient because the potentially enormous dataset remains unchanged.

---

# 4. `rand.rs`: deterministic randomness

The random module implements **MT19937 (Mersenne Twister)**.

The important feature here isn't just that it generates random numbers. It is designed to be **numerically compatible with PyTorch**.

```rust
manual_seed(&mut state, 137);
```

means the same seed should produce the same sequence as the reference implementation.

That is extremely useful for a port of `llm.c`, because it allows verification like:

```text
C implementation
      ↓
same seed
      ↓
same random numbers
      ↓
Rust implementation
```

If the random initialization differs, model weights and training behavior can diverge immediately.

---

## `wrapping_mul` is important

This code:

```rust
1812433253u32
    .wrapping_mul(...)
    .wrapping_add(j as u32)
```

is deliberately using wrapping arithmetic.

C code using `uint32_t` naturally performs arithmetic modulo \(2^{32}\). Rust normally checks integer overflow in debug builds, so this port explicitly says:

> "Perform the same 32-bit wrapping behavior."

That's a good example of a Rust port preserving the **numerical semantics of C**, rather than merely translating the syntax.

---

## Fisher-Yates shuffle

```rust
pub fn random_permutation(data: &mut [i32], state: &mut Mt19937State) {
    for i in (1..numel).rev() {
        let j = (randint32(state) as usize) % (i + 1);
        data.swap(i, j);
    }
}
```

This is the classic **Fisher-Yates shuffle**.

Starting with:

```text
[0, 1, 2, 3, 4]
```

it produces a permutation where every element appears exactly once.

This is why the test checks both:

```rust
assert_eq!(s, i as i32);
```

and determinism with the same seed.

---

# 5. `tokenizer.rs`: tokens → text

The tokenizer is intentionally **decode-only**.

```rust
pub fn decode(&self, token_id: u32) -> Option<&[u8]>
```

It performs:

```text
Token ID
   ↓
Vocabulary table
   ↓
Raw bytes
```

For example, conceptually:

```text
15496 → "Hello"
995   → " world"
```

The tokenizer stores bytes rather than Rust `String`s:

```rust
pub token_table: Vec<Vec<u8>>,
```

This is a good choice for GPT-2 tokenization because tokens aren't necessarily valid standalone UTF-8 strings.

---

## Why does `safe_printf` exist?

Some tokens can represent individual control characters:

```text
0x00
0x08  ← backspace
0x1B  ← escape
```

Blindly printing them could cause strange terminal behavior.

So:

```rust
if piece.len() == 1 {
    let byte_val = piece[0];
    let printable = byte_val.is_ascii_graphic()
        || byte_val.is_ascii_whitespace();

    if !printable {
        return;
    }
}
```

filters problematic **single-byte tokens** while allowing normal multi-byte token sequences to be written as raw bytes.

This closely follows the practical behavior of the original C implementation.

---

# 6. `utils.rs`: replacing C-style helper macros

The C implementation apparently uses functions/macros such as:

```c
fopenCheck(...)
freadCheck(...)
fseekCheck(...)
```

In Rust, these become ordinary functions:

```rust
pub fn fopen_check(...)
pub fn fread_check(...)
pub fn fseek_check(...)
```

For example:

```rust
stream.read_exact(buf)
```

is the Rust equivalent of saying:

> Read exactly this many bytes, and fail if you cannot.

The helper then provides the error reporting behavior expected by the original program.

---

## Typed binary reading

These functions are especially important:

```rust
pub fn read_u32s(...)
pub fn read_i32s(...)
pub fn read_f32s(...)
pub fn read_u16s(...)
```

They bridge the gap between a binary file and typed Rust values.

For example:

```rust
bytemuck::cast_slice(&read_bytes(stream, n * 4))
    .to_vec()
```

Conceptually:

```text
Raw bytes from file
        ↓
[0x01, 0x00, 0x00, 0x00, ...]
        ↓
reinterpret as u32/i32/f32
        ↓
Vec<T>
```

This is analogous to C code using `fread()` directly into an array of a particular type.

### One thing to keep in mind

This approach assumes the file's byte representation matches the native representation of the machine. In practice, the `llm.c` data format and typical little-endian systems make that appropriate for this direct port, but a more portable serialization format would explicitly specify endianness with functions such as `u32::from_le_bytes`.

---

# The complete training data flow

Putting all the files together, the architecture looks like this:

```text
                Training data files
                       │
                       ▼
              ┌─────────────────┐
              │   DataLoader    │
              │ dataloader.rs   │
              └────────┬────────┘
                       │
             B × T input tokens
             B × T target tokens
                       │
                       ▼
                 GPT-2 model
                       │
                       ▼
                   Loss / training


Randomness ──► rand.rs ──► shuffling / initialization


Generated token IDs
        │
        ▼
 tokenizer.rs
        │
        ▼
      Text output


Binary file operations
        │
        ▼
     utils.rs
```

## Overall assessment

This is a fairly faithful Rust design for a C codebase:

* **`DataLoader`** replaces manually managed C structs and file state with a Rust struct.
* **`Drop` automatically replaces `*_free()` functions**, eliminating manual cleanup.
* **`Vec<T>` replaces `malloc`/`free` buffers.**
* **Explicit `wrapping_*` operations preserve C numerical behavior where necessary.**
* The public API remains structurally similar to the original `llm.c` API, making the port easier to compare and verify.

The strongest part of this design is that it keeps the original algorithm recognizable while using Rust ownership naturally—for example, `DataLoader` owns its buffers and optionally owns the currently open `File`, so there is no separate `dataloader_free()` to forget to call.
