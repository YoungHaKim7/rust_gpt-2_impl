# `bin/gen_synth.rs`

- This `bin/gen_synth.rs` program is a **synthetic test-data generator** for your Rust GPT-2 implementation. Its main purpose is not to train a useful language model—it creates **small, deterministic binary files** that allow the Rust implementation and the original `llm.c` implementation to be tested under exactly the same conditions.

## 1. The big picture

The program generates four files:

```text
output_dir/
├── gpt2_124M.bin
├── gpt2_tokenizer.bin
└── dev/data/tinyshakespeare/
    ├── tiny_shakespeare_train.bin
    └── tiny_shakespeare_val.bin
```

The flow is:

```text
                 MT19937 RNG
                 seed = 42
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Model weights   Tokens      Dataset tokens
        │            │            │
        ▼            ▼            ▼
  gpt2_124M.bin  tokenizer.bin  train.bin / val.bin
        │            │            │
        └────────────┼────────────┘
                     ▼
             train_gpt2 program
                     │
                     ▼
              Compare C vs Rust
```

The key word is **deterministic**. Running:

```bash
gen_synth test_data
```

twice should produce the same bytes, assuming the implementation and platform representation match the expected format.

---

# 2. Why is this useful?

Normally, testing a machine-learning training implementation is difficult.

If C and Rust are given different:

* initial weights
* random seeds
* training data
* shuffle order

then different loss values don't necessarily mean that one implementation has a bug.

This program solves that problem:

```text
Same checkpoint
      +
Same tokenizer
      +
Same training data
      +
Same random-number sequence
      ↓
C trainer and Rust trainer
      ↓
Loss trajectories should match
```

So `gen_synth.rs` acts as a **verification fixture generator**.

---

# 3. A deliberately tiny GPT-2 configuration

```rust
const MAX_SEQ_LEN: usize = 64;
const VOCAB_SIZE: usize = 512;
const NUM_LAYERS: usize = 4;
const NUM_HEADS: usize = 4;
const CHANNELS: usize = 128;
```

Despite the output filename being:

```text
gpt2_124M.bin
```

this is **not actually a 124-million-parameter GPT-2 model**.

The filename is fixed because the training program expects that name:

```rust
// gpt2_124M.bin (the fixed name train_gpt2 reads)
```

The actual model is intentionally tiny:

```text
Real GPT-2 124M        Synthetic model
─────────────────      ───────────────
Many layers            4 layers
Large vocabulary       512 tokens
Large hidden size      128 channels
Expensive              Fast to test
```

This is an excellent testing strategy: **keep the file format and algorithm identical while shrinking the dimensions**.

---

# 4. Deterministic random number generation

This section connects directly to your `llmc/rand.rs`:

```rust
let mut rng = Mt19937State::default();
manual_seed(&mut rng, 42);
```

Every subsequent call to:

```rust
randint32(&mut rng)
normal_(..., &mut rng)
```

uses the same MT19937 state.

Importantly, the program uses **one continuous RNG stream**:

```text
seed(42)
   │
   ├── generate model weights
   │
   ├── generate training tokens
   │
   └── generate validation tokens
```

That means changing the order of these operations would change all subsequent generated values. This is intentional: the exact generation order is part of reproducibility.

---

# 5. Generating the model checkpoint

The configuration is assembled as:

```rust
let config = GPT2Config {
    max_seq_len: MAX_SEQ_LEN,
    vocab_size: VOCAB_SIZE,
    padded_vocab_size: VOCAB_SIZE,
    num_layers: NUM_LAYERS,
    num_heads: NUM_HEADS,
    channels: CHANNELS,
};
```

Then:

```rust
fill_in_parameter_sizes(&mut param_sizes, &config);
let num_parameters: usize = param_sizes.iter().sum();
```

This is an important design decision.

Instead of duplicating the GPT-2 parameter-size calculations inside `gen_synth.rs`, it calls the same function used by the model implementation:

```text
GPT-2 configuration
        │
        ▼
fill_in_parameter_sizes()
        │
        ▼
Exact number of parameters
        │
        ▼
Generate that many f32 values
```

This reduces the risk of the generator and model disagreeing about the checkpoint layout.

---

## Checkpoint header

```rust
write_i32_header(
    &mut f,
    &[
        20240326, // magic number
        3,        // version
        config.max_seq_len as i32,
        config.vocab_size as i32,
        config.num_layers as i32,
        config.num_heads as i32,
        config.channels as i32,
        config.padded_vocab_size as i32,
    ],
);
```

This uses the helper you showed earlier:

```rust
write_i32_header()
```

which writes a fixed 256-`i32` header:

```text
┌─────────────────────────────┐
│ Magic                       │
│ Version                     │
│ Model configuration         │
│ ...                         │
│ Zero padding                │
│                             │
│ Total: 256 × 4 = 1024 bytes│
└─────────────────────────────┘
```

The Rust generator is therefore producing the same kind of binary format that the C trainer expects.

---

## Initializing model weights

```rust
let mut params = vec![0.0f32; num_parameters];
normal_(&mut params, 0.0, 0.02, &mut rng);
```

This generates weights approximately distributed as:

$$
w \sim \mathcal{N}(0, 0.02)
$$

That is a common transformer-style initialization scale.

The important verification benefit is that your `normal_()` implementation is the same PyTorch-compatible random implementation from `rand.rs`. Therefore, this generator also tests that part of the port.

Then:

```rust
write_f32s(&mut f, &params);
```

writes the weights directly after the header.

The checkpoint layout is therefore:

```text
gpt2_124M.bin

┌──────────────────────┐
│ 256 × i32 header     │
├──────────────────────┤
│ Parameter tensor 0   │
├──────────────────────┤
│ Parameter tensor 1   │
├──────────────────────┤
│ ...                  │
├──────────────────────┤
│ Parameter tensor N   │
└──────────────────────┘
```

---

# 6. Generating a simple tokenizer

The tokenizer header is:

```rust
header[0] = 20240328;
header[1] = 2;
header[2] = VOCAB_SIZE as u32;
header[3] = (VOCAB_SIZE - 1) as u32;
```

This matches the version-2 format expected by your `Tokenizer::init()`:

```rust
if version == 2 {
    tokenizer.eot_token = header[3] as i32;
}
```

So the generator and reader fit together exactly.

---

## Why generate fake tokens like `aa`, `ba`, etc.?

The code creates short alphabetic strings:

```rust
token.push((b'a' + (i % 26) as u8) as char);
token.push((b'a' + ((i / 26) % 26) as u8) as char);
```

The first tokens will look conceptually like:

```text
aa
ba
ca
...
za
ab
bb
...
```

This is **not a real GPT-2 vocabulary**. That's intentional.

A verification tokenizer only needs to satisfy:

1. The binary format is correct.
2. Every token ID has a valid representation.
3. Generated output is readable by humans.

Using printable tokens makes generation output easier to inspect than arbitrary binary bytes.

The on-disk format is:

```text
Tokenizer header
       │
       ▼
[length][token bytes]
[length][token bytes]
[length][token bytes]
...
```

For example:

```text
02 aa 02 ba 02 ca ...
```

This matches:

```rust
let mut len_buf = [0u8; 1];
fread_check(&mut len_buf, &mut file);
let length = len_buf[0] as usize;
let token_bytes = read_bytes(&mut file, length);
```

in `tokenizer.rs`.

---

# 7. Generating the training datasets

The helper function:

```rust
fn write_tokens_file(
    path: &Path,
    ntok: usize,
    vocab: usize,
    rng: &mut Mt19937State
)
```

creates a binary token file.

First, it writes:

```rust
write_i32_header(&mut f, &[20240520, 1, ntok as i32]);
```

This corresponds exactly to the validation in `DataLoader`:

```rust
if header[0] != 20240520 {
    println!("Bad magic in the data file");
    exit(1);
}

if header[1] != 1 {
    println!("Bad version in data file");
    exit(1);
}

let ntok = header[2];
```

That is a particularly nice property of this project: the writer and reader can be understood side by side.

---

## Token generation

```rust
let tokens: Vec<u16> = (0..ntok)
    .map(|_| (randint32(rng) % vocab as u32) as u16)
    .collect();
```

Since:

```text
VOCAB_SIZE = 512
```

every generated token is in:

```text
0 ≤ token < 512
```

Therefore it is safe for the model vocabulary.

The dataset layout is:

```text
tiny_shakespeare_train.bin

┌──────────────────────────┐
│ 256 × i32 header         │
├──────────────────────────┤
│ token 0 : u16            │
│ token 1 : u16            │
│ token 2 : u16            │
│ ...                      │
└──────────────────────────┘
```

This directly matches `DataLoader`'s expectation:

```rust
let expected_file_size =
    (HEADER_SIZE * 4 + ntok as usize * 2) as i64;
```

So there is a very useful symmetry:

| Generator               | Reader                  |
| ----------------------- | ----------------------- |
| `write_i32_header()`    | `read_i32s()`           |
| `write_u16s()`          | `read_u16s()`           |
| writes magic `20240520` | checks magic `20240520` |
| writes version `1`      | checks version `1`      |

---

# 8. Why are there exactly 20 training batches?

```rust
const TRAIN_NTOK: usize = 4 * 64 * 20;
```

This corresponds to:

```text
B = 4
T = 64
20 batches
```

So:

$$
4 \times 64 \times 20 = 5120
$$

tokens.

Likewise:

```rust
const VAL_NTOK: usize = 4 * 64 * 8;
```

provides eight batches' worth of validation data.

One subtle detail: the `DataLoader` needs `B * T + 1` tokens to construct inputs and shifted targets. Its sample calculation accounts for this boundary requirement.

---

# 9. How this fits with the `llmc` code you showed

All the modules now form a coherent system:

```text
                  gen_synth.rs
                       │
          ┌────────────┼─────────────┐
          │            │             │
          ▼            ▼             ▼
       rand.rs       utils.rs    GPT-2 layout
          │            │
          ▼            ▼
    Random values   Binary files
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
  DataLoader      Tokenizer       GPT-2 checkpoint
        │              │               │
        └──────────────┼───────────────┘
                       ▼
                  train_gpt2
```

### Module responsibilities

| File            | Responsibility                                             |
| --------------- | ---------------------------------------------------------- |
| `rand.rs`       | Reproducible random numbers compatible with the reference  |
| `utils.rs`      | Read/write the binary formats safely                       |
| `dataloader.rs` | Turn token files into `(inputs, targets)` training batches |
| `tokenizer.rs`  | Turn generated token IDs into readable text                |
| `gen_synth.rs`  | Generate small deterministic files for testing everything  |

---

## The most important architectural idea

`gen_synth.rs` is effectively an **integration-testing tool for binary compatibility**.

Unit tests can tell you:

> Does `randint32()` return the right number?

But this generator enables a much stronger test:

> Can the Rust implementation read the same checkpoint and dataset format, perform the same computation, and produce the same training behavior as the original C implementation?

For a project that ports a numerical training system from C to Rust, that is one of the best possible verification strategies.

**In short:** this file is the bridge between *individual unit correctness* and *whole-program compatibility*. It creates a controlled world where differences between the C and Rust trainers are much easier to detect and debug.

