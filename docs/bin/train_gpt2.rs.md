# `bin/train_gpt2.rs`

- `bin/train_gpt2.rs` is the **actual GPT-2 training program** in your Rust port. While `test_gpt2.rs` verifies that the mathematical implementation matches the reference, this program puts all the pieces together and performs a small real training run.

## Overall structure

```text
                    gpt2_124M.bin
                          │
                          ▼
                    GPT2 model
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼
   Training DataLoader                Validation DataLoader
          │                                │
          ▼                                ▼
     Training batches                Validation loss
          │
          ▼
   Forward → Backward → Update
          │
          ├──────────────► Print training loss
          │
          └──────────────► Occasionally generate text
```

---

# 1. The sampler: choosing the next token

The first part implements a small random-number generator and a probability sampler.

### `random_u32`

```rust
fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
}
```

This is a **xorshift-based pseudorandom number generator**. The mutable `state` means every call changes the RNG state:

```text
state₀ → random number → state₁ → random number → state₂ → ...
```

The important difference from `llmc/rand.rs` is that this RNG is used for **sampling generated text**, whereas `Mt19937State` is used where compatibility with the reference implementation matters, such as initialization and data shuffling.

---

### `sample_mult`

```rust
fn sample_mult(probabilities: &[f32], n: usize, coin: f32) -> usize {
    let mut cdf = 0.0f32;
    for i in 0..n {
        cdf += probabilities[i];
        if coin < cdf {
            return i;
        }
    }
    n - 1
}
```

This performs **multinomial sampling** using a cumulative distribution function (CDF).

For example, suppose the model predicts:

```text
Token A: 0.50
Token B: 0.30
Token C: 0.20
```

The cumulative ranges become:

```text
0.0        0.5       0.8       1.0
│ Token A  │ Token B  │ Token C │
```

A random number such as `0.65` selects **Token B**.

This is different from always selecting the largest probability:

```text
argmax → always choose the most likely token
sample_mult → randomly choose, weighted by probabilities
```

The second approach produces more varied generated text.

---

# 2. Loading the GPT-2 model

```rust
let mut model = GPT2::build_from_checkpoint("gpt2_124M.bin");
```

The model begins from a checkpoint rather than being constructed with completely random parameters inside this program.

In your project, `gen_synth.rs` can generate a small compatible synthetic checkpoint for verification purposes, while the normal training workflow can use the GPT-2 checkpoint expected by the original `llm.c` format.

---

# 3. Selecting the dataset

The program supports two datasets:

```rust
let tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
let tiny_shakespeare_train =
    "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
```

It prefers Tiny Shakespeare when its files exist:

```rust
let train_tokens = if Path::new(tiny_shakespeare_train).exists() {
    tiny_shakespeare_train
} else {
    tiny_stories_train
};
```

So the logic is:

```text
Tiny Shakespeare available?
        │
       Yes ──► Use Tiny Shakespeare
        │
       No
        ▼
Use TinyStories
```

---

# 4. Creating the `DataLoader`s

```rust
let B: usize = 4;
let T: usize = 64;

let mut train_loader = DataLoader::init(train_tokens, B, T, 0, 1, 1);
let mut val_loader = DataLoader::init(val_tokens, B, T, 0, 1, 0);
```

These values mean:

* **`B = 4`** → four independent sequences per batch
* **`T = 64`** → 64 tokens per sequence

Therefore, each batch contains:

```text
B × T = 4 × 64 = 256 training token positions
```

The training loader has shuffling enabled:

```rust
should_shuffle = 1
```

while the validation loader does not:

```rust
should_shuffle = 0
```

This is a sensible distinction:

```text
Training:   shuffle data → improve training behavior
Validation: fixed order   → consistent measurement
```

Your `DataLoader` implementation also supports distributed training parameters (`process_rank` and `num_processes`). Here:

```rust
DataLoader::init(..., 0, 1, ...)
```

means **rank 0 of one process**, so this particular training program is running in a single-process configuration.

---

# 5. The validation loop

Every 10 steps:

```rust
if step % 10 == 0 {
```

the program resets the validation loader and evaluates five batches:

```rust
val_loader.reset();

for _ in 0..val_num_batches {
    val_loader.next_batch();
    model.gpt2_forward(
        &val_loader.inputs,
        Some(&val_loader.targets),
        B,
        T,
    );
    val_loss += model.mean_loss;
}
```

Notice that validation performs only:

```text
Forward pass → calculate loss
```

There is no:

```text
Backward pass
Optimizer update
```

because validation is measuring the model rather than training it.

The average is then printed:

```text
val loss X.XXXXXX
```

---

# 6. Text generation during training

Every 20 steps, except step zero:

```rust
if step > 0 && step % 20 == 0 {
```

the program generates a sample.

### Starting with the EOT token

```rust
for tok in gen_tokens.iter_mut() {
    *tok = tokenizer.eot_token;
}
```

Initially, the entire generation buffer contains the special **end-of-text (EOT)** token.

Then the program generates tokens one at a time:

```text
[EOT] → predict token 1
[EOT, token 1] → predict token 2
[EOT, token 1, token 2] → predict token 3
...
```

This is called **autoregressive generation**.

---

## Why is `gpt2_forward()` called repeatedly?

Inside the loop:

```rust
for t in 1..genT {
    model.gpt2_forward(&gen_tokens, None, B, T);
    // select a token
    gen_tokens[t] = next_token as i32;
}
```

GPT-2 needs the previous tokens to predict the next one. Therefore, after adding a new token, the model is run again to obtain the next probability distribution.

The code itself notes that this is inefficient because it recalculates the whole `(B, T)` sequence every time.

A more optimized transformer inference implementation would use a **KV cache**, avoiding recomputation of previous attention keys and values. But for a training sanity check, this simple approach is easier to implement and verify.

---

# 7. Why `Vp` instead of `V`?

```rust
let Vp = model.config.padded_vocab_size;
```

The activation buffer may use a padded vocabulary size for memory layout or computational reasons.

The code gets the probability vector using:

```rust
let probs = model.acts.unwrap().probs.slice(
    model.acts_memory.as_ref().unwrap(),
    (t - 1) * Vp,
    Vp,
);
```

But when sampling, it deliberately considers only the real vocabulary:

```rust
let next_token =
    sample_mult(probs, model.config.vocab_size, coin);
```

So:

```text
V  = actual vocabulary
Vp = padded vocabulary storage size

Sample only from V
Ignore the padding
```

This prevents nonexistent padding entries from becoming generated tokens.

---

# 8. The actual training step

This is the core of the entire program:

```rust
train_loader.next_batch();

model.gpt2_forward(
    &train_loader.inputs,
    Some(&train_loader.targets),
    B,
    T,
);

model.gpt2_zero_grad();
model.gpt2_backward();

model.gpt2_update(...);
```

Conceptually:

### Step ① Get training data

```text
DataLoader
    ↓
inputs:  [token₀, token₁, token₂, ...]
targets: [token₁, token₂, token₃, ...]
```

The `DataLoader` shifts the tokens by one position to create the input/target relationship.

### Step ② Forward pass

```text
Input tokens
     ↓
GPT-2 Transformer
     ↓
Token probabilities
     ↓
Compare with target tokens
     ↓
Loss
```

### Step ③ Backward pass

```text
Loss
 ↓
Backpropagation
 ↓
Gradients for every parameter
```

### Step ④ Update parameters

```rust
model.gpt2_update(
    1e-4f32,
    0.9f32,
    0.999f32,
    1e-8f32,
    0.0f32,
    (step + 1) as i32,
);
```

These parameters correspond to an Adam/AdamW-style optimizer configuration:

| Parameter     |   Value | Purpose                               |
| ------------- | ------: | ------------------------------------- |
| Learning rate |  `1e-4` | Size of parameter updates             |
| Beta 1        |   `0.9` | First-moment averaging                |
| Beta 2        | `0.999` | Second-moment averaging               |
| Epsilon       |  `1e-8` | Numerical stability                   |
| Weight decay  |   `0.0` | No weight decay in this training loop |

---

# The complete training loop in one picture

```text
for each training step
│
├── Every 10 steps
│     └── Validation
│           └── Forward → Loss
│
├── Every 20 steps
│     └── Generate sample text
│           └── Forward → Sample token → Repeat
│
└── Every step
      ├── DataLoader.next_batch()
      ├── GPT-2 forward pass
      ├── Calculate loss
      ├── Zero gradients
      ├── Backward pass
      ├── AdamW update
      └── Print training loss and timing
```

## Relationship to the other files you provided

At this point, the architecture of your Rust port is nicely connected:

```text
                    train_gpt2.rs
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        GPT2        DataLoader       Tokenizer
          │              │              │
          │              ▼              ▼
          │          token files    generated text
          │
          ▼
Forward / Backward / Update
          │
          ▼
      Model training


test_gpt2.rs
     │
     └── Verifies that GPT2's calculations match
         the reference implementation


gen_synth.rs
     │
     └── Creates deterministic small test inputs
         for development and verification
```

### The key takeaway

**`train_gpt2.rs` is the integration point of the project.** `dataloader.rs` provides data, the GPT-2 implementation performs learning, `tokenizer.rs` displays generated output, and the sampler converts the model's probability distribution into actual token choices.

In other words, this file answers the question:

> **"Now that we have implemented all the individual GPT-2 components in Rust, how do we actually train and observe the model?"**
