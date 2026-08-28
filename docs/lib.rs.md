# 코드 구성


```bash

├── dev
│  └── verify_vs_c.sh
├── llm.c
├── Makefile
├── src
│  ├── bin
│  │  ├── gen_synth.rs
│  │  ├── test_gpt2.rs
│  │  └── train_gpt2.rs
│  ├── lib.rs
│  └── llmc
│     ├── dataloader.rs
│     ├── mod.rs
│     ├── rand.rs
│     ├── tokenizer.rs
│     └── utils.rs
└── tests
   └── unit.rs

```

# 공부 참고할 자료
- Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20 #481 | karpathy started this conversation in General | karpathy | on May 29, 2024
  - https://github.com/karpathy/llm.c/discussions/481

# `lib.rs`

Excellent. This is essentially a **complete GPT-2 training engine written in safe Rust**. It implements the major pieces needed to train GPT-2 from scratch:

> **Tokens → Embeddings → Transformer blocks → Logits → Softmax → Loss → Backpropagation → AdamW update**

Because this file is very large, explaining literally every line in one response would become difficult to follow. So I'll explain it **in the order the program works**, while still examining the important Rust syntax and individual operations closely. We can then zoom into any function in later messages.

# 1. What is `lib.rs`?

`lib.rs` is the conventional root module of a Rust **library crate**.

```rust
pub mod llmc;
```

This declares a public module named `llmc`. Code outside this crate can access its public items.

```rust
use rayon::prelude::*;
```

This imports Rayon traits needed for parallel iteration, such as:

```rust
.par_chunks_exact_mut(...)
.for_each(...)
```

Rayon is the Rust equivalent of much of what OpenMP provides in the original C implementation.

---

# 2. The design philosophy of this implementation

The opening comment is extremely important:

```text
Where the C code carves all tensors out of one big allocation with raw pointers,
this port keeps the exact same layout but tracks tensors as (start, len) views
into a single Vec
```

The original C implementation probably looks conceptually like this:

```c
float *memory = malloc(total_size * sizeof(float));

float *wte = memory + offset1;
float *wpe = memory + offset2;
float *ln1w = memory + offset3;
```

This Rust implementation keeps the **same memory layout**, but avoids raw pointers.

Instead:

```rust
Vec<f32>
```

owns one large allocation, and each tensor is represented by:

```rust
TensorView {
    start: ...,
    len: ...,
}
```

So conceptually:

```text
┌──────────────────────────────────────────────┐
│              Vec<f32>                       │
├──────────┬──────────┬──────────┬────────────┤
│   wte    │   wpe    │   ln1w   │    ...     │
└──────────┴──────────┴──────────┴────────────┘
     ▲          ▲          ▲
 TensorView  TensorView  TensorView
```

This is one of the most interesting Rust design decisions in the entire program.

---

# 3. Compiler attributes

```rust
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
```

The `#![...]` syntax is an **inner attribute**, applying to the entire crate.

### `non_snake_case`

Rust normally prefers:

```rust
let batch_size = 32;
```

But machine-learning code conventionally uses mathematical names:

```rust
B, T, C, V
```

So this warning is disabled.

### `too_many_arguments`

Functions such as matrix multiplication naturally require many parameters:

```rust
out, inp, weight, bias, B, T, C, OC
```

Clippy would normally suggest grouping them into a struct, but doing so might make this low-level numerical code less readable.

---

# 4. Tensor dimensions: `(B, T, C)`

The comment says:

```rust
// B = batch_size, T = sequence_length, C = channels, V = vocab_size
```

These symbols appear everywhere:

| Symbol | Meaning                        | Example     |
| ------ | ------------------------------ | ----------- |
| `B`    | Batch size                     | 4 sentences |
| `T`    | Sequence length                | 128 tokens  |
| `C`    | Channels / embedding dimension | 768         |
| `V`    | Vocabulary size                | 50,257      |
| `NH`   | Number of attention heads      | 12          |
| `L`    | Number of Transformer layers   | 12          |

A tensor with shape `(B, T, C)` is stored in a flat `&[f32]`.

The element `(b, t, c)` is located at:

```text
b * T * C + t * C + c
```

This is called **row-major flattening**.

---

# 5. `encoder_forward`: converting tokens into vectors

```rust
pub fn encoder_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
)
```

Let's look at the types.

### Output

```rust
out: &mut [f32]
```

A mutable slice. This function writes the embedding vectors here.

### Input tokens

```rust
inp: &[i32]
```

An immutable slice containing token IDs.

For example:

```text
"Hello world"
    ↓ tokenizer
[15496, 995]
```

### Token embedding weights

```rust
wte: &[f32]
```

`wte` means **Weight Token Embeddings**.

Its conceptual shape is:

```text
(V, C)
```

Each vocabulary token has a vector of length `C`.

### Positional embeddings

```rust
wpe: &[f32]
```

Its shape is:

```text
(maxT, C)
```

GPT-2 needs positional information because attention alone doesn't inherently know whether a token is first or tenth.

---

## The main loops

```rust
for b in 0..B {
    for t in 0..T {
```

For every batch and every token position.

```rust
let out_bt = &mut out[b * T * C + t * C..][..C];
```

This is a slightly unusual but useful slice expression.

Suppose:

```text
B = 2
T = 4
C = 3
```

The starting position of `(b, t)` is:

```text
b * T * C + t * C
```

Then:

```rust
[..C]
```

takes exactly `C` elements.

So `out_bt` is effectively:

```text
out[b][t][0..C]
```

but without actually storing a 3-dimensional Rust array.

Next:

```rust
let ix = inp[b * T + t] as usize;
```

Gets the token ID.

The conversion is necessary because slice indices in Rust use `usize`.

```rust
let wte_ix = &wte[ix * C..][..C];
```

Gets the embedding vector for token `ix`.

```rust
let wpe_t = &wpe[t * C..][..C];
```

Gets the positional embedding for position `t`.

Finally:

```rust
for i in 0..C {
    out_bt[i] = wte_ix[i] + wpe_t[i];
}
```

Mathematically:

$$ \text{embedding}(token, position) = \text{token embedding}(token) + \text{position embedding}(position)$$

---

# 6. `encoder_backward`: backpropagation through addition

The forward operation is:

```text
out = token_embedding + position_embedding
```

The derivative of addition is simple.

If:

$$
y = a + b
$$

then:

$$ \frac{\partial L}{\partial a} = \frac{\partial L}{\partial y} $$

and:

$$ \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} $$

Therefore:

```rust
dwte_ix[i] += d;
dwpe_t[i] += d;
```

Notice the `+=`, not `=`.

Multiple positions can contain the same token:

```text
"The cat and the dog"
```

Both occurrences of `"the"` contribute gradients to the same token embedding. Therefore the gradients must be **accumulated**.

---

# 7. Layer Normalization

The function:

```rust
pub fn layernorm_forward(...)
```

implements LayerNorm for each `(b, t)` vector independently.

For a vector:

$$
x = [x_1, x_2, ..., x_C]
$$

### Step 1: Calculate the mean

```rust
let mut m = 0.0f32;
for i in 0..C {
    m += x[i];
}
m /= C as f32;
```

This calculates:

$$
m = \frac{1}{C}\sum_i x_i
$$

The explicit `as f32` converts the integer `C` into a floating-point value.

### Step 2: Calculate variance

```rust
let mut v = 0.0f32;
for i in 0..C {
    let xshift = x[i] - m;
    v += xshift * xshift;
}
v /= C as f32;
```

This calculates:

$$
v = \frac{1}{C}\sum_i(x_i-m)^2
$$

### Step 3: Reciprocal standard deviation

```rust
let eps = 1e-5f32;
let s = 1.0f32 / (v + eps).sqrt();
```

Mathematically:

$$
s = \frac{1}{\sqrt{v+\epsilon}}
$$

`eps` prevents division by zero and improves numerical stability.

### Step 4: Normalize, scale, and shift

```rust
let n = s * (x[i] - m);
let o = n * weight[i] + bias[i];
```

This is:

$$
y_i =
\gamma_i
\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta_i
$$

where `weight` is $$\(\gamma\)$$ and `bias` is $$\(\beta\)$$.

---

# 8. Matrix multiplication and Rayon parallelism

The naive implementation begins with:

```rust
out.par_chunks_exact_mut(OC)
    .enumerate()
    .for_each(|(bt, out_bt)| {
```

This deserves careful attention.

Normally, you might write:

```rust
for bt in 0..B * T {
```

But Rayon allows:

```rust
.par_chunks_exact_mut(OC)
```

to divide the output buffer into chunks of size `OC`.

Conceptually:

```text
out:

┌──────────── OC ────────────┐
│ output for token position 0│
├────────────────────────────┤
│ output for token position 1│
├────────────────────────────┤
│ output for token position 2│
└────────────────────────────┘
```

Each chunk is a separate mutable slice. Since they cannot overlap, Rust can safely let Rayon process them concurrently.

This is a major advantage of Rust's ownership model:

> Parallel mutation is safe because each thread receives a distinct `&mut [f32]`.

Inside:

```rust
let inp_bt = &inp[bt * C..(bt + 1) * C];
```

gets one input vector.

Then:

```rust
for o in 0..OC {
    let mut val = bias.map_or(0.0f32, |bias| bias[o]);
```

`bias` has the type:

```rust
Option<&[f32]>
```

So it may be either:

```rust
Some(bias)
```

or:

```rust
None
```

`map_or` means:

> If there is a bias, use `bias[o]`; otherwise use `0.0`.

Then the actual matrix multiplication is:

```rust
for i in 0..C {
    val += inp_bt[i] * weight[o * C + i];
}
```

Mathematically:

$$
out_o = bias_o + \sum_i input_i \times weight_{o,i}
$$

---

# 9. Optimized matrix multiplication

```rust
const LOOP_UNROLL: usize = 8;
```

Instead of calculating one `(B,T)` position at a time, this implementation calculates **eight positions simultaneously**.

```rust
let mut result = [0.0f32; LOOP_UNROLL];
```

This creates a fixed-size Rust array:

```text
[result0, result1, ..., result7]
```

The compiler may keep these values in CPU registers.

The important optimization is:

```rust
let w = weight[i + o * C];

for ibt in 0..LOOP_UNROLL {
    let bt = obt + ibt;
    result[ibt] += inp[bt * C + i] * w;
}
```

The same weight value `w` is loaded once and reused eight times.

This improves cache and register usage without requiring CPU-specific SIMD intrinsics.

---

# 10. Attention: the heart of GPT

The attention input has shape:

```text
(B, T, 3C)
```

The `3C` consists of:

```text
Query | Key | Value
  C       C      C
```

```rust
let C3 = C * 3;
let hs = C / NH;
```

For GPT-2 Small:

```text
C  = 768
NH = 12
hs = 64
```

Each attention head therefore works with a vector of 64 numbers.

The scale:

```rust
let scale = 1.0f32 / (hs as f32).sqrt();
```

implements:

$$
\frac{1}{\sqrt{d_k}}
$$

from the Transformer attention formula.

---

## Attention has four passes

### Pass 1: Query × Key

```rust
for t2 in 0..=t {
```

The `=` is crucial. It means:

> Include `t` itself.

For causal GPT attention, token `t` can attend to:

```text
0, 1, 2, ..., t
```

but not future tokens.

The dot product is:

```rust
let mut val = 0.0f32;
for i in 0..hs {
    val += query_t[i] * key_t2[i];
}
val *= scale;
```

This calculates:

$$
Q_t \cdot K_{t2} / \sqrt{d_k}
$$

---

### Pass 2: Exponentiation

```rust
let expv = (preatt_bth[t2] - maxval).exp();
```

Subtracting `maxval` is a standard **numerically stable softmax** technique.

Instead of:

$$
e^{x_i}
$$

we calculate:

$$
e^{x_i-\max(x)}
$$

The probabilities remain mathematically identical after normalization, but overflow is less likely.

---

### Pass 3: Normalize

```rust
att_bth[t2] *= expsum_inv;
```

This produces attention probabilities:

$$
attention_i =
\frac{e^{score_i}}
{\sum_j e^{score_j}}
$$

Future positions are explicitly set to zero:

```rust
att_bth[t2] = 0.0f32;
```

This is the **causal mask** that makes GPT autoregressive.

---

### Pass 4: Weighted sum of Values

```rust
out_bth[i] += att_btht2 * value_t2[i];
```

This is:

$$
output_t = \sum_{t2 \leq t}
attention_{t,t2} \times V_{t2}
$$

This is how a GPT token can gather information from earlier tokens.

---

# 11. GELU activation

```rust
pub const GELU_SCALING_FACTOR: f32 = 0.797_884_560_802_865_4;
```

The comment says this is approximately:

$$
\sqrt{\frac{2}{\pi}}
$$

The forward implementation:

```rust
let x = inp[i];
let cube = 0.044715f32 * x * x * x;
out[i] = 0.5f32 * x
    * (1.0f32 + (GELU_SCALING_FACTOR * (x + cube)).tanh());
```

This is the approximate GELU function used by GPT-2:

$$
\mathrm{GELU}(x)
\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right]
$$

It is applied element-by-element.

---

# 12. Residual connections

The entire function is:

```rust
pub fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], N: usize) {
    for i in 0..N {
        out[i] = inp1[i] + inp2[i];
    }
}
```

Mathematically:

$$
output = input + transformation(input)
$$

This is the **residual connection** used throughout Transformers.

Backward is:

```rust
dinp1[i] += dout[i];
dinp2[i] += dout[i];
```

The gradient flows through both branches.

---

# 13. Softmax and cross-entropy

GPT produces raw scores called **logits**.

For example:

```text
"logits"

cat    2.1
dog    1.5
bird  -0.3
```

Softmax converts them into probabilities.

```rust
probs_bt[i] = (logits_bt[i] - maxval).exp();
```

Then:

```rust
probs_bt[i] /= sum;
```

Now:

$$
\sum_i probability_i = 1
$$

The padded vocabulary is interesting:

```rust
V  = actual vocabulary size
Vp = padded vocabulary size
```

GPT-2 might have:

```text
V  = 50257
Vp = 50304
```

The extra values exist for computational efficiency but don't represent real tokens.

---

## Cross-entropy loss

```rust
losses[b * T + t] = -probs_bt[ix].ln();
```

Mathematically:

$$
L = -\log(P(\text{correct token}))
$$

If the correct token has a high probability, the loss is small.

---

# 14. The beautiful Softmax + Cross-Entropy derivative

This function combines both operations:

```rust
crossentropy_softmax_backward(...)
```

The core line is:

```rust
dlogits_bt[i] += (p - indicator) * dloss;
```

where:

```rust
let indicator = if i == ix { 1.0f32 } else { 0.0f32 };
```

This represents the famous simplification:

$$\frac{\partial L}{\partial logits_i}=p_i - y_i$$

where `y` is the one-hot target vector.

This is one of the most important equations in neural network training.

---

# 15. `TensorView`: safe tensor pointers

```rust
#[derive(Clone, Copy, Debug)]
pub struct TensorView {
    pub start: usize,
    pub len: usize,
}
```

This struct doesn't own any memory.

It simply says:

> "My tensor begins at index `start` and contains `len` floating-point numbers."

For example:

```rust
TensorView {
    start: 1000,
    len: 768,
}
```

means:

```text
memory[1000..1768]
```

### Why `Copy`?

`TensorView` contains only two `usize` values. Copying it is cheap and doesn't involve copying tensor data.

This allows code such as:

```rust
let a = self.acts.unwrap();
let p = self.params;
```

without moving the actual large model memory.

Only the small metadata structs are copied.

---

# 16. `split_disjoint`: one of the key Rust techniques

This function solves a fundamental Rust borrowing problem.

Suppose one `Vec<f32>` contains multiple tensors. You want:

```rust
let tensor_a = &mut memory[0..100];
let tensor_b = &mut memory[100..200];
```

Rust can sometimes understand this directly, but when ranges are dynamically calculated, borrowing becomes harder.

`split_disjoint` safely creates multiple mutable slices after verifying they don't overlap.

Its signature uses **const generics**:

```rust
pub(crate) fn split_disjoint<'a, const N: usize>(
    buf: &'a mut [f32],
    ranges: [(usize, usize); N],
) -> [&'a mut [f32]; N]
```

`N` is known at compile time.

If you call it with three ranges:

```rust
split_disjoint(buffer, [range1, range2, range3])
```

then Rust infers:

```text
N = 3
```

and returns:

```rust
[&mut [f32]; 3]
```

Before borrowing, it checks:

```rust
assert!(start + len <= buf.len(), "tensor view out of bounds");
```

Then:

```rust
assert!(start >= prev_end, "tensor views must be pairwise disjoint");
```

This guarantees that no two returned mutable slices overlap.

That is the crucial safety property:

```text
One big allocation
       │
       ▼
┌─────────────────────────────┐
│           Vec<f32>          │
└─────────────────────────────┘
       │
       ▼ split_disjoint
 ┌─────┼─────┐
 ▼     ▼     ▼
&mut A &mut B &mut C

All regions are proven disjoint.
```

This lets the implementation retain C-like memory efficiency while remaining **100% safe Rust**.

---

# 17. Parameter tensors

```rust
pub struct ParameterTensors {
    pub wte: TensorView,
    pub wpe: TensorView,
    ...
}
```

This struct contains the locations of every trainable parameter.

For example:

```rust
pub qkvw: TensorView, // (L, 3*C, C)
```

means the Query/Key/Value projection weights for all `L` Transformer layers.

Importantly, `ParameterTensors` does **not contain the weights themselves**.

The actual data is here:

```rust
pub params_memory: Vec<f32>,
```

So:

```text
GPT2
 ├── params_memory ───────► [all floating-point parameters]
 │
 └── params
       ├── wte ──► start + length
       ├── wpe ──► start + length
       ├── qkvw ─► start + length
       └── ...
```

---

# 18. Activation tensors

The activations are intermediate results produced during the forward pass.

For example:

```rust
pub qkv: TensorView,
pub att: TensorView,
pub fch_gelu: TensorView,
pub logits: TensorView,
pub probs: TensorView,
pub losses: TensorView,
```

Why store all these?

Because **backpropagation needs many forward-pass values**.

For example, the GELU backward pass needs the original input:

```rust
gelu_backward(dl_fch, l_fch, dl_fch_gelu, ...)
```

Training therefore requires substantially more memory than inference.

---

# 19. The `GPT2` struct: the complete model state

```rust
pub struct GPT2 {
```

This represents the entire trainable model.

The most important groups are:

### Model configuration

```rust
pub config: GPT2Config,
```

### Model parameters

```rust
pub params_memory: Vec<f32>,
```

### Parameter gradients

```rust
pub grads_memory: Option<Vec<f32>>,
```

These are `Option`s because gradients aren't needed until training begins.

### AdamW optimizer state

```rust
pub m_memory: Option<Vec<f32>>,
pub v_memory: Option<Vec<f32>>,
```

AdamW maintains two values per parameter:

* `m`: momentum / first moment
* `v`: squared-gradient / second moment

### Activations

```rust
pub acts_memory: Option<Vec<f32>>,
```

Allocated during the first forward pass because the required size depends on `B` and `T`.

This is called **lazy allocation**.

---

# 20. Loading the checkpoint

```rust
pub fn build_from_checkpoint(checkpoint_path: &str) -> GPT2
```

The argument:

```rust
&str
```

is a borrowed string slice. The function doesn't need to own the path.

```rust
let mut model_file = fopen_check(checkpoint_path, "rb");
```

Opens the checkpoint.

```rust
let model_header = read_i32s(&mut model_file, 256);
```

Reads 256 `i32` values from the file header.

Then:

```rust
if model_header[0] != 20240326 {
```

checks the **magic number**. This verifies that the file is in the expected format.

```rust
if model_header[1] != 3 {
```

checks the file format version.

---

# 21. The forward pass: the complete GPT-2 pipeline

The most important function is:

```rust
pub fn gpt2_forward(&mut self, ...)
```

The `&mut self` means this function can modify the model's internal state, including:

* activation buffers
* cached inputs
* cached targets
* mean loss

The pipeline is:

```text
Input Tokens
     │
     ▼
Token + Position Embeddings
     │
     ▼
┌─────────────────────────────┐
│ Transformer Layer 0         │
│ LayerNorm                   │
│ QKV Projection              │
│ Causal Multi-Head Attention │
│ Residual Connection         │
│ LayerNorm                   │
│ MLP → GELU → MLP            │
│ Residual Connection         │
└──────────────┬──────────────┘
               │
              ...
               │
     ▼
Final LayerNorm
     │
     ▼
Vocabulary Projection
     │
     ▼
Softmax
     │
     ▼
Cross-Entropy Loss
```

The loop:

```rust
for l in 0..L {
```

runs every Transformer layer.

Inside, these lines are essentially the architecture of GPT-2 written directly as code:

```rust
layernorm_forward(...)
matmul_forward(...)      // create Q, K, V
attention_forward(...)
matmul_forward(...)      // attention output projection
residual_forward(...)
layernorm_forward(...)
matmul_forward(...)      // MLP expansion: C → 4C
gelu_forward(...)
matmul_forward(...)      // MLP projection: 4C → C
residual_forward(...)
```

That's the GPT-2 Transformer block.

---

# 22. Why is `split_disjoint` used in the forward pass?

This code:

```rust
let [l_ln1, l_ln1_mean, ..., residual] =
    split_disjoint(acts_memory, [...]);
```

destructures an array of mutable slices.

It is similar in spirit to:

```rust
let [a, b, c] = array;
```

But here every variable becomes a separate `&mut [f32]`.

Without a mechanism like this, Rust would reject multiple mutable references into the same `acts_memory` buffer because overlapping mutable references could cause undefined behavior.

`split_disjoint` proves that they don't overlap.

This is an excellent example of adapting a low-level C memory layout to Rust's ownership rules.

---

# 23. Backpropagation goes in reverse

The backward function contains:

```rust
for l in (0..L).rev() {
```

This means:

```text
L - 1
L - 2
...
1
0
```

Backpropagation must run in the reverse order of the forward computation because of the **chain rule**.

For example, the forward MLP is:

```text
LayerNorm
   ↓
Matrix Multiplication
   ↓
GELU
   ↓
Matrix Multiplication
```

The backward pass is:

```text
Matrix Multiplication Backward
   ↑
GELU Backward
   ↑
Matrix Multiplication Backward
   ↑
LayerNorm Backward
```

The code mirrors this exactly.

---

# 24. Starting backpropagation

```rust
let dloss_mean = 1.0f32 / (B * T) as f32;
```

The final loss is the mean:

$$
L =
\frac{1}{BT}
\sum_{b,t} loss_{b,t}
$$

Therefore:

$$\frac{\partial L}{\partial loss_{b,t}}=\frac{1}{BT}$$

The code initializes every loss gradient with this value:

```rust
dlosses[i] = dloss_mean;
```

This is the starting point for the entire chain rule.

---

# 25. AdamW parameter update

The final major function is:

```rust
pub fn gpt2_update(...)
```

For every parameter:

```rust
for i in 0..*num_parameters {
```

the optimizer updates two running averages.

### First moment

```rust
let m = beta1 * m_memory[i]
    + (1.0f32 - beta1) * grad;
```

This is a smoothed average of gradients.

### Second moment

```rust
let v = beta2 * v_memory[i]
    + (1.0f32 - beta2) * grad * grad;
```

This tracks the magnitude of gradients.

### Bias correction

```rust
let m_hat = m / (1.0f32 - beta1.powf(t as f32));
let v_hat = v / (1.0f32 - beta2.powf(t as f32));
```

At the beginning of training, both moment estimates are biased toward zero. These formulas correct that bias.

Finally:

```rust
params_memory[i] -= learning_rate
    * (m_hat / (v_hat.sqrt() + eps)
       + weight_decay * param);
```

This updates the model parameter.

The important detail is that all model parameters are in one `Vec<f32>`, so the optimizer can simply iterate through them linearly.

---

# 26. Rust ownership and memory management in this entire program

One of the nicest things about this implementation is how naturally Rust handles cleanup.

The original C implementation likely has something conceptually like:

```c
gpt2_free(model);
```

But the Rust code ends with:

```rust
// gpt2_free() has no Rust equivalent: all model memory is freed when the GPT2 is dropped
```

This is **RAII**, which connects directly to the C++ and Rust ownership discussions we've had.

When:

```rust
GPT2
```

goes out of scope, Rust automatically drops its fields:

```text
GPT2 dropped
   │
   ├── Vec<f32> parameters freed
   ├── Vec<f32> gradients freed
   ├── Vec<f32> activations freed
   ├── AdamW buffers freed
   └── other fields dropped
```

No manual `free()` is necessary.

---

# The most important overall idea

I would summarize this `lib.rs` as having **three layers of design**:

### ① Mathematical layer

The neural-network operations:

```text
LayerNorm
Matrix Multiplication
Attention
GELU
Softmax
Cross-Entropy
Backpropagation
AdamW
```

### ② Memory-layout layer

All tensors are packed into a few large:

```rust
Vec<f32>
```

allocations for efficiency.

### ③ Rust safety layer

Instead of raw pointers, the implementation uses:

```rust
TensorView { start, len }
split_disjoint(...)
&[f32]
&mut [f32]
Option<Vec<f32>>
```

to preserve memory efficiency while preventing unsafe aliasing.

**That combination—low-level tensor memory management with safe Rust slices—is probably the most important architectural idea to understand before studying the rest of the implementation in even greater detail.**

When you send the next file or section, I can continue with the same approach and go **literally line by line**, including Rust syntax, tensor shapes, memory layout, and the GPT-2 mathematics behind each operation.

