# `llmc/tokenizer.rs`

- This `tokenizer.rs` completes another important part of your `llmc` Rust port. Unlike the `DataLoader`, which prepares data for **training**, the `Tokenizer` is mainly used to turn the model's output numbers back into **human-readable text**.

# Overall purpose

A GPT model doesn't directly produce strings such as `"Hello"`.

It produces **token IDs**:

```text
Model output:
[15496, 995, 0]
```

The tokenizer maps those IDs to their corresponding byte sequences:

```text
15496 → b"Hello"
995   → b" world"
0     → b"!"
```

Then the generated text becomes:

```text
Hello world!
```

Your module therefore performs:

```text
┌─────────────┐
│ GPT Model   │
│             │
│ token ID    │
│  15496      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Tokenizer   │
│ decode()    │
└──────┬──────┘
       │
       ▼
    b"Hello"
       │
       ▼
  safe_printf()
       │
       ▼
     Hello
```

---

# `Tokenizer` structure

```rust
pub struct Tokenizer {
    pub vocab_size: u32,
    pub token_table: Vec<Vec<u8>>,
    pub init_ok: bool,
    pub eot_token: i32,
}
```

Let's look at each field.

### `vocab_size`

```rust
pub vocab_size: u32,
```

This is the total number of tokens the tokenizer knows about.

For the original GPT-2 tokenizer, this is typically:

```text
50257 tokens
```

So valid token IDs are:

```text
0 ... 50256
```

---

### `token_table`

```rust
pub token_table: Vec<Vec<u8>>,
```

This is essentially the vocabulary:

```text
token ID ────────────────► bytes

0      ──────────────────► b"!"
1      ──────────────────► b"\""
...
15496  ──────────────────► b"Hello"
...
50256  ──────────────────► b"<|endoftext|>"
```

An important detail is that you store **`Vec<u8>` rather than `String`**.

That is the correct choice for this tokenizer format because tokens may contain arbitrary bytes that aren't valid UTF-8.

For example:

```rust
let token: Vec<u8> = vec![0xFF];
```

This cannot be represented as a normal Rust `String`, because `String` requires valid UTF-8.

So:

```rust
Vec<u8>
```

preserves the original tokenizer data exactly.

---

# `Tokenizer::init()`: Loading the vocabulary

The initialization process is:

```text
tokenizer.bin
     │
     ▼
┌─────────────────────┐
│ Read 256-word header│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Validate magic      │
│ Validate version    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Read vocab size     │
│ Read EOT token      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Read every token    │
│ as raw bytes        │
└─────────────────────┘
```

The magic number check:

```rust
assert!(header[0] == 20240328, "bad magic in tokenizer file");
```

is a file-format validation mechanism. It helps prevent accidentally reading the wrong file as a tokenizer.

---

## Tokenizer file versions

Your code supports two versions:

```rust
if version == 1 {
    tokenizer.eot_token = 50256;
} else if version == 2 {
    tokenizer.eot_token = header[3] as i32;
}
```

### Version 1

The EOT token ID wasn't stored in the file, so the code assumes GPT-2's traditional value:

```text
50256 = <|endoftext|>
```

### Version 2

The EOT token ID is explicitly stored in the header:

```text
header[3] → EOT token ID
```

This makes the format more flexible because future tokenizers don't necessarily have to use GPT-2's vocabulary layout.

---

# How each token is stored

The vocabulary loading loop is:

```rust
for _ in 0..tokenizer.vocab_size {
    let mut len_buf = [0u8; 1];
    fread_check(&mut len_buf, &mut file);

    let length = len_buf[0] as usize;
    assert!(length > 0);

    let token_bytes = read_bytes(&mut file, length);
    token_table.push(token_bytes);
}
```

Conceptually, the file looks something like:

```text
[length][token bytes]
[length][token bytes]
[length][token bytes]
...
```

For example:

```text
05 H e l l o
06 ␠ w o r l d
01 !
```

which becomes:

```rust
vec![
    b"Hello".to_vec(),
    b" world".to_vec(),
    b"!".to_vec(),
]
```

Notice that a token doesn't necessarily represent a whole word. GPT-style tokenizers can have tokens for:

* whole words
* parts of words
* spaces plus words
* punctuation
* individual bytes

---

# `decode()`: Token ID → bytes

The core decoding function is very simple:

```rust
pub fn decode(&self, token_id: u32) -> Option<&[u8]> {
    if !self.init_ok {
        return None;
    }

    if token_id < self.vocab_size {
        Some(&self.token_table[token_id as usize])
    } else {
        println!("invalid token id {token_id}!");
        None
    }
}
```

A nice Rust design decision here is the return type:

```rust
Option<&[u8]>
```

Let's break it down:

```text
Option
 ├── Some(...) → valid token
 └── None      → tokenizer unavailable or invalid token
```

And:

```rust
&[u8]
```

is a borrowed byte slice. No new allocation occurs when decoding.

So this:

```rust
let piece = tokenizer.decode(token_id);
```

does **not copy the token data**. It simply returns a reference into `token_table`.

This is efficient and idiomatic Rust.

---

# Why `safe_printf()` is necessary

This function is especially important because the tokenizer contains raw bytes:

```rust
pub fn safe_printf(piece: &[u8]) {
```

Imagine a token containing a terminal control character such as:

```text
ESC
Backspace
Bell
```

Blindly printing such bytes could cause undesirable terminal behavior.

Your code handles single-byte tokens specially:

```rust
if piece.len() == 1 {
    let byte_val = piece[0];
    let printable =
        byte_val.is_ascii_graphic() ||
        byte_val.is_ascii_whitespace();

    if !printable {
        return;
    }
}
```

For example:

```text
b"A"    → printed
b" "    → printed
b"\n"   → printed
b"\x08" → ignored
b"\x1B" → ignored
```

This follows the original `llm.c` philosophy: **preserve the tokenizer's raw representation but avoid printing dangerous or strange individual control bytes.**

Then the actual output is:

```rust
let mut stdout = std::io::stdout();
let _ = stdout.write_all(piece);
```

This is closer to C's raw byte output than using:

```rust
println!("{}", String::from_utf8_lossy(piece));
```

because `from_utf8_lossy()` could transform invalid UTF-8 bytes into replacement characters (`�`), changing the original data.

---

# How this connects to the rest of `llmc`

You now have several pieces of the overall system:

```text
                        Training
                           │
                           ▼
                 ┌─────────────────┐
                 │ DataLoader      │
                 │ dataloader.rs   │
                 └────────┬────────┘
                          │
                    input/target IDs
                          │
                          ▼
                    ┌───────────┐
                    │ GPT Model │
                    └─────┬─────┘
                          │
                    predicted token IDs
                          │
                          ▼
                 ┌─────────────────┐
                 │ Tokenizer       │
                 │ tokenizer.rs    │
                 └────────┬────────┘
                          │
                       &[u8]
                          │
                          ▼
                    safe_printf()
                          │
                          ▼
                       Terminal
```

Meanwhile, `rand.rs` supports reproducibility and random operations:

```text
rand.rs
   │
   ├── DataLoader shuffling
   │
   └── Model initialization / sampling
```

---

# One important distinction: decoding vs encoding

Your comment says:

> Only supports decoding, i.e.: tokens (integers) -> strings

This means the tokenizer supports:

```text
Token ID → Text
```

but not:

```text
Text → Token ID
```

Therefore, unconditional generation works:

```text
Start with a special token
        ↓
Model generates token 15496
        ↓
Decode → "Hello"
        ↓
Model generates next token
        ↓
Decode → " world"
```

But user prompting would require encoding:

```text
"Hello, how are you?"
        ↓
Tokenizer encoding
        ↓
[15496, ..., ...]
        ↓
Feed IDs into model
```

So the comment correctly explains why encoding would be needed later for an interactive or prompt-based inference program.

---

## Summary

Your `tokenizer.rs` is a compact and faithful Rust implementation of the **output side of GPT-2 tokenization**:

| Component           | Job                                                   |
| ------------------- | ----------------------------------------------------- |
| `Tokenizer::init()` | Loads the binary vocabulary                           |
| `token_table`       | Maps token IDs to raw bytes                           |
| `decode()`          | Returns a token's byte representation without copying |
| `safe_printf()`     | Safely writes decoded bytes to the terminal           |
| `eot_token`         | Identifies the end-of-text token                      |

The use of `Vec<Vec<u8>>` and `Option<&[u8]>` is particularly appropriate in Rust: the vocabulary is owned by the `Tokenizer`, while decoding provides an efficient **borrowed view** of the existing token bytes. This is a good example of Rust ownership improving clarity without changing the original C algorithm.
