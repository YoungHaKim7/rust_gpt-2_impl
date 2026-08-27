/*
Verification helper: generates a small synthetic "GPT-2 checkpoint + tokenizer +
token dataset" in exactly the binary formats that train_gpt2 (both the original
llm.c C version and this Rust port) reads. Deterministic (mt19937 seed 42), so
two runs produce identical bytes — which lets us run the C and Rust trainers
side by side on the same inputs and compare their loss trajectories.

Usage: gen_synth <output_dir>
Writes:
  <output_dir>/gpt2_124M.bin                                 (the fixed name train_gpt2 reads)
  <output_dir>/gpt2_tokenizer.bin
  <output_dir>/dev/data/tinyshakespeare/tiny_shakespeare_train.bin
  <output_dir>/dev/data/tinyshakespeare/tiny_shakespeare_val.bin
*/

#![allow(non_snake_case)]

use std::path::Path;

use rust_gpt_2_impl::{
    llmc::rand::{Mt19937State, manual_seed, normal_, randint32},
    llmc::utils::{fopen_check, write_f32s, write_i32_header, write_u16s, write_u32s},
    {GPT2Config, NUM_PARAMETER_TENSORS, fill_in_parameter_sizes},
};

// a tiny GPT-2 config: small enough to train 40 steps quickly on any machine,
// with max_seq_len == 64 == the T that train_gpt2 uses
const MAX_SEQ_LEN: usize = 64;
const VOCAB_SIZE: usize = 512;
const NUM_LAYERS: usize = 4;
const NUM_HEADS: usize = 4;
const CHANNELS: usize = 128;

const TRAIN_NTOK: usize = 4 * 64 * 20; // 20 batches worth of tokens
const VAL_NTOK: usize = 4 * 64 * 8; // 8 batches worth of tokens

fn write_tokens_file(path: &Path, ntok: usize, vocab: usize, rng: &mut Mt19937State) {
    let mut f = fopen_check(&path.to_string_lossy(), "wb");
    write_i32_header(&mut f, &[20240520, 1, ntok as i32]);
    let tokens: Vec<u16> = (0..ntok)
        .map(|_| (randint32(rng) % vocab as u32) as u16)
        .collect();
    write_u16s(&mut f, &tokens);
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: gen_synth <output_dir>");
    let out_dir = Path::new(&out_dir);
    std::fs::create_dir_all(out_dir.join("dev/data/tinyshakespeare"))
        .expect("could not create output dirs");

    // deterministic generator
    let mut rng = Mt19937State::default();
    manual_seed(&mut rng, 42);

    // ---- the model checkpoint -------------------------------------------------
    let config = GPT2Config {
        max_seq_len: MAX_SEQ_LEN,
        vocab_size: VOCAB_SIZE,
        padded_vocab_size: VOCAB_SIZE,
        num_layers: NUM_LAYERS,
        num_heads: NUM_HEADS,
        channels: CHANNELS,
    };
    let mut param_sizes = [0usize; NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(&mut param_sizes, &config);
    let num_parameters: usize = param_sizes.iter().sum();

    let mut f = fopen_check(&out_dir.join("gpt2_124M.bin").to_string_lossy(), "wb");
    write_i32_header(
        &mut f,
        &[
            20240326,
            3,
            config.max_seq_len as i32,
            config.vocab_size as i32,
            config.num_layers as i32,
            config.num_heads as i32,
            config.channels as i32,
            config.padded_vocab_size as i32,
        ],
    );
    // weights ~ N(0, 0.02), like a fresh transformer init
    let mut params = vec![0.0f32; num_parameters];
    normal_(&mut params, 0.0, 0.02, &mut rng);
    write_f32s(&mut f, &params);
    println!(
        "wrote {} ({} params)",
        out_dir.join("gpt2_124M.bin").display(),
        num_parameters
    );

    // ---- the tokenizer --------------------------------------------------------
    let mut f = fopen_check(&out_dir.join("gpt2_tokenizer.bin").to_string_lossy(), "wb");
    let mut header = vec![0u32; 256];
    header[0] = 20240328; // magic
    header[1] = 2; // version 2 (includes the EOT token id)
    header[2] = VOCAB_SIZE as u32;
    header[3] = (VOCAB_SIZE - 1) as u32; // eot token id
    write_u32s(&mut f, &header);
    let mut token = String::new();
    for i in 0..VOCAB_SIZE {
        // short printable tokens (2-3 chars), so generation output is legible
        token.clear();
        token.push((b'a' + (i % 26) as u8) as char);
        token.push((b'a' + ((i / 26) % 26) as u8) as char);
        if i >= 26 * 26 {
            token.push((b'a' + ((i / (26 * 26)) % 26) as u8) as char);
        }
        let bytes = token.as_bytes();
        let len_byte = [bytes.len() as u8];
        rust_gpt_2_impl::llmc::utils::fwrite_check(&len_byte, &mut f);
        rust_gpt_2_impl::llmc::utils::fwrite_check(bytes, &mut f);
    }
    println!("wrote {}", out_dir.join("gpt2_tokenizer.bin").display());

    // ---- the token datasets ---------------------------------------------------
    write_tokens_file(
        &out_dir.join("dev/data/tinyshakespeare/tiny_shakespeare_train.bin"),
        TRAIN_NTOK,
        VOCAB_SIZE,
        &mut rng,
    );
    write_tokens_file(
        &out_dir.join("dev/data/tinyshakespeare/tiny_shakespeare_val.bin"),
        VAL_NTOK,
        VOCAB_SIZE,
        &mut rng,
    );
    println!(
        "wrote {} train + {} val tokens under {}",
        TRAIN_NTOK,
        VAL_NTOK,
        out_dir.join("dev/data/tinyshakespeare").display()
    );
}
