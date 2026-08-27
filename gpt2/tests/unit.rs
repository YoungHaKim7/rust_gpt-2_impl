/*
Unit tests for the Rust port of llm.c. These are self-contained (no PyTorch,
no downloads):
- matmul_forward (tiled/unrolled) == matmul_forward_naive, including the
  B*T % 8 != 0 fallback path
- finite-difference gradient check of the whole model (forward + backward),
  which exercises every backward kernel end to end
- tokenizer and dataloader round-trips on synthetic files
The strongest end-to-end check is `make verify` (dev/verify_vs_c.sh), which
runs this port against the compiled original C code on identical data.
*/

#![allow(non_snake_case)]

use std::path::PathBuf;

use gpt2::llmc::dataloader::DataLoader;
use gpt2::llmc::rand::{manual_seed, normal_, randint32, Mt19937State};
use gpt2::llmc::utils::{fopen_check, write_f32s, write_i32_header, write_u16s, write_u32s};
use gpt2::llmc::tokenizer::Tokenizer;
use gpt2::{matmul_forward, matmul_forward_naive, fill_in_parameter_sizes, GPT2, GPT2Config, NUM_PARAMETER_TENSORS};

/// tiny xorshift for test data (independent of the llmc rng)
struct TestRng(u64);
impl TestRng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0 // in [-1, 1)
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gpt2_test_{}_{tag}_{}", std::process::id(), std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_tokenizer_file(path: &std::path::Path, vocab: u32, eot: u32, tokens: &[&str]) {
    let mut f = fopen_check(&path.to_string_lossy(), "wb");
    let mut header = vec![0u32; 256];
    header[0] = 20240328;
    header[1] = 2;
    header[2] = vocab;
    header[3] = eot;
    write_u32s(&mut f, &header);
    for tok in tokens {
        let len_byte = [tok.len() as u8];
        gpt2::llmc::utils::fwrite_check(&len_byte, &mut f);
        gpt2::llmc::utils::fwrite_check(tok.as_bytes(), &mut f);
    }
}

fn write_tokens_file(path: &std::path::Path, tokens: &[u16]) {
    let mut f = fopen_check(&path.to_string_lossy(), "wb");
    write_i32_header(&mut f, &[20240520, 1, tokens.len() as i32]);
    write_u16s(&mut f, tokens);
}

fn write_tiny_checkpoint(path: &std::path::Path) -> GPT2Config {
    let config = GPT2Config {
        max_seq_len: 16,
        vocab_size: 32,
        padded_vocab_size: 32,
        num_layers: 2,
        num_heads: 2,
        channels: 16,
    };
    let mut param_sizes = [0usize; NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(&mut param_sizes, &config);
    let num_parameters: usize = param_sizes.iter().sum();

    let mut rng = Mt19937State::default();
    manual_seed(&mut rng, 7);
    let mut params = vec![0.0f32; num_parameters];
    // std 0.1 (not a realistic init) so that gradients are large enough to be
    // checked against f32 finite differences
    normal_(&mut params, 0.0, 0.1, &mut rng);

    let mut f = fopen_check(&path.to_string_lossy(), "wb");
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
    write_f32s(&mut f, &params);
    config
}

#[test]
fn matmul_forward_tiled_matches_naive() {
    // (B, T, C, OC) — the third case has B*T % 8 != 0, exercising the fallback
    for &(B, T, C, OC) in &[(2usize, 4, 16, 32), (4, 16, 32, 64), (3, 5, 8, 16), (1, 8, 8, 8)] {
        let mut rng = TestRng(0x1234_5678_9abc_def0 ^ (B as u64 * 1000 + T as u64));
        let inp: Vec<f32> = (0..B * T * C).map(|_| rng.next_f32()).collect();
        let weight: Vec<f32> = (0..OC * C).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..OC).map(|_| rng.next_f32()).collect();

        let mut out_naive = vec![0.0f32; B * T * OC];
        matmul_forward_naive(&mut out_naive, &inp, &weight, Some(&bias), B, T, C, OC);
        let mut out_tiled = vec![0.0f32; B * T * OC];
        matmul_forward(&mut out_tiled, &inp, &weight, Some(&bias), B, T, C, OC);

        let mut maxdiff = 0.0f32;
        for i in 0..out_naive.len() {
            let diff = (out_naive[i] - out_tiled[i]).abs();
            assert!(diff < 1e-5, "B={B} T={T} C={C} OC={OC}: index {i}: {diff}");
            maxdiff = maxdiff.max(diff);
        }
        println!("matmul B={B} T={T} C={C} OC={OC}: maxdiff {maxdiff:e}");
    }
}

#[test]
fn finite_difference_gradient_check() {
    let dir = temp_dir("fdgrad");
    let ckpt = dir.join("tiny.bin");
    let _config = write_tiny_checkpoint(&ckpt);

    let mut model = GPT2::build_from_checkpoint(&ckpt.to_string_lossy());

    let B = 2usize;
    let T = 8usize; // <= max_seq_len (16)
    let V = model.config.vocab_size;
    let mut rng = Mt19937State::default();
    manual_seed(&mut rng, 1234);
    let inputs: Vec<i32> = (0..B * T).map(|_| (randint32(&mut rng) % V as u32) as i32).collect();
    let targets: Vec<i32> = (0..B * T).map(|_| (randint32(&mut rng) % V as u32) as i32).collect();

    // analytic gradients from the backward pass
    model.gpt2_forward(&inputs, Some(&targets), B, T);
    let expected_loss = model.mean_loss;
    model.gpt2_zero_grad();
    model.gpt2_backward();
    let analytic = model.grads_memory.clone().unwrap();

    // sanity: loss should be near ln(V) for a random-init model
    assert!((expected_loss - (V as f32).ln()).abs() < 0.5, "unexpected initial loss {expected_loss}");

    // numerical gradients by central finite differences on a spread of parameters
    let n = model.num_parameters;
    let h = 2e-2f32;
    let num_samples = 96usize;
    let mut checked = 0usize;
    let mut max_rel_err = 0.0f32;
    for s in 0..num_samples {
        let i = s * n / num_samples;
        let a = analytic[i];
        let old = model.params_memory[i];

        model.params_memory[i] = old + h;
        model.gpt2_forward(&inputs, Some(&targets), B, T);
        let l_plus = model.mean_loss;
        model.params_memory[i] = old - h;
        model.gpt2_forward(&inputs, Some(&targets), B, T);
        let l_minus = model.mean_loss;
        model.params_memory[i] = old;

        let fd = (l_plus - l_minus) / (2.0 * h);
        if a.abs() < 1e-4 {
            continue; // gradient too small to compare in f32
        }
        // for small gradients the f32 noise of the loss difference dominates the
        // relative error, so accept a small absolute error there; larger gradients
        // (where a real kernel bug would show up as an O(100%) error) get the tight
        // relative check
        let ok = (fd - a).abs() < 6e-5 || (fd - a).abs() / a.abs() < 5e-2;
        assert!(
            ok,
            "param {i}: analytic {a:e} vs finite-diff {fd:e} (rel err {:.3})",
            (fd - a).abs() / a.abs()
        );
        max_rel_err = max_rel_err.max((fd - a).abs() / a.abs());
        checked += 1;
    }
    println!("finite-difference check: {checked}/{num_samples} params compared, max rel err {max_rel_err:e}");
    assert!(checked >= 50, "too few comparable gradients: {checked}");
}

#[test]
fn tokenizer_roundtrip() {
    let dir = temp_dir("tok");
    let path = dir.join("tok.bin");
    let tokens: Vec<&str> = vec!["hello", " ", "world", "a", "bb", "ccc", "eot", "x"];
    write_tokenizer_file(&path, tokens.len() as u32, 7, &tokens);

    let tokenizer = Tokenizer::init(&path.to_string_lossy());
    assert!(tokenizer.init_ok);
    assert_eq!(tokenizer.vocab_size, 8);
    assert_eq!(tokenizer.eot_token, 7);
    assert_eq!(tokenizer.decode(0), Some(&b"hello"[..]));
    assert_eq!(tokenizer.decode(2), Some(&b"world"[..]));
    assert_eq!(tokenizer.decode(8), None); // out of range
}

#[test]
fn dataloader_serves_shifted_batches_and_wraps() {
    let dir = temp_dir("dl");
    let path = dir.join("tokens.bin");
    // 100 tokens; with B=2,T=4 (8 tokens moved per batch) that allows 12 samples
    let tokens: Vec<u16> = (0..100u16).collect();
    write_tokens_file(&path, &tokens);

    let mut loader = DataLoader::init(&path.to_string_lossy(), 2, 4, 0, 1, 0);
    assert_eq!(loader.num_tokens, 100);

    // batch 0: inputs = tokens[0..8], targets = tokens[1..9]
    loader.next_batch();
    assert_eq!(&loader.inputs[..8], &(0..8i32).collect::<Vec<_>>()[..]);
    assert_eq!(&loader.targets[..8], &(1..9i32).collect::<Vec<_>>()[..]);
    let first = loader.inputs.clone();

    // batch 1 starts at sample idx 1 -> byte offset 16 -> token 8
    loader.next_batch();
    assert_eq!(loader.inputs[0], 8);
    assert_eq!(loader.targets[7], 16);

    // consume the rest of the epoch; the loader should wrap back to sample 0
    for _ in 0..11 {
        loader.next_batch();
    }
    // 13 next_batch calls total on 12 samples: wrapped around to batch 0 again
    assert_eq!(loader.inputs, first);
}
