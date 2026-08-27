/*
This binary trains the GPT-2 model. Port of the sampler + main() training loop
at the bottom of llm.c/train_gpt2.c (the part guarded by `#ifndef TESTING`).
*/

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)] // the loops mirror the C reference index-by-index

use std::{io::Write, path::Path, time::Instant};

use rust_gpt_2_impl::{
    GPT2,
    llmc::dataloader::DataLoader,
    llmc::tokenizer::{Tokenizer, safe_printf},
};

// ----------------------------------------------------------------------------
// sampler

fn random_u32(state: &mut u64) -> u32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    // random float32 in [0,1)
    (random_u32(state) >> 8) as f32 / 16777216.0f32
}

fn sample_mult(probabilities: &[f32], n: usize, coin: f32) -> usize {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    let mut cdf = 0.0f32;
    for i in 0..n {
        cdf += probabilities[i];
        if coin < cdf {
            return i;
        }
    }
    n - 1 // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
fn main() {
    // build the GPT-2 model from a checkpoint
    let mut model = GPT2::build_from_checkpoint("gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    let tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    let tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    let tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    let tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    let train_tokens = if Path::new(tiny_shakespeare_train).exists() {
        tiny_shakespeare_train
    } else {
        tiny_stories_train
    };
    let val_tokens = if Path::new(tiny_shakespeare_val).exists() {
        tiny_shakespeare_val
    } else {
        tiny_stories_val
    };
    let B: usize = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    let T: usize = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    let mut train_loader = DataLoader::init(train_tokens, B, T, 0, 1, 1);
    let mut val_loader = DataLoader::init(val_tokens, B, T, 0, 1, 0);
    println!(
        "train dataset num_batches: {}",
        train_loader.num_tokens / (B * T)
    );
    println!(
        "val dataset num_batches: {}",
        val_loader.num_tokens / (B * T)
    );
    let val_num_batches = 5;

    // build the Tokenizer
    let tokenizer = Tokenizer::init("gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    let mut rng_state: u64 = 1337;
    let mut gen_tokens = vec![0i32; B * T];
    let genT: usize = 64; // number of steps of inference we will do

    // train
    for step in 0..=40usize {
        // once in a while estimate the validation loss
        if step % 10 == 0 {
            let mut val_loss = 0.0f32;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch();
                model.gpt2_forward(&val_loader.inputs, Some(&val_loader.targets), B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("val loss {val_loss:.6}");
        }

        // once in a while do model inference to print generated text
        if step > 0 && step % 20 == 0 {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for tok in gen_tokens.iter_mut() {
                *tok = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            println!("generating:\n---");
            for t in 1..genT {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                model.gpt2_forward(&gen_tokens, None, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                let Vp = model.config.padded_vocab_size;
                let probs = model.acts.unwrap().probs.slice(
                    model.acts_memory.as_ref().unwrap(),
                    (t - 1) * Vp,
                    Vp,
                );
                let coin = random_f32(&mut rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                let next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token as i32;
                // print the generated token, either using the Tokenizer or a fallback
                if tokenizer.init_ok {
                    if let Some(token_str) = tokenizer.decode(next_token as u32) {
                        safe_printf(token_str);
                    }
                } else {
                    // fall back to printing the token id
                    print!("{next_token} ");
                }
                let _ = std::io::stdout().flush();
            }
            println!("\n---");
        }

        // do a training step
        let start = Instant::now();
        train_loader.next_batch();
        model.gpt2_forward(&train_loader.inputs, Some(&train_loader.targets), B, T);
        model.gpt2_zero_grad();
        model.gpt2_backward();
        model.gpt2_update(
            1e-4f32,
            0.9f32,
            0.999f32,
            1e-8f32,
            0.0f32,
            (step + 1) as i32,
        );
        let time_elapsed_s = start.elapsed().as_secs_f64();
        println!(
            "step {}: train loss {:.6} (took {:.6} ms)",
            step,
            model.mean_loss,
            time_elapsed_s * 1000.0
        );
    }

    // free (everything is dropped automatically)
}
