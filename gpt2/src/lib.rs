/*
This library trains the GPT-2 model. It is a faithful Rust port of the clean,
minimal, pure-CPU reference in llm.c/train_gpt2.c:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use rayon parallel iterators wherever the C code has OpenMP pragmas,
  as this is a large speedup at very low cost of code complexity.
Where the C code carves all tensors out of one big allocation with raw pointers,
this port keeps the exact same layout but tracks tensors as (start, len) views
into a single Vec, materialized as slices with `split_disjoint` at use sites.
There will be other versions of this code that specialize it and make it fast.
*/

#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod llmc;

use rayon::prelude::*;

use crate::llmc::utils::{fopen_check, read_f32s, read_i32s};

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

pub fn encoder_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for b in 0..B {
        for t in 0..T {
            // seek to the output position in out[b,t,:]
            let out_bt = &mut out[b * T * C + t * C..][..C];
            // get the index of the token at inp[b, t]
            let ix = inp[b * T + t] as usize;
            // seek to the position in wte corresponding to the token
            let wte_ix = &wte[ix * C..][..C];
            // seek to the position in wpe corresponding to the position
            let wpe_t = &wpe[t * C..][..C];
            // add the two vectors and store the result in out[b,t,:]
            for i in 0..C {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

pub fn encoder_backward(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[b * T * C + t * C..][..C];
            let ix = inp[b * T + t] as usize;
            let dwte_ix = &mut dwte[ix * C..][..C];
            let dwpe_t = &mut dwpe[t * C..][..C];
            for i in 0..C {
                let d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

pub fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    let eps = 1e-5f32;
    for b in 0..B {
        for t in 0..T {
            // seek to the input position inp[b,t,:]
            let x = &inp[b * T * C + t * C..][..C];
            // calculate the mean
            let mut m = 0.0f32;
            for i in 0..C {
                m += x[i];
            }
            m /= C as f32;
            // calculate the variance (without any bias correction)
            let mut v = 0.0f32;
            for i in 0..C {
                let xshift = x[i] - m;
                v += xshift * xshift;
            }
            v /= C as f32;
            // calculate the rstd (reciprocal standard deviation)
            let s = 1.0f32 / (v + eps).sqrt();
            // seek to the output position in out[b,t,:]
            let out_bt = &mut out[b * T * C + t * C..][..C];
            for i in 0..C {
                let n = s * (x[i] - m); // normalize
                let o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

pub fn layernorm_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[b * T * C + t * C..][..C];
            let inp_bt = &inp[b * T * C + t * C..][..C];
            let dinp_bt = &mut dinp[b * T * C + t * C..][..C];
            let mean_bt = mean[b * T + t];
            let rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            let mut dnorm_mean = 0.0f32;
            let mut dnorm_norm_mean = 0.0f32;
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // now iterate again and accumulate all the gradients
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                let mut dval = 0.0f32;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// B and T are kept in the signature to mirror the C reference (the parallel
// iteration is over the collapsed B*T rows)
#[allow(unused_variables)]
pub fn matmul_forward_naive(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    // (the C code has `#pragma omp parallel for collapse(2)` over b,t)
    out.par_chunks_exact_mut(OC)
        .enumerate()
        .for_each(|(bt, out_bt)| {
            let inp_bt = &inp[bt * C..(bt + 1) * C];
            for o in 0..OC {
                let mut val = bias.map_or(0.0f32, |bias| bias[o]);
                for i in 0..C {
                    val += inp_bt[i] * weight[o * C + i];
                }
                out_bt[o] = val;
            }
        });
}

const LOOP_UNROLL: usize = 8;

// B and T are kept in the signature to mirror the C reference (the parallel
// iteration is over the collapsed B*T rows); the modulo condition is verbatim C
#[allow(unused_variables)]
#[allow(clippy::manual_is_multiple_of)]
pub fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    if B * T % LOOP_UNROLL != 0 {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // (the C code has `#pragma omp parallel for` over obt)
    out.par_chunks_exact_mut(OC * LOOP_UNROLL)
        .enumerate()
        .for_each(|(obt_block, out_block)| {
            let obt = obt_block * LOOP_UNROLL;
            for o in 0..OC {
                // we'll keep LOOP_UNROLL many results in registers
                let mut result = [0.0f32; LOOP_UNROLL];
                // initialize the bias, if it exists
                for (ibt, r) in result.iter_mut().enumerate() {
                    *r = bias.map_or(0.0f32, |bias| bias[o]);
                    let _ = ibt;
                }
                // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
                // the value of weight[i + o * C] and reuse it.
                // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
                for i in 0..C {
                    let w = weight[i + o * C];
                    for ibt in 0..LOOP_UNROLL {
                        let bt = obt + ibt;
                        result[ibt] += inp[bt * C + i] * w;
                    }
                }
                // write back results to main memory
                for ibt in 0..LOOP_UNROLL {
                    out_block[ibt * OC + o] = result[ibt];
                }
            }
        });
}

#[allow(unused_variables)] // B and T kept for signature parity with the C reference
pub fn matmul_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    // (the C code has `#pragma omp parallel for collapse(2)`)
    dinp.par_chunks_exact_mut(C)
        .enumerate()
        .for_each(|(bt, dinp_bt)| {
            let dout_bt = &dout[bt * OC..(bt + 1) * OC];
            for o in 0..OC {
                let wrow = &weight[o * C..(o + 1) * C];
                let d = dout_bt[o];
                for i in 0..C {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        });
    // backward into weight/bias, parallelize over output channels OC
    // (the C code has `#pragma omp parallel for` over o)
    dweight
        .par_chunks_exact_mut(C)
        .enumerate()
        .for_each(|(o, dwrow)| {
            for b in 0..B {
                for t in 0..T {
                    let bt = b * T + t;
                    let dout_bt = &dout[bt * OC..(bt + 1) * OC];
                    let inp_bt = &inp[bt * C..(bt + 1) * C];
                    let d = dout_bt[o];
                    for i in 0..C {
                        dwrow[i] += inp_bt[i] * d;
                    }
                }
            }
        });
    // accumulate the bias gradients in the same (o, b, t) order as the C code does,
    // keeping the floating point accumulation order identical
    if let Some(dbias) = dbias {
        for o in 0..OC {
            for b in 0..B {
                for t in 0..T {
                    let bt = b * T + t;
                    dbias[o] += dout[bt * OC + o];
                }
            }
        }
    }
}

#[allow(unused_variables)] // B kept for signature parity with the C reference
pub fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    let C3 = C * 3;
    let hs = C / NH; // head size
    let scale = 1.0f32 / (hs as f32).sqrt();

    // the C code has `#pragma omp parallel for collapse(3)` over (b,t,h);
    // all (b,t,h) computations are independent, so we parallelize over b
    // (the natural disjoint chunking of the three output buffers) and
    // keep t,h sequential inside, which is equivalent up to scheduling
    preatt
        .par_chunks_exact_mut(NH * T * T)
        .zip(att.par_chunks_exact_mut(NH * T * T))
        .zip(out.par_chunks_exact_mut(T * C))
        .enumerate()
        .for_each(|(b, ((preatt_b, att_b), out_b))| {
            let inp_b = &inp[b * T * C3..(b + 1) * T * C3];
            for t in 0..T {
                for h in 0..NH {
                    let query_t = &inp_b[t * C3 + h * hs..][..hs];
                    let preatt_bth = &mut preatt_b[h * T * T + t * T..][..T];
                    let att_bth = &mut att_b[h * T * T + t * T..][..T];

                    // pass 1: calculate query dot key and maxval
                    let mut maxval = -10000.0f32; // TODO something better
                    for t2 in 0..=t {
                        let key_t2 = &inp_b[t2 * C3 + h * hs + C..][..hs]; // +C because it's key

                        // (query_t) dot (key_t2)
                        let mut val = 0.0f32;
                        for i in 0..hs {
                            val += query_t[i] * key_t2[i];
                        }
                        val *= scale;
                        if val > maxval {
                            maxval = val;
                        }

                        preatt_bth[t2] = val;
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    let mut expsum = 0.0f32;
                    for t2 in 0..=t {
                        let expv = (preatt_bth[t2] - maxval).exp();
                        expsum += expv;
                        att_bth[t2] = expv;
                    }
                    let expsum_inv = if expsum == 0.0f32 { 0.0f32 } else { 1.0f32 / expsum };

                    // pass 3: normalize to get the softmax
                    for t2 in 0..T {
                        if t2 <= t {
                            att_bth[t2] *= expsum_inv;
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att_bth[t2] = 0.0f32;
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    let out_bth = &mut out_b[t * C + h * hs..][..hs];
                    for i in 0..hs {
                        out_bth[i] = 0.0f32;
                    }
                    for t2 in 0..=t {
                        let value_t2 = &inp_b[t2 * C3 + h * hs + C * 2..][..hs]; // +C*2 because it's value
                        let att_btht2 = att_bth[t2];
                        for i in 0..hs {
                            out_bth[i] += att_btht2 * value_t2[i];
                        }
                    }
                }
            }
        });
}

pub fn attention_backward(
    dinp: &mut [f32],
    dpreatt: &mut [f32],
    datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    let C3 = C * 3;
    let hs = C / NH; // head size
    let scale = 1.0f32 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let att_bth = &att[b * NH * T * T + h * T * T + t * T..][..T];
                let datt_bth = &mut datt[b * NH * T * T + h * T * T + t * T..][..T];
                let dpreatt_bth = &mut dpreatt[b * NH * T * T + h * T * T + t * T..][..T];
                // note: the query/key/value gradient bands live in the same dinp buffer,
                // so (like the C code, which works on raw indices) we address dinp directly
                // instead of holding sub-slices

                // backward pass 4, through the value accumulation
                for t2 in 0..=t {
                    let value_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C * 2..][..hs]; // +C*2 because it's value
                    let dout_bth = &dout[b * T * C + t * C + h * hs..][..hs];
                    for i in 0..hs {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dinp[b * T * C3 + t2 * C3 + h * hs + C * 2 + i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0f32 } else { 0.0f32 };
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                let query_t = &inp[b * T * C3 + t * C3 + h * hs..][..hs];
                for t2 in 0..=t {
                    let key_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C..][..hs]; // +C because it's key
                    for i in 0..hs {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dinp[b * T * C3 + t * C3 + h * hs + i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dinp[b * T * C3 + t2 * C3 + h * hs + C + i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

// the full-precision literal is sqrt(2/pi), matching the C macro exactly
#[allow(clippy::excessive_precision)]
pub const GELU_SCALING_FACTOR: f32 = 0.797_884_560_802_865_4;

pub fn gelu_forward(out: &mut [f32], inp: &[f32], N: usize) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for i in 0..N {
        let x = inp[i];
        let cube = 0.044715f32 * x * x * x;
        out[i] = 0.5f32 * x * (1.0f32 + (GELU_SCALING_FACTOR * (x + cube)).tanh());
    }
}

// Rust's float math is IEEE-strict by default, so the C code's
// `#pragma float_control(precise, on)` workaround for -Ofast (#168) is not needed here
pub fn gelu_backward(dinp: &mut [f32], inp: &[f32], dout: &[f32], N: usize) {
    for i in 0..N {
        let x = inp[i];
        let cube = 0.044715f32 * x * x * x;
        let tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        let tanh_out = tanh_arg.tanh();
        let coshf_out = tanh_arg.cosh();
        let sech_out = 1.0f32 / (coshf_out * coshf_out);
        let local_grad = 0.5f32 * (1.0f32 + tanh_out)
            + x * 0.5f32 * sech_out * GELU_SCALING_FACTOR * (1.0f32 + 3.0f32 * 0.044715f32 * x * x);
        dinp[i] += local_grad * dout[i];
    }
}

pub fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], N: usize) {
    for i in 0..N {
        out[i] = inp1[i] + inp2[i];
    }
}

pub fn residual_backward(dinp1: &mut [f32], dinp2: &mut [f32], dout: &[f32], N: usize) {
    for i in 0..N {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

#[allow(unused_variables)] // B and T kept for signature parity with the C reference
pub fn softmax_forward(probs: &mut [f32], logits: &[f32], B: usize, T: usize, V: usize, Vp: usize) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    // (the C code has `#pragma omp parallel for collapse(2)` over b,t)
    probs
        .par_chunks_exact_mut(Vp)
        .enumerate()
        .for_each(|(bt, probs_bt)| {
            // probs <- softmax(logits)
            let logits_bt = &logits[bt * Vp..(bt + 1) * Vp];

            // maxval is only calculated and subtracted for numerical stability
            let mut maxval = -10000.0f32; // TODO something better
            for i in 0..V {
                if logits_bt[i] > maxval {
                    maxval = logits_bt[i];
                }
            }
            let mut sum = 0.0f32;
            for i in 0..V {
                probs_bt[i] = (logits_bt[i] - maxval).exp();
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for i in 0..V {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for i in V..Vp {
                probs_bt[i] = 0.0f32;
            }
        });
}

pub fn crossentropy_forward(
    losses: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    Vp: usize,
) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for b in 0..B {
        for t in 0..T {
            // loss = -log(probs[target])
            let probs_bt = &probs[b * T * Vp + t * Vp..][..Vp];
            let ix = targets[b * T + t] as usize;
            losses[b * T + t] = -probs_bt[ix].ln();
        }
    }
}

pub fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &[f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    // backwards through both softmax and crossentropy
    for b in 0..B {
        for t in 0..T {
            let dlogits_bt = &mut dlogits[b * T * Vp + t * Vp..][..Vp];
            let probs_bt = &probs[b * T * Vp + t * Vp..][..Vp];
            let dloss = dlosses[b * T + t];
            let ix = targets[b * T + t] as usize;
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for i in 0..V {
                let p = probs_bt[i];
                let indicator = if i == ix { 1.0f32 } else { 0.0f32 };
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

#[derive(Clone, Copy, Debug)]
pub struct GPT2Config {
    pub max_seq_len: usize, // max sequence length, e.g. 1024
    pub vocab_size: usize, // vocab size, e.g. 50257
    pub padded_vocab_size: usize, // padded to e.g. %128==0, 50304
    pub num_layers: usize, // number of layers, e.g. 12
    pub num_heads: usize, // number of heads in attention, e.g. 12
    pub channels: usize, // number of channels, e.g. 768
}

/// a (start, len) view of one tensor inside the single big allocation,
/// the safe equivalent of the C code's pointers into params_memory / acts_memory
#[derive(Clone, Copy, Debug)]
pub struct TensorView {
    pub start: usize,
    pub len: usize,
}

impl TensorView {
    pub const EMPTY: TensorView = TensorView { start: 0, len: 0 };

    /// the (start, len) range of this tensor at `offset` elements in, `len` elements long
    pub fn range(&self, offset: usize, len: usize) -> (usize, usize) {
        (self.start + offset, len)
    }

    /// a shared slice of this tensor at `offset` elements in, `len` elements long
    pub fn slice<'a>(&self, buf: &'a [f32], offset: usize, len: usize) -> &'a [f32] {
        &buf[self.start + offset..self.start + offset + len]
    }
}

/// carve N disjoint (start, len) ranges out of one buffer as mutable slices;
/// this is what lets us keep the C code's "point many tensors into one allocation"
/// design in 100% safe Rust. Panics if the ranges overlap or run out of bounds.
pub(crate) fn split_disjoint<'a, const N: usize>(
    buf: &'a mut [f32],
    ranges: [(usize, usize); N],
) -> [&'a mut [f32]; N] {
    for &(start, len) in &ranges {
        assert!(start + len <= buf.len(), "tensor view out of bounds");
    }
    // walk the ranges left to right, slicing each one off the remainder
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by_key(|&i| ranges[i].0);
    let mut slots: Vec<Option<&'a mut [f32]>> = (0..N).map(|_| None).collect();
    let mut rest = buf;
    let mut prev_end = 0;
    for &i in &order {
        let (start, len) = ranges[i];
        assert!(start >= prev_end, "tensor views must be pairwise disjoint");
        let (_, after) = rest.split_at_mut(start - prev_end);
        let (slice, tail) = after.split_at_mut(len);
        slots[i] = Some(slice);
        rest = tail;
        prev_end = start + len;
    }
    let mut result: [Option<&'a mut [f32]>; N] = std::array::from_fn(|_| None);
    for (i, slot) in slots.into_iter().enumerate() {
        result[i] = slot;
    }
    result.map(|slot| slot.expect("slot filled"))
}

// the parameters of the model
pub const NUM_PARAMETER_TENSORS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct ParameterTensors {
    pub wte: TensorView, // (V, C)
    pub wpe: TensorView, // (maxT, C)
    pub ln1w: TensorView, // (L, C)
    pub ln1b: TensorView, // (L, C)
    pub qkvw: TensorView, // (L, 3*C, C)
    pub qkvb: TensorView, // (L, 3*C)
    pub attprojw: TensorView, // (L, C, C)
    pub attprojb: TensorView, // (L, C)
    pub ln2w: TensorView, // (L, C)
    pub ln2b: TensorView, // (L, C)
    pub fcw: TensorView, // (L, 4*C, C)
    pub fcb: TensorView, // (L, 4*C)
    pub fcprojw: TensorView, // (L, C, 4*C)
    pub fcprojb: TensorView, // (L, C)
    pub lnfw: TensorView, // (C)
    pub lnfb: TensorView, // (C)
}

impl ParameterTensors {
    const EMPTY: ParameterTensors = ParameterTensors {
        wte: TensorView::EMPTY,
        wpe: TensorView::EMPTY,
        ln1w: TensorView::EMPTY,
        ln1b: TensorView::EMPTY,
        qkvw: TensorView::EMPTY,
        qkvb: TensorView::EMPTY,
        attprojw: TensorView::EMPTY,
        attprojb: TensorView::EMPTY,
        ln2w: TensorView::EMPTY,
        ln2b: TensorView::EMPTY,
        fcw: TensorView::EMPTY,
        fcb: TensorView::EMPTY,
        fcprojw: TensorView::EMPTY,
        fcprojb: TensorView::EMPTY,
        lnfw: TensorView::EMPTY,
        lnfb: TensorView::EMPTY,
    };
}

pub fn fill_in_parameter_sizes(param_sizes: &mut [usize; NUM_PARAMETER_TENSORS], config: &GPT2Config) {
    let Vp = config.padded_vocab_size;
    let C = config.channels;
    let maxT = config.max_seq_len;
    let L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

/// allocate memory for the parameters and point the individual tensors to the right places
pub fn malloc_and_point_parameters(param_sizes: &[usize; NUM_PARAMETER_TENSORS]) -> (Vec<f32>, ParameterTensors) {
    let num_parameters: usize = param_sizes.iter().sum();
    // malloc all parameters all at once
    let params_memory = vec![0.0f32; num_parameters];
    // assign all the tensors
    let mut t = ParameterTensors::EMPTY;
    let mut it = 0usize;
    t.wte = TensorView { start: it, len: param_sizes[0] };
    it += param_sizes[0];
    t.wpe = TensorView { start: it, len: param_sizes[1] };
    it += param_sizes[1];
    t.ln1w = TensorView { start: it, len: param_sizes[2] };
    it += param_sizes[2];
    t.ln1b = TensorView { start: it, len: param_sizes[3] };
    it += param_sizes[3];
    t.qkvw = TensorView { start: it, len: param_sizes[4] };
    it += param_sizes[4];
    t.qkvb = TensorView { start: it, len: param_sizes[5] };
    it += param_sizes[5];
    t.attprojw = TensorView { start: it, len: param_sizes[6] };
    it += param_sizes[6];
    t.attprojb = TensorView { start: it, len: param_sizes[7] };
    it += param_sizes[7];
    t.ln2w = TensorView { start: it, len: param_sizes[8] };
    it += param_sizes[8];
    t.ln2b = TensorView { start: it, len: param_sizes[9] };
    it += param_sizes[9];
    t.fcw = TensorView { start: it, len: param_sizes[10] };
    it += param_sizes[10];
    t.fcb = TensorView { start: it, len: param_sizes[11] };
    it += param_sizes[11];
    t.fcprojw = TensorView { start: it, len: param_sizes[12] };
    it += param_sizes[12];
    t.fcprojb = TensorView { start: it, len: param_sizes[13] };
    it += param_sizes[13];
    t.lnfw = TensorView { start: it, len: param_sizes[14] };
    it += param_sizes[14];
    t.lnfb = TensorView { start: it, len: param_sizes[15] };
    (params_memory, t)
}

pub const NUM_ACTIVATION_TENSORS: usize = 23;

#[derive(Clone, Copy, Debug)]
pub struct ActivationTensors {
    pub encoded: TensorView, // (B, T, C)
    pub ln1: TensorView, // (L, B, T, C)
    pub ln1_mean: TensorView, // (L, B, T)
    pub ln1_rstd: TensorView, // (L, B, T)
    pub qkv: TensorView, // (L, B, T, 3*C)
    pub atty: TensorView, // (L, B, T, C)
    pub preatt: TensorView, // (L, B, NH, T, T)
    pub att: TensorView, // (L, B, NH, T, T)
    pub attproj: TensorView, // (L, B, T, C)
    pub residual2: TensorView, // (L, B, T, C)
    pub ln2: TensorView, // (L, B, T, C)
    pub ln2_mean: TensorView, // (L, B, T)
    pub ln2_rstd: TensorView, // (L, B, T)
    pub fch: TensorView, // (L, B, T, 4*C)
    pub fch_gelu: TensorView, // (L, B, T, 4*C)
    pub fcproj: TensorView, // (L, B, T, C)
    pub residual3: TensorView, // (L, B, T, C)
    pub lnf: TensorView, // (B, T, C)
    pub lnf_mean: TensorView, // (B, T)
    pub lnf_rstd: TensorView, // (B, T)
    pub logits: TensorView, // (B, T, V)
    pub probs: TensorView, // (B, T, V)
    pub losses: TensorView, // (B, T)
}

impl ActivationTensors {
    const EMPTY: ActivationTensors = ActivationTensors {
        encoded: TensorView::EMPTY,
        ln1: TensorView::EMPTY,
        ln1_mean: TensorView::EMPTY,
        ln1_rstd: TensorView::EMPTY,
        qkv: TensorView::EMPTY,
        atty: TensorView::EMPTY,
        preatt: TensorView::EMPTY,
        att: TensorView::EMPTY,
        attproj: TensorView::EMPTY,
        residual2: TensorView::EMPTY,
        ln2: TensorView::EMPTY,
        ln2_mean: TensorView::EMPTY,
        ln2_rstd: TensorView::EMPTY,
        fch: TensorView::EMPTY,
        fch_gelu: TensorView::EMPTY,
        fcproj: TensorView::EMPTY,
        residual3: TensorView::EMPTY,
        lnf: TensorView::EMPTY,
        lnf_mean: TensorView::EMPTY,
        lnf_rstd: TensorView::EMPTY,
        logits: TensorView::EMPTY,
        probs: TensorView::EMPTY,
        losses: TensorView::EMPTY,
    };
}

pub fn fill_in_activation_sizes(
    act_sizes: &mut [usize; NUM_ACTIVATION_TENSORS],
    config: &GPT2Config,
    B: usize,
    T: usize,
) {
    let C = config.channels;
    let NH = config.num_heads;
    let L = config.num_layers;
    let Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

/// allocate memory for the activations and point the individual tensors to the right places
pub fn malloc_and_point_activations(
    act_sizes: &[usize; NUM_ACTIVATION_TENSORS],
) -> (Vec<f32>, ActivationTensors) {
    let num_activations: usize = act_sizes.iter().sum();
    let acts_memory = vec![0.0f32; num_activations];
    let mut t = ActivationTensors::EMPTY;
    let mut it = 0usize;
    t.encoded = TensorView { start: it, len: act_sizes[0] };
    it += act_sizes[0];
    t.ln1 = TensorView { start: it, len: act_sizes[1] };
    it += act_sizes[1];
    t.ln1_mean = TensorView { start: it, len: act_sizes[2] };
    it += act_sizes[2];
    t.ln1_rstd = TensorView { start: it, len: act_sizes[3] };
    it += act_sizes[3];
    t.qkv = TensorView { start: it, len: act_sizes[4] };
    it += act_sizes[4];
    t.atty = TensorView { start: it, len: act_sizes[5] };
    it += act_sizes[5];
    t.preatt = TensorView { start: it, len: act_sizes[6] };
    it += act_sizes[6];
    t.att = TensorView { start: it, len: act_sizes[7] };
    it += act_sizes[7];
    t.attproj = TensorView { start: it, len: act_sizes[8] };
    it += act_sizes[8];
    t.residual2 = TensorView { start: it, len: act_sizes[9] };
    it += act_sizes[9];
    t.ln2 = TensorView { start: it, len: act_sizes[10] };
    it += act_sizes[10];
    t.ln2_mean = TensorView { start: it, len: act_sizes[11] };
    it += act_sizes[11];
    t.ln2_rstd = TensorView { start: it, len: act_sizes[12] };
    it += act_sizes[12];
    t.fch = TensorView { start: it, len: act_sizes[13] };
    it += act_sizes[13];
    t.fch_gelu = TensorView { start: it, len: act_sizes[14] };
    it += act_sizes[14];
    t.fcproj = TensorView { start: it, len: act_sizes[15] };
    it += act_sizes[15];
    t.residual3 = TensorView { start: it, len: act_sizes[16] };
    it += act_sizes[16];
    t.lnf = TensorView { start: it, len: act_sizes[17] };
    it += act_sizes[17];
    t.lnf_mean = TensorView { start: it, len: act_sizes[18] };
    it += act_sizes[18];
    t.lnf_rstd = TensorView { start: it, len: act_sizes[19] };
    it += act_sizes[19];
    t.logits = TensorView { start: it, len: act_sizes[20] };
    it += act_sizes[20];
    t.probs = TensorView { start: it, len: act_sizes[21] };
    it += act_sizes[21];
    t.losses = TensorView { start: it, len: act_sizes[22] };
    (acts_memory, t)
}

pub struct GPT2 {
    pub config: GPT2Config,
    // the weights (parameters) of the model, and their sizes
    pub params: ParameterTensors,
    pub param_sizes: [usize; NUM_PARAMETER_TENSORS],
    pub params_memory: Vec<f32>,
    pub num_parameters: usize,
    // gradients of the weights (allocated lazily in gpt2_backward)
    pub grads: Option<ParameterTensors>,
    pub grads_memory: Option<Vec<f32>>,
    // buffers for the AdamW optimizer (allocated lazily in gpt2_update)
    pub m_memory: Option<Vec<f32>>,
    pub v_memory: Option<Vec<f32>>,
    // the activations of the model, and their sizes (allocated lazily in gpt2_forward)
    pub acts: Option<ActivationTensors>,
    pub act_sizes: [usize; NUM_ACTIVATION_TENSORS],
    pub acts_memory: Option<Vec<f32>>,
    pub num_activations: usize,
    // gradients of the activations (allocated lazily in gpt2_backward)
    pub grads_acts: Option<ActivationTensors>,
    pub grads_acts_memory: Option<Vec<f32>>,
    // other run state configuration
    pub batch_size: usize, // the batch size (B) of current forward pass
    pub seq_len: usize, // the sequence length (T) of current forward pass
    pub inputs: Vec<i32>, // the input tokens for the current forward pass
    pub targets: Vec<i32>, // the target tokens for the current forward pass
    pub mean_loss: f32, // after a forward pass with targets, will be populated with the mean loss
}

impl GPT2 {
    /// port of gpt2_build_from_checkpoint()
    pub fn build_from_checkpoint(checkpoint_path: &str) -> GPT2 {
        use std::process::exit;

        // read in model from a checkpoint file
        let mut model_file = fopen_check(checkpoint_path, "rb");
        let model_header = read_i32s(&mut model_file, 256);
        if model_header[0] != 20240326 {
            println!("Bad magic model file");
            exit(1);
        }
        if model_header[1] != 3 {
            println!("Bad version in model file");
            println!("---> HINT: try to re-run `python train_gpt2.py`");
            exit(1);
        }

        // read in hyperparameters
        let maxT = model_header[2] as usize;
        let V = model_header[3] as usize;
        let L = model_header[4] as usize;
        let NH = model_header[5] as usize;
        let C = model_header[6] as usize;
        let Vp = model_header[7] as usize;
        let config = GPT2Config {
            max_seq_len: maxT,
            vocab_size: V,
            num_layers: L,
            num_heads: NH,
            channels: C,
            padded_vocab_size: Vp,
        };
        println!("[GPT-2]");
        println!("max_seq_len: {maxT}");
        println!("vocab_size: {V}");
        println!("padded_vocab_size: {Vp}");
        println!("num_layers: {L}");
        println!("num_heads: {NH}");
        println!("channels: {C}");

        // allocate space for all the parameters and read them in
        let mut param_sizes = [0usize; NUM_PARAMETER_TENSORS];
        fill_in_parameter_sizes(&mut param_sizes, &config);

        // count the number of parameters
        let num_parameters: usize = param_sizes.iter().sum();
        println!("num_parameters: {num_parameters}");

        // read in all the parameters from file
        let (mut params_memory, params) = malloc_and_point_parameters(&param_sizes);
        let read_params = read_f32s(&mut model_file, num_parameters);
        params_memory.copy_from_slice(&read_params);
        // (model_file closed on drop)

        // other inits
        GPT2 {
            config,
            params,
            param_sizes,
            params_memory,
            num_parameters,
            grads: None,
            grads_memory: None,
            m_memory: None,
            v_memory: None,
            acts: None,
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: None,
            num_activations: 0,
            grads_acts: None,
            grads_acts_memory: None,
            batch_size: 0,
            seq_len: 0,
            inputs: Vec::new(),
            targets: Vec::new(),
            mean_loss: -1.0f32, // -1.0f will designate no loss
        }
    }

    /// port of gpt2_forward(); targets are optional
    pub fn gpt2_forward(&mut self, inputs: &[i32], targets: Option<&[i32]>, B: usize, T: usize) {
        use std::process::exit;

        // convenience parameters (size_t to help prevent int overflow)
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // validate inputs, all indices must be in the range [0, V)
        for i in 0..B * T {
            assert!(0 <= inputs[i] && (inputs[i] as usize) < V);
            if let Some(targets) = targets {
                assert!(0 <= targets[i] && (targets[i] as usize) < V);
            }
        }

        // allocate space for all the activations if needed (done here, lazily)
        if self.acts_memory.is_none() {
            // record the current B,T as well
            self.batch_size = B;
            self.seq_len = T;
            // and now allocate the space
            fill_in_activation_sizes(&mut self.act_sizes, &self.config, B, T);
            let num_activations: usize = self.act_sizes.iter().sum();
            println!("num_activations: {num_activations}");
            self.num_activations = num_activations;
            let (acts_memory, acts) = malloc_and_point_activations(&self.act_sizes);
            self.acts_memory = Some(acts_memory);
            self.acts = Some(acts);
            // also create memory for caching inputs and targets
            self.inputs = vec![0; B * T];
            self.targets = vec![0; B * T]; // might be unused if we never have targets but it's small
        } else {
            // validate B,T is consistent with how we've allocated the memory before
            // in principle we could get more clever here in the future, for now this is safest
            if B != self.batch_size || T != self.seq_len {
                println!("Model: B={} T={}, Desired: B={} T={}", self.batch_size, self.seq_len, B, T);
                exit(1);
            }
        }

        // cache the inputs/targets
        self.inputs.copy_from_slice(&inputs[..B * T]);
        if let Some(targets) = targets {
            self.targets.copy_from_slice(&targets[..B * T]);
        }

        // forward pass
        let params_memory = &self.params_memory[..];
        let p = self.params; // for brevity
        let acts_memory = self.acts_memory.as_mut().unwrap();
        let a = self.acts.unwrap();

        encoder_forward(
            split_disjoint(acts_memory, [a.encoded.range(0, B * T * C)])[0],
            inputs,
            p.wte.slice(params_memory, 0, Vp * C),
            p.wpe.slice(params_memory, 0, T * C),
            B,
            T,
            C,
        ); // encoding goes into residual[0]
        for l in 0..L {
            // get the views of the activations for this layer
            // (all disjoint inside acts_memory; split_disjoint checks that)
            let residual_range =
                if l == 0 { a.encoded.range(0, B * T * C) } else { a.residual3.range((l - 1) * B * T * C, B * T * C) };
            let [l_ln1, l_ln1_mean, l_ln1_rstd, l_qkv, l_atty, l_preatt, l_att, l_attproj, l_residual2,
                l_ln2, l_ln2_mean, l_ln2_rstd, l_fch, l_fch_gelu, l_fcproj, l_residual3, residual] =
                split_disjoint(acts_memory, [
                    a.ln1.range(l * B * T * C, B * T * C),
                    a.ln1_mean.range(l * B * T, B * T),
                    a.ln1_rstd.range(l * B * T, B * T),
                    a.qkv.range(l * B * T * 3 * C, B * T * 3 * C),
                    a.atty.range(l * B * T * C, B * T * C),
                    a.preatt.range(l * B * NH * T * T, B * NH * T * T),
                    a.att.range(l * B * NH * T * T, B * NH * T * T),
                    a.attproj.range(l * B * T * C, B * T * C),
                    a.residual2.range(l * B * T * C, B * T * C),
                    a.ln2.range(l * B * T * C, B * T * C),
                    a.ln2_mean.range(l * B * T, B * T),
                    a.ln2_rstd.range(l * B * T, B * T),
                    a.fch.range(l * B * T * 4 * C, B * T * 4 * C),
                    a.fch_gelu.range(l * B * T * 4 * C, B * T * 4 * C),
                    a.fcproj.range(l * B * T * C, B * T * C),
                    a.residual3.range(l * B * T * C, B * T * C),
                    residual_range,
                ]);

            // get the pointers of the weights for this layer
            let l_ln1w = p.ln1w.slice(params_memory, l * C, C);
            let l_ln1b = p.ln1b.slice(params_memory, l * C, C);
            let l_qkvw = p.qkvw.slice(params_memory, l * 3 * C * C, 3 * C * C);
            let l_qkvb = p.qkvb.slice(params_memory, l * 3 * C, 3 * C);
            let l_attprojw = p.attprojw.slice(params_memory, l * C * C, C * C);
            let l_attprojb = p.attprojb.slice(params_memory, l * C, C);
            let l_ln2w = p.ln2w.slice(params_memory, l * C, C);
            let l_ln2b = p.ln2b.slice(params_memory, l * C, C);
            let l_fcw = p.fcw.slice(params_memory, l * 4 * C * C, 4 * C * C);
            let l_fcb = p.fcb.slice(params_memory, l * 4 * C, 4 * C);
            let l_fcprojw = p.fcprojw.slice(params_memory, l * C * 4 * C, C * 4 * C);
            let l_fcprojb = p.fcprojb.slice(params_memory, l * C, C);

            // now do the forward pass
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(l_qkv, l_ln1, l_qkvw, Some(l_qkvb), B, T, C, 3 * C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, Some(l_attprojb), B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, Some(l_fcb), B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, Some(l_fcprojb), B, T, 4 * C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        // last residual is in residual3
        let [lnf, lnf_mean, lnf_rstd, logits, probs, losses, residual] = split_disjoint(acts_memory, [
            a.lnf.range(0, B * T * C),
            a.lnf_mean.range(0, B * T),
            a.lnf_rstd.range(0, B * T),
            a.logits.range(0, B * T * Vp),
            a.probs.range(0, B * T * Vp),
            a.losses.range(0, B * T),
            a.residual3.range((L - 1) * B * T * C, B * T * C),
        ]);
        layernorm_forward(
            lnf,
            lnf_mean,
            lnf_rstd,
            residual,
            p.lnfw.slice(params_memory, 0, C),
            p.lnfb.slice(params_memory, 0, C),
            B,
            T,
            C,
        );
        matmul_forward(logits, lnf, p.wte.slice(params_memory, 0, Vp * C), None, B, T, C, Vp);
        softmax_forward(probs, logits, B, T, V, Vp);

        // also forward the cross-entropy loss function if we have the targets
        let mean_loss;
        if let Some(targets) = targets {
            crossentropy_forward(losses, probs, targets, B, T, Vp);
            // for convenience also evaluate the mean loss
            let mut m = 0.0f32;
            for i in 0..B * T {
                m += losses[i];
            }
            m /= (B * T) as f32;
            mean_loss = m;
        } else {
            // if we don't have targets, we don't have a loss
            mean_loss = -1.0f32;
        }
        self.mean_loss = mean_loss;
    }

    /// port of gpt2_zero_grad()
    pub fn gpt2_zero_grad(&mut self) {
        if let Some(grads_memory) = &mut self.grads_memory {
            grads_memory.fill(0.0);
        }
        if let Some(grads_acts_memory) = &mut self.grads_acts_memory {
            grads_acts_memory.fill(0.0);
        }
    }

    /// port of gpt2_backward()
    pub fn gpt2_backward(&mut self) {
        use std::process::exit;

        // double check we forwarded previously, with targets
        if self.mean_loss == -1.0f32 {
            println!("Error: must forward with targets before backward");
            exit(1);
        }

        // lazily allocate the memory for gradients of the weights and activations, if needed
        if self.grads_memory.is_none() {
            let (grads_memory, grads) = malloc_and_point_parameters(&self.param_sizes);
            self.grads_memory = Some(grads_memory);
            self.grads = Some(grads);
            let (grads_acts_memory, grads_acts) = malloc_and_point_activations(&self.act_sizes);
            self.grads_acts_memory = Some(grads_acts_memory);
            self.grads_acts = Some(grads_acts);
            self.gpt2_zero_grad();
        }

        // convenience shortcuts (and size_t to help prevent int overflow)
        let B = self.batch_size;
        let T = self.seq_len;
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // backward pass: go in the reverse order of the forward pass, and call backward() functions
        let params_memory = &self.params_memory[..];
        let acts_memory = self.acts_memory.as_ref().unwrap();
        let p = self.params; // for brevity
        let a = self.acts.unwrap();
        let grads_memory = self.grads_memory.as_mut().unwrap();
        let g = self.grads.unwrap();
        let grads_acts_memory = self.grads_acts_memory.as_mut().unwrap();
        let ga = self.grads_acts.unwrap();
        let targets = &self.targets[..];
        let inputs = &self.inputs[..];

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        let dloss_mean = 1.0f32 / (B * T) as f32;
        {
            let [dlosses] = split_disjoint(grads_acts_memory, [ga.losses.range(0, B * T)]);
            for i in 0..B * T {
                dlosses[i] = dloss_mean;
            }
        }

        // the final layernorm and classifier, before the layer loop
        {
            let [dlogits, dlosses, dlnf, dresidual] = split_disjoint(grads_acts_memory, [
                ga.logits.range(0, B * T * Vp),
                ga.losses.range(0, B * T),
                ga.lnf.range(0, B * T * C),
                ga.residual3.range((L - 1) * B * T * C, B * T * C),
            ]);
            let probs = a.probs.slice(acts_memory, 0, B * T * Vp);
            crossentropy_softmax_backward(dlogits, dlosses, probs, targets, B, T, V, Vp);
            let [dwte, dlnfw, dlnfb] = split_disjoint(grads_memory, [
                g.wte.range(0, Vp * C),
                g.lnfw.range(0, C),
                g.lnfb.range(0, C),
            ]);
            matmul_backward(
                dlnf,
                dwte,
                None,
                dlogits,
                a.lnf.slice(acts_memory, 0, B * T * C),
                p.wte.slice(params_memory, 0, Vp * C),
                B,
                T,
                C,
                Vp,
            );
            layernorm_backward(
                dresidual,
                dlnfw,
                dlnfb,
                dlnf,
                a.residual3.slice(acts_memory, (L - 1) * B * T * C, B * T * C),
                p.lnfw.slice(params_memory, 0, C),
                a.lnf_mean.slice(acts_memory, 0, B * T),
                a.lnf_rstd.slice(acts_memory, 0, B * T),
                B,
                T,
                C,
            );
        }

        for l in (0..L).rev() {
            let dresidual_range =
                if l == 0 { ga.encoded.range(0, B * T * C) } else { ga.residual3.range((l - 1) * B * T * C, B * T * C) };

            // get the views of the activations for this layer (reads come from acts_memory)
            let l_ln1 = a.ln1.slice(acts_memory, l * B * T * C, B * T * C);
            let l_ln1_mean = a.ln1_mean.slice(acts_memory, l * B * T, B * T);
            let l_ln1_rstd = a.ln1_rstd.slice(acts_memory, l * B * T, B * T);
            let l_qkv = a.qkv.slice(acts_memory, l * B * T * 3 * C, B * T * 3 * C);
            let l_atty = a.atty.slice(acts_memory, l * B * T * C, B * T * C);
            let l_att = a.att.slice(acts_memory, l * B * NH * T * T, B * NH * T * T);
            let l_residual2 = a.residual2.slice(acts_memory, l * B * T * C, B * T * C);
            let l_ln2 = a.ln2.slice(acts_memory, l * B * T * C, B * T * C);
            let l_ln2_mean = a.ln2_mean.slice(acts_memory, l * B * T, B * T);
            let l_ln2_rstd = a.ln2_rstd.slice(acts_memory, l * B * T, B * T);
            let l_fch = a.fch.slice(acts_memory, l * B * T * 4 * C, B * T * 4 * C);
            let l_fch_gelu = a.fch_gelu.slice(acts_memory, l * B * T * 4 * C, B * T * 4 * C);
            let residual = if l == 0 {
                a.encoded.slice(acts_memory, 0, B * T * C)
            } else {
                a.residual3.slice(acts_memory, (l - 1) * B * T * C, B * T * C)
            };

            // get the views of the gradients of the activations for this layer
            let [dl_ln1, dl_qkv, dl_atty, dl_preatt, dl_att, dl_attproj, dl_residual2, dl_ln2,
                dl_fch, dl_fch_gelu, dl_fcproj, dl_residual3, dresidual] = split_disjoint(
                grads_acts_memory,
                [
                    ga.ln1.range(l * B * T * C, B * T * C),
                    ga.qkv.range(l * B * T * 3 * C, B * T * 3 * C),
                    ga.atty.range(l * B * T * C, B * T * C),
                    ga.preatt.range(l * B * NH * T * T, B * NH * T * T),
                    ga.att.range(l * B * NH * T * T, B * NH * T * T),
                    ga.attproj.range(l * B * T * C, B * T * C),
                    ga.residual2.range(l * B * T * C, B * T * C),
                    ga.ln2.range(l * B * T * C, B * T * C),
                    ga.fch.range(l * B * T * 4 * C, B * T * 4 * C),
                    ga.fch_gelu.range(l * B * T * 4 * C, B * T * 4 * C),
                    ga.fcproj.range(l * B * T * C, B * T * C),
                    ga.residual3.range(l * B * T * C, B * T * C),
                    dresidual_range,
                ],
            );

            // get the views of the gradients of the weights for this layer
            let [dl_ln1w, dl_ln1b, dl_qkvw, dl_qkvb, dl_attprojw, dl_attprojb, dl_ln2w, dl_ln2b,
                dl_fcw, dl_fcb, dl_fcprojw, dl_fcprojb] = split_disjoint(grads_memory, [
                g.ln1w.range(l * C, C),
                g.ln1b.range(l * C, C),
                g.qkvw.range(l * 3 * C * C, 3 * C * C),
                g.qkvb.range(l * 3 * C, 3 * C),
                g.attprojw.range(l * C * C, C * C),
                g.attprojb.range(l * C, C),
                g.ln2w.range(l * C, C),
                g.ln2b.range(l * C, C),
                g.fcw.range(l * 4 * C * C, 4 * C * C),
                g.fcb.range(l * 4 * C, 4 * C),
                g.fcprojw.range(l * C * 4 * C, C * 4 * C),
                g.fcprojb.range(l * C, C),
            ]);

            // get the pointers of the weights for this layer
            let l_ln1w = p.ln1w.slice(params_memory, l * C, C);
            let l_qkvw = p.qkvw.slice(params_memory, l * 3 * C * C, 3 * C * C);
            let l_attprojw = p.attprojw.slice(params_memory, l * C * C, C * C);
            let l_ln2w = p.ln2w.slice(params_memory, l * C, C);
            let l_fcw = p.fcw.slice(params_memory, l * 4 * C * C, 4 * C * C);
            let l_fcprojw = p.fcprojw.slice(params_memory, l * C * 4 * C, C * 4 * C);

            // backprop this layer
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
            matmul_backward(dl_fch_gelu, dl_fcprojw, Some(dl_fcprojb), dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, dl_fcw, Some(dl_fcb), dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
            layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
            matmul_backward(dl_atty, dl_attprojw, Some(dl_attprojb), dl_attproj, l_atty, l_attprojw, B, T, C, C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
            matmul_backward(dl_ln1, dl_qkvw, Some(dl_qkvb), dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
        }
        {
            // encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C)
            let [dencoded] = split_disjoint(grads_acts_memory, [ga.encoded.range(0, B * T * C)]);
            let [dwte, dwpe] =
                split_disjoint(grads_memory, [g.wte.range(0, Vp * C), g.wpe.range(0, T * C)]);
            encoder_backward(dwte, dwpe, dencoded, inputs, B, T, C);
        }
    }

    /// port of gpt2_update()
    pub fn gpt2_update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        t: i32,
    ) {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        // lazily allocate the memory for m_memory and v_memory
        if self.m_memory.is_none() {
            self.m_memory = Some(vec![0.0f32; self.num_parameters]);
            self.v_memory = Some(vec![0.0f32; self.num_parameters]);
        }

        let GPT2 { params_memory, grads_memory, m_memory, v_memory, num_parameters, .. } = self;
        let grads_memory = grads_memory
            .as_mut()
            .expect("grads not allocated; call gpt2_backward first");
        let m_memory = m_memory.as_mut().unwrap();
        let v_memory = v_memory.as_mut().unwrap();

        for i in 0..*num_parameters {
            let param = params_memory[i];
            let grad = grads_memory[i];

            // update the first moment (momentum)
            let m = beta1 * m_memory[i] + (1.0f32 - beta1) * grad;
            // update the second moment (RMSprop)
            let v = beta2 * v_memory[i] + (1.0f32 - beta2) * grad * grad;
            // bias-correct both moments
            let m_hat = m / (1.0f32 - beta1.powf(t as f32));
            let v_hat = v / (1.0f32 - beta2.powf(t as f32));

            // update
            m_memory[i] = m;
            v_memory[i] = v;
            params_memory[i] -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
        }
    }
}

// gpt2_free() has no Rust equivalent: all model memory is freed when the GPT2 is dropped
