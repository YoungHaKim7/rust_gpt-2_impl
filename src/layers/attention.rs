use rayon::prelude::*;

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
                    let expsum_inv = if expsum == 0.0f32 {
                        0.0f32
                    } else {
                        1.0f32 / expsum
                    };

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
                        dinp[b * T * C3 + t2 * C3 + h * hs + C * 2 + i] +=
                            att_bth[t2] * dout_bth[i];
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
                        dinp[b * T * C3 + t * C3 + h * hs + i] +=
                            key_t2[i] * dpreatt_bth[t2] * scale;
                        dinp[b * T * C3 + t2 * C3 + h * hs + C + i] +=
                            query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}
