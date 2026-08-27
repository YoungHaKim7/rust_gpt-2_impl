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
