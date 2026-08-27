use rayon::prelude::*;

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
