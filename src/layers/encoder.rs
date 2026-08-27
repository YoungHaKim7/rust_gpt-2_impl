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
