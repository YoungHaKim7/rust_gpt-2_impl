use rayon::prelude::*;

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
