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
