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
