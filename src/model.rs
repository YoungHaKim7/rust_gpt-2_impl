// GPT-2 model definition: the config, the parameter/activation tensor layouts,
// and the size-filling + allocation helpers that point tensors into one big Vec.
// defines: GPT2Config, ParameterTensors, ActivationTensors,
//          fill_in_parameter_sizes, fill_in_activation_sizes,
//          malloc_and_point_parameters, malloc_and_point_activations

use crate::tensor::TensorView;

#[derive(Clone, Copy, Debug)]
pub struct GPT2Config {
    pub max_seq_len: usize,       // max sequence length, e.g. 1024
    pub vocab_size: usize,        // vocab size, e.g. 50257
    pub padded_vocab_size: usize, // padded to e.g. %128==0, 50304
    pub num_layers: usize,        // number of layers, e.g. 12
    pub num_heads: usize,         // number of heads in attention, e.g. 12
    pub channels: usize,          // number of channels, e.g. 768
}

// the parameters of the model
pub const NUM_PARAMETER_TENSORS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct ParameterTensors {
    pub wte: TensorView,      // (V, C)
    pub wpe: TensorView,      // (maxT, C)
    pub ln1w: TensorView,     // (L, C)
    pub ln1b: TensorView,     // (L, C)
    pub qkvw: TensorView,     // (L, 3*C, C)
    pub qkvb: TensorView,     // (L, 3*C)
    pub attprojw: TensorView, // (L, C, C)
    pub attprojb: TensorView, // (L, C)
    pub ln2w: TensorView,     // (L, C)
    pub ln2b: TensorView,     // (L, C)
    pub fcw: TensorView,      // (L, 4*C, C)
    pub fcb: TensorView,      // (L, 4*C)
    pub fcprojw: TensorView,  // (L, C, 4*C)
    pub fcprojb: TensorView,  // (L, C)
    pub lnfw: TensorView,     // (C)
    pub lnfb: TensorView,     // (C)
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

pub fn fill_in_parameter_sizes(
    param_sizes: &mut [usize; NUM_PARAMETER_TENSORS],
    config: &GPT2Config,
) {
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
pub fn malloc_and_point_parameters(
    param_sizes: &[usize; NUM_PARAMETER_TENSORS],
) -> (Vec<f32>, ParameterTensors) {
    let num_parameters: usize = param_sizes.iter().sum();
    // malloc all parameters all at once
    let params_memory = vec![0.0f32; num_parameters];
    // assign all the tensors
    let mut t = ParameterTensors::EMPTY;
    let mut it = 0usize;
    t.wte = TensorView {
        start: it,
        len: param_sizes[0],
    };
    it += param_sizes[0];
    t.wpe = TensorView {
        start: it,
        len: param_sizes[1],
    };
    it += param_sizes[1];
    t.ln1w = TensorView {
        start: it,
        len: param_sizes[2],
    };
    it += param_sizes[2];
    t.ln1b = TensorView {
        start: it,
        len: param_sizes[3],
    };
    it += param_sizes[3];
    t.qkvw = TensorView {
        start: it,
        len: param_sizes[4],
    };
    it += param_sizes[4];
    t.qkvb = TensorView {
        start: it,
        len: param_sizes[5],
    };
    it += param_sizes[5];
    t.attprojw = TensorView {
        start: it,
        len: param_sizes[6],
    };
    it += param_sizes[6];
    t.attprojb = TensorView {
        start: it,
        len: param_sizes[7],
    };
    it += param_sizes[7];
    t.ln2w = TensorView {
        start: it,
        len: param_sizes[8],
    };
    it += param_sizes[8];
    t.ln2b = TensorView {
        start: it,
        len: param_sizes[9],
    };
    it += param_sizes[9];
    t.fcw = TensorView {
        start: it,
        len: param_sizes[10],
    };
    it += param_sizes[10];
    t.fcb = TensorView {
        start: it,
        len: param_sizes[11],
    };
    it += param_sizes[11];
    t.fcprojw = TensorView {
        start: it,
        len: param_sizes[12],
    };
    it += param_sizes[12];
    t.fcprojb = TensorView {
        start: it,
        len: param_sizes[13],
    };
    it += param_sizes[13];
    t.lnfw = TensorView {
        start: it,
        len: param_sizes[14],
    };
    it += param_sizes[14];
    t.lnfb = TensorView {
        start: it,
        len: param_sizes[15],
    };
    (params_memory, t)
}

pub const NUM_ACTIVATION_TENSORS: usize = 23;

#[derive(Clone, Copy, Debug)]
pub struct ActivationTensors {
    pub encoded: TensorView,   // (B, T, C)
    pub ln1: TensorView,       // (L, B, T, C)
    pub ln1_mean: TensorView,  // (L, B, T)
    pub ln1_rstd: TensorView,  // (L, B, T)
    pub qkv: TensorView,       // (L, B, T, 3*C)
    pub atty: TensorView,      // (L, B, T, C)
    pub preatt: TensorView,    // (L, B, NH, T, T)
    pub att: TensorView,       // (L, B, NH, T, T)
    pub attproj: TensorView,   // (L, B, T, C)
    pub residual2: TensorView, // (L, B, T, C)
    pub ln2: TensorView,       // (L, B, T, C)
    pub ln2_mean: TensorView,  // (L, B, T)
    pub ln2_rstd: TensorView,  // (L, B, T)
    pub fch: TensorView,       // (L, B, T, 4*C)
    pub fch_gelu: TensorView,  // (L, B, T, 4*C)
    pub fcproj: TensorView,    // (L, B, T, C)
    pub residual3: TensorView, // (L, B, T, C)
    pub lnf: TensorView,       // (B, T, C)
    pub lnf_mean: TensorView,  // (B, T)
    pub lnf_rstd: TensorView,  // (B, T)
    pub logits: TensorView,    // (B, T, V)
    pub probs: TensorView,     // (B, T, V)
    pub losses: TensorView,    // (B, T)
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
    t.encoded = TensorView {
        start: it,
        len: act_sizes[0],
    };
    it += act_sizes[0];
    t.ln1 = TensorView {
        start: it,
        len: act_sizes[1],
    };
    it += act_sizes[1];
    t.ln1_mean = TensorView {
        start: it,
        len: act_sizes[2],
    };
    it += act_sizes[2];
    t.ln1_rstd = TensorView {
        start: it,
        len: act_sizes[3],
    };
    it += act_sizes[3];
    t.qkv = TensorView {
        start: it,
        len: act_sizes[4],
    };
    it += act_sizes[4];
    t.atty = TensorView {
        start: it,
        len: act_sizes[5],
    };
    it += act_sizes[5];
    t.preatt = TensorView {
        start: it,
        len: act_sizes[6],
    };
    it += act_sizes[6];
    t.att = TensorView {
        start: it,
        len: act_sizes[7],
    };
    it += act_sizes[7];
    t.attproj = TensorView {
        start: it,
        len: act_sizes[8],
    };
    it += act_sizes[8];
    t.residual2 = TensorView {
        start: it,
        len: act_sizes[9],
    };
    it += act_sizes[9];
    t.ln2 = TensorView {
        start: it,
        len: act_sizes[10],
    };
    it += act_sizes[10];
    t.ln2_mean = TensorView {
        start: it,
        len: act_sizes[11],
    };
    it += act_sizes[11];
    t.ln2_rstd = TensorView {
        start: it,
        len: act_sizes[12],
    };
    it += act_sizes[12];
    t.fch = TensorView {
        start: it,
        len: act_sizes[13],
    };
    it += act_sizes[13];
    t.fch_gelu = TensorView {
        start: it,
        len: act_sizes[14],
    };
    it += act_sizes[14];
    t.fcproj = TensorView {
        start: it,
        len: act_sizes[15],
    };
    it += act_sizes[15];
    t.residual3 = TensorView {
        start: it,
        len: act_sizes[16],
    };
    it += act_sizes[16];
    t.lnf = TensorView {
        start: it,
        len: act_sizes[17],
    };
    it += act_sizes[17];
    t.lnf_mean = TensorView {
        start: it,
        len: act_sizes[18],
    };
    it += act_sizes[18];
    t.lnf_rstd = TensorView {
        start: it,
        len: act_sizes[19],
    };
    it += act_sizes[19];
    t.logits = TensorView {
        start: it,
        len: act_sizes[20],
    };
    it += act_sizes[20];
    t.probs = TensorView {
        start: it,
        len: act_sizes[21],
    };
    it += act_sizes[21];
    t.losses = TensorView {
        start: it,
        len: act_sizes[22],
    };
    (acts_memory, t)
}
