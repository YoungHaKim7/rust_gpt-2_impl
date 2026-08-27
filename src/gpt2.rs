// the GPT2 model itself: weights, gradients, activations, and the
// forward / backward / update passes that tie all the layers together.
// defines: GPT2 (build_from_checkpoint, gpt2_forward, gpt2_zero_grad,
//                  gpt2_backward, gpt2_update)

use crate::llmc::utils::{fopen_check, read_f32s, read_i32s};
use crate::layers::*;
use crate::model::{
    ActivationTensors, GPT2Config, NUM_ACTIVATION_TENSORS, NUM_PARAMETER_TENSORS, ParameterTensors,
    fill_in_activation_sizes, fill_in_parameter_sizes, malloc_and_point_activations,
    malloc_and_point_parameters,
};
use crate::tensor::split_disjoint;

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
