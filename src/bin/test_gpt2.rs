/*
Port of llm.c/test_gpt2.c: checks the forward and backward passes against the
reference values exported by `python train_gpt2.py` into gpt2_124M_debug_state.bin.
*/

#![allow(non_snake_case)]
// the expected_losses literals are copied verbatim from test_gpt2.c, where they
// are the exact f32 bit patterns exported by PyTorch — do not round them
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)] // the loops mirror the C reference index-by-index

use rust_gpt_2_impl::{
    llmc::utils::{read_f32s, read_i32s},
    {GPT2, malloc_and_point_parameters},
};

// poor man's tensor checker
fn check_tensor(a: &[f32], b: &[f32], label: &str) -> bool {
    let print_upto = 5;
    let n = a.len().min(b.len());
    let mut ok = true;
    let mut maxdiff = 0.0f32;
    let tol = 2e-2f32;
    println!("{label}");
    for i in 0..n {
        // look at the diffence at position i of these two tensors
        let diff = (a[i] - b[i]).abs();

        // keep track of the overall error
        ok = ok && (diff <= tol);
        if diff > maxdiff {
            maxdiff = diff;
        }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        if i < print_upto {
            if diff <= tol {
                print!("OK ");
            } else {
                print!("NOT OK ");
            }
            println!("{:.6} {:.6}", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    if ok {
        println!("TENSOR OK, maxdiff = {maxdiff:e}");
    } else {
        println!("TENSOR NOT OK, maxdiff = {maxdiff:e}");
    }
    ok
}

fn main() {
    // build the GPT-2 model from a checkpoint
    let mut model = GPT2::build_from_checkpoint("gpt2_124M.bin");

    let C = model.config.channels;
    let V = model.config.vocab_size;
    let Vp = model.config.padded_vocab_size;
    let maxT = model.config.max_seq_len;
    let L = model.config.num_layers;

    // load additional information that we will use for debugging and error checking
    let state_file = std::fs::File::open("gpt2_124M_debug_state.bin");
    let mut state_file = match state_file {
        Ok(f) => f,
        Err(_) => {
            println!("Error opening state file");
            std::process::exit(1);
        }
    };
    let state_header = read_i32s(&mut state_file, 256);
    if state_header[0] != 20240327 {
        println!("Bad magic state file");
        std::process::exit(1);
    }
    if state_header[1] != 2 {
        println!("Bad version in state file");
        println!("---> HINT: try to re-run `python train_gpt2.py`");
        std::process::exit(1);
    }
    let B = state_header[2] as usize; // batch size, e.g. 4
    let T = state_header[3] as usize; // time / sequence length (e.g. 64, up to maxT)
    println!("[State]");
    println!("batch_size: {B}");
    println!("seq_len: {T}");

    let (mut expected_grads_memory, expected_grads) =
        malloc_and_point_parameters(&model.param_sizes);

    // inputs and expected outputs, only used for error checking
    let x = read_i32s(&mut state_file, B * T);
    let y = read_i32s(&mut state_file, B * T);
    let expected_logits = read_f32s(&mut state_file, B * T * V);
    let expected_loss = read_f32s(&mut state_file, 1);
    // read reference information from Python
    let read_grads = read_f32s(&mut state_file, model.num_parameters);
    expected_grads_memory.copy_from_slice(&read_grads);

    // overall OK signal for the test
    let mut allok = true;

    // let's do 10 training iterations, following the pytorch code
    let expected_losses: [f32; 10] = [
        5.270007133483887,
        4.059706687927246,
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564195040283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615,
    ];
    for step in 0..10usize {
        let start = std::time::Instant::now();

        model.gpt2_forward(&x, Some(&y), B, T);
        model.gpt2_zero_grad();
        model.gpt2_backward();

        let time_elapsed_s = start.elapsed().as_secs_f64();

        if step == 0 {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            let mut logits_ok = true;
            let calculated_logits = model.acts.unwrap().logits.slice(
                model.acts_memory.as_ref().unwrap(),
                0,
                B * T * Vp,
            );
            let mut max_diff = 0.0f32;
            'outer: for bt in 0..B * T {
                for v in 0..V {
                    // note we only loop to V (ignoring padding)
                    let i = bt * Vp + v; // linearized index, using Vp
                    if i < 10 {
                        println!("{:.6}, {:.6}", expected_logits[i], calculated_logits[i]);
                    }
                    let diff = (expected_logits[bt * V + v] - calculated_logits[i]).abs();
                    max_diff = max_diff.max(diff);
                    if diff >= 1e-2f32 {
                        println!("MISMATCH AT INDEX {bt},{v}: ");
                        println!(
                            "{:.6} {:.6}",
                            expected_logits[bt * V + v],
                            calculated_logits[i]
                        );
                        logits_ok = false;
                        break 'outer; // to break out of both loops
                    }
                }
            }
            if !logits_ok {
                print!("NOT ");
            }
            println!("OK (LOGITS), max_diff = {max_diff:e}");
            allok = allok && logits_ok;

            // compare the achieved loss
            if (model.mean_loss - expected_loss[0]).abs() >= 1e-2 {
                println!(
                    "LOSS MISMATCH: {:.6} {:.6}",
                    model.mean_loss, expected_loss[0]
                );
                allok = false;
            } else {
                println!("LOSS OK: {:.6} {:.6}", model.mean_loss, expected_loss[0]);
            }

            // finally check all the gradients
            let g = model.grads.unwrap();
            let gm = &model.grads_memory.as_ref().unwrap();
            let checks: [(&[f32], &[f32], &str); 16] = [
                (
                    g.wte.slice(gm, 0, V * C),
                    expected_grads.wte.slice(&expected_grads_memory, 0, V * C),
                    "dwte",
                ),
                (
                    g.wpe.slice(gm, 0, maxT * C),
                    expected_grads
                        .wpe
                        .slice(&expected_grads_memory, 0, maxT * C),
                    "dwpe",
                ),
                (
                    g.ln1w.slice(gm, 0, L * C),
                    expected_grads.ln1w.slice(&expected_grads_memory, 0, L * C),
                    "dln1w",
                ),
                (
                    g.ln1b.slice(gm, 0, L * C),
                    expected_grads.ln1b.slice(&expected_grads_memory, 0, L * C),
                    "dln1b",
                ),
                (
                    g.qkvw.slice(gm, 0, L * 3 * C * C),
                    expected_grads
                        .qkvw
                        .slice(&expected_grads_memory, 0, L * 3 * C * C),
                    "dqkvw",
                ),
                (
                    g.qkvb.slice(gm, 0, L * 3 * C),
                    expected_grads
                        .qkvb
                        .slice(&expected_grads_memory, 0, L * 3 * C),
                    "dqkvb",
                ),
                (
                    g.attprojw.slice(gm, 0, L * C * C),
                    expected_grads
                        .attprojw
                        .slice(&expected_grads_memory, 0, L * C * C),
                    "dattprojw",
                ),
                (
                    g.attprojb.slice(gm, 0, L * C),
                    expected_grads
                        .attprojb
                        .slice(&expected_grads_memory, 0, L * C),
                    "dattprojb",
                ),
                (
                    g.ln2w.slice(gm, 0, L * C),
                    expected_grads.ln2w.slice(&expected_grads_memory, 0, L * C),
                    "dln2w",
                ),
                (
                    g.ln2b.slice(gm, 0, L * C),
                    expected_grads.ln2b.slice(&expected_grads_memory, 0, L * C),
                    "dln2b",
                ),
                (
                    g.fcw.slice(gm, 0, L * 4 * C * C),
                    expected_grads
                        .fcw
                        .slice(&expected_grads_memory, 0, L * 4 * C * C),
                    "dfcw",
                ),
                (
                    g.fcb.slice(gm, 0, L * 4 * C),
                    expected_grads
                        .fcb
                        .slice(&expected_grads_memory, 0, L * 4 * C),
                    "dfcb",
                ),
                (
                    g.fcprojw.slice(gm, 0, L * C * 4 * C),
                    expected_grads
                        .fcprojw
                        .slice(&expected_grads_memory, 0, L * C * 4 * C),
                    "dfcprojw",
                ),
                (
                    g.fcprojb.slice(gm, 0, L * C),
                    expected_grads
                        .fcprojb
                        .slice(&expected_grads_memory, 0, L * C),
                    "dfcprojb",
                ),
                (
                    g.lnfw.slice(gm, 0, C),
                    expected_grads.lnfw.slice(&expected_grads_memory, 0, C),
                    "dlnfw",
                ),
                (
                    g.lnfb.slice(gm, 0, C),
                    expected_grads.lnfb.slice(&expected_grads_memory, 0, C),
                    "dlnfb",
                ),
            ];
            for (a, b, label) in checks {
                let gradok = check_tensor(a, b, label);
                allok = allok && gradok;
            }
        }

        model.gpt2_update(
            1e-4f32,
            0.9f32,
            0.999f32,
            1e-8f32,
            0.01f32,
            (step + 1) as i32,
        );

        // compare the losses
        let expected_loss_step = expected_losses[step];
        let actual_loss = model.mean_loss;
        let step_loss_ok = (expected_loss_step - actual_loss).abs() < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        println!(
            "step {}: loss {:.6} (took {:.6} ms) OK = {}",
            step,
            model.mean_loss,
            time_elapsed_s * 1000.0,
            step_loss_ok as i32
        );
    }

    // final judgement
    println!("overall okay: {}", allok as i32);

    // (deviation from the C reference, which always returns 0: exit nonzero on failure,
    // so the test can be used in scripts and CI)
    std::process::exit(if allok { 0 } else { 1 });
}
