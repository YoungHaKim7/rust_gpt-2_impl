/*
This library trains the GPT-2 model. It is a faithful Rust port of the clean,
minimal, pure-CPU reference in llm.c/train_gpt2.c:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use rayon parallel iterators wherever the C code has OpenMP pragmas,
  as this is a large speedup at very low cost of code complexity.
Where the C code carves all tensors out of one big allocation with raw pointers,
this port keeps the exact same layout but tracks tensors as (start, len) views
into a single Vec, materialized as slices with `split_disjoint` at use sites.
There will be other versions of this code that specialize it and make it fast.

The crate is divided by function, mirroring llmc/:
- tensor.rs: the (start, len) TensorView + split_disjoint memory machinery
- layers/:  the individual layers' forward and backward passes, one file per layer
- model.rs: the GPT-2 model definition (config, tensor layouts, allocation)
- gpt2.rs:  the GPT2 model struct and its forward/backward/update passes
- llmc/:    ports of the llm.c/llmc/ shared headers (tokenizer, dataloader, ...)
Everything is re-exported at the crate root below, so the public API is flat.
*/

#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod llmc;

mod gpt2;
mod layers;
mod model;
mod tensor;

pub use crate::gpt2::GPT2;
pub use crate::layers::*;
pub use crate::model::{
    ActivationTensors, GPT2Config, NUM_ACTIVATION_TENSORS, NUM_PARAMETER_TENSORS, ParameterTensors,
    fill_in_activation_sizes, fill_in_parameter_sizes, malloc_and_point_activations,
    malloc_and_point_parameters,
};
pub use crate::tensor::TensorView;
