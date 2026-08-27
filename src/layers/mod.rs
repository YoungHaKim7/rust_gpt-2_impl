// all the individual layers' forward and backward passes, one file per layer:
// defines: encoder_forward, encoder_backward (encoder.rs)
// defines: layernorm_forward, layernorm_backward (layernorm.rs)
// defines: matmul_forward_naive, matmul_forward, matmul_backward (matmul.rs)
// defines: attention_forward, attention_backward (attention.rs)
// defines: GELU_SCALING_FACTOR, gelu_forward, gelu_backward (gelu.rs)
// defines: residual_forward, residual_backward (residual.rs)
// defines: softmax_forward, crossentropy_forward, crossentropy_softmax_backward (loss.rs)
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

mod attention;
mod encoder;
mod gelu;
mod layernorm;
mod loss;
mod matmul;
mod residual;

pub use self::attention::{attention_backward, attention_forward};
pub use self::encoder::{encoder_backward, encoder_forward};
pub use self::gelu::{GELU_SCALING_FACTOR, gelu_backward, gelu_forward};
pub use self::layernorm::{layernorm_backward, layernorm_forward};
pub use self::loss::{crossentropy_forward, crossentropy_softmax_backward, softmax_forward};
pub use self::matmul::{matmul_backward, matmul_forward, matmul_forward_naive};
pub use self::residual::{residual_backward, residual_forward};
