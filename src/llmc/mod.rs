// ports of the llm.c/llmc/ shared headers:
// defines: fopen_check, fread_check, fseek_check (utils.rs)
// defines: Tokenizer, tokenizer_init, tokenizer_decode, safe_printf (tokenizer.rs)
// defines: DataLoader, dataloader_init, dataloader_reset, dataloader_next_batch (dataloader.rs)
// defines: mt19937_state, manual_seed, randint32, random_permutation (rand.rs)
pub mod dataloader;
pub mod rand;
pub mod tokenizer;
pub mod utils;
