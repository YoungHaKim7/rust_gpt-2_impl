/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add encoding.
Which could be tricky in C because of the regex involved, to look into later.
Ported 1:1 from llm.c/llmc/tokenizer.h.
*/

#![allow(non_snake_case)]

use std::io::Write;
use std::process::exit;

use super::utils::{fread_check, read_bytes, read_u32s};

// ----------------------------------------------------------------------------

pub struct Tokenizer {
    pub vocab_size: u32,
    pub token_table: Vec<Vec<u8>>,
    pub init_ok: bool,
    pub eot_token: i32, // <|endoftext|> token id
}

pub fn safe_printf(piece: &[u8]) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if piece.is_empty() {
        return;
    }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if piece.len() == 1 {
        let byte_val = piece[0];
        let printable = byte_val.is_ascii_graphic() || byte_val.is_ascii_whitespace();
        if !printable {
            return; // weird byte, don't print it
        }
    }
    // write the raw bytes out, like C's printf("%s", piece)
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(piece);
}

impl Tokenizer {
    pub fn init(filename: &str) -> Tokenizer {
        let mut tokenizer = Tokenizer {
            vocab_size: 0,
            token_table: Vec::new(),
            init_ok: false,
            eot_token: 0,
        };
        let file = std::fs::File::open(filename);
        let mut file = match file {
            Ok(f) => f,
            Err(_) => {
                // try to be more helpful as we just added this feature, erase later
                println!("---");
                println!("WARNING: Failed to open the tokenizer file {filename}");
                println!("The Tokenizer is a new feature added April 14 2024.");
                println!("Re-run `python train_gpt2.py` to write it");
                println!("---");
                return tokenizer; // init_ok = 0
            }
        };
        // read in the header
        let header = read_u32s(&mut file, 256);
        assert!(header[0] == 20240328, "bad magic in tokenizer file");
        let version = header[1];
        tokenizer.vocab_size = header[2];
        if version == 1 {
            // version 1 didn't include the EOT token id
            // so we assume it is 50256, the EOT in GPT-2
            assert!(tokenizer.vocab_size == 50257, "let's be defensive here");
            tokenizer.eot_token = 50256;
        } else if version == 2 {
            tokenizer.eot_token = header[3] as i32;
        } else {
            eprintln!("Tokenizer model file {filename} has bad version: {version}");
            exit(1);
        }
        // read in all the tokens
        let mut token_table = Vec::with_capacity(tokenizer.vocab_size as usize);
        for _ in 0..tokenizer.vocab_size {
            let mut len_buf = [0u8; 1];
            fread_check(&mut len_buf, &mut file);
            let length = len_buf[0] as usize;
            assert!(length > 0, "every token should be at least one character");
            let token_bytes = read_bytes(&mut file, length);
            token_table.push(token_bytes);
        }
        tokenizer.token_table = token_table;
        // cleanups (file closed on drop)
        tokenizer.init_ok = true;
        tokenizer
    }

    pub fn decode(&self, token_id: u32) -> Option<&[u8]> {
        if !self.init_ok {
            return None;
        }
        if token_id < self.vocab_size {
            Some(&self.token_table[token_id as usize])
        } else {
            println!("invalid token id {token_id}!");
            None
        }
    }
}

// tokenizer_free() has no Rust equivalent: the table is freed when dropped
