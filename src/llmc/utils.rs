/*
 This file contains utilities shared between the different training scripts.
 Ported from llm.c/llmc/utils.h. In particular, the C code defines a series of
 macros xxxCheck that call the corresponding C standard library function and
 check its return code. If an error was reported, the program prints some debug
 information and exits; the Rust port keeps the same messages (minus __FILE__/__LINE__,
 which have no direct equivalent) and the same exit(1) behavior.
*/

#![allow(non_snake_case)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::exit;

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking
// simple replace fopen, fread, fclose, fseek
// with fopen_check, fread_check, fseek_check

pub fn fopen_check(path: &str, mode: &str) -> File {
    let file = if mode.starts_with('r') {
        File::open(path)
    } else if mode.starts_with('w') {
        File::create(path)
    } else if mode.starts_with('a') {
        OpenOptions_append(path)
    } else {
        panic!("unsupported mode: {mode}");
    };
    match file {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Error: Failed to open file '{path}'");
            eprintln!("Error details:");
            eprintln!("  Path: {path}");
            eprintln!("  Mode: {mode}");
            eprintln!(
                "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README"
            );
            eprintln!("---> HINT 2: possibly try to re-run `python train_gpt2.py`");
            exit(1);
        }
    }
}

fn OpenOptions_append(path: &str) -> std::io::Result<File> {
    std::fs::OpenOptions::new().append(true).open(path)
}

pub fn fread_check(buf: &mut [u8], stream: &mut File) {
    match stream.read_exact(buf) {
        Ok(()) => {}
        Err(e) => {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                eprintln!("Error: Unexpected end of file");
            } else {
                eprintln!("Error: File read error");
            }
            eprintln!("Error details:");
            eprintln!("  {e}");
            eprintln!("  Expected bytes: {}", buf.len());
            exit(1);
        }
    }
}

pub fn fseek_check(fp: &mut File, whence: SeekFrom) {
    if fp.seek(whence).is_err() {
        eprintln!("Error: Failed to seek in file");
        exit(1);
    }
}

pub fn fwrite_check(buf: &[u8], stream: &mut File) {
    match stream.write_all(buf) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: File write error");
            eprintln!("Error details:");
            eprintln!("  {e}");
            eprintln!("  Expected bytes: {}", buf.len());
            exit(1);
        }
    }
}

// ----------------------------------------------------------------------------
// typed binary I/O helpers (the raw fread/fwrite of C structs of u32/i32/f32/u16)

pub fn read_bytes(stream: &mut File, n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    fread_check(&mut buf, stream);
    buf
}

pub fn read_u32s(stream: &mut File, n: usize) -> Vec<u32> {
    bytemuck::cast_slice(&read_bytes(stream, n * 4)).to_vec()
}

pub fn read_i32s(stream: &mut File, n: usize) -> Vec<i32> {
    bytemuck::cast_slice(&read_bytes(stream, n * 4)).to_vec()
}

pub fn read_f32s(stream: &mut File, n: usize) -> Vec<f32> {
    bytemuck::cast_slice(&read_bytes(stream, n * 4)).to_vec()
}

pub fn read_u16s(stream: &mut File, n: usize) -> Vec<u16> {
    bytemuck::cast_slice(&read_bytes(stream, n * 2)).to_vec()
}

pub fn write_f32s(stream: &mut File, vals: &[f32]) {
    fwrite_check(bytemuck::cast_slice(vals), stream);
}

pub fn write_u16s(stream: &mut File, vals: &[u16]) {
    fwrite_check(bytemuck::cast_slice(vals), stream);
}

pub fn write_u32s(stream: &mut File, vals: &[u32]) {
    fwrite_check(bytemuck::cast_slice(vals), stream);
}

pub fn write_i32s(stream: &mut File, vals: &[i32]) {
    fwrite_check(bytemuck::cast_slice(vals), stream);
}

/// write a header of `vals` followed by zeros up to 256 int32s, like the Python/C writers do
pub fn write_i32_header(stream: &mut File, vals: &[i32]) {
    let mut header = vec![0i32; 256];
    header[..vals.len()].copy_from_slice(vals);
    write_i32s(stream, &header);
}

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util
// (in Rust, allocation failure aborts by itself, so mallocCheck has no equivalent)

// ----------------------------------------------------------------------------
// check that all tokens are within range

pub fn token_check(tokens: &[i32], vocab_size: i32) {
    for (i, &tok) in tokens.iter().enumerate() {
        if !(0 <= tok && tok < vocab_size) {
            eprintln!("Error: Token out of vocabulary");
            eprintln!("Error details:");
            eprintln!("  Token: {tok}");
            eprintln!("  Position: {i}");
            eprintln!("  Vocab: {vocab_size}");
            exit(1);
        }
    }
}
