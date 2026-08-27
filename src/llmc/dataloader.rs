/*
Implements:
- DataLoader for model training. Reads and serves data shards.
(The EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag, is not
ported: the pure-CPU train_gpt2.c reference does not use it.)
Ported 1:1 from llm.c/llmc/dataloader.h.
*/

#![allow(non_snake_case)]

use std::{fs::File, path::PathBuf, process::exit};

use super::{
    rand::{Mt19937State, init_identity_permutation, manual_seed, random_permutation},
    utils::{fopen_check, fseek_check, read_i32s, read_u16s},
};

// ----------------------------------------------------------------------------
// Distributed Data Loader
pub const HEADER_SIZE: usize = 256;

pub struct DataLoader {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    pub process_rank: i32,
    pub num_processes: i32,
    // batch and token information
    pub B: usize,
    pub T: usize,
    pub num_tokens: usize,        // total number of tokens
    pub shard_num_samples: usize, // total number of samples in the current shard per process
    // shards and current position
    pub shard_paths: Vec<PathBuf>, // the glob results, i.e. all shards we want to iterate
    pub current_shard_idx: usize,  // the current shard we are reading from
    pub current_sample_idx: usize, // the current sample we are reading from
    // file handle
    pub tokens_file: Option<File>,
    // data buffers
    pub buffer: Vec<u16>,  // we fread data from file into this buffer
    pub inputs: Vec<i32>,  // input tokens into transformer
    pub targets: Vec<i32>, // target tokens for the transformer
    // random shuffle related variables
    pub shuffle_rng: Mt19937State,
    pub should_shuffle: bool,
    pub shard_indices: Vec<i32>,
    pub intra_shard_indices: Vec<i32>,
    // sizes in bytes
    pub total_batch_size_bytes: usize, // total across all processes
    pub local_batch_offset_bytes: usize, // inner-sample offset for this process
    pub header_bytes: usize,           // header size in bytes
    pub file_size_bytes: i64,
}

impl DataLoader {
    fn dataloader_load_shard_(&mut self, shard_index: usize) -> i64 {
        let shard_index = if self.should_shuffle {
            self.shard_indices[shard_index] as usize
        } else {
            shard_index
        };
        // use the first glob match as the filename for now
        let filename = self.shard_paths[shard_index].to_string_lossy().into_owned();
        // open the input file for reading. also only a single file can be opened at a time
        self.tokens_file = Some(fopen_check(&filename, "rb"));
        let tokens_file = self.tokens_file.as_mut().unwrap();
        // validate the header
        let header = read_i32s(tokens_file, HEADER_SIZE);
        if header[0] != 20240520 {
            println!("Bad magic in the data file");
            println!("---> HINT: Are you passing in a correct file?");
            println!(
                "---> HINT: The data encoding may have changed, re-run data prepro or refer again to README."
            );
            exit(1);
        }
        if header[1] != 1 {
            println!("Bad version in data file");
            exit(1);
        }
        let ntok = header[2] as i64; // number of tokens in the file
        assert!(ntok > 0); // we expect some tokens in the file. this should never trip, right?
        // determine the file size and make sure it is consistent with the number of tokens
        self.file_size_bytes = std::fs::metadata(&filename)
            .expect("could not stat data file")
            .len() as i64;
        // we expect ntok in the file to be consistent with filesize, assert that is the case
        let expected_file_size = (HEADER_SIZE * 4 + ntok as usize * 2) as i64;
        if self.file_size_bytes != expected_file_size {
            println!("Error: file size is not as expected");
            exit(1);
        }
        // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
        self.shard_num_samples = (ntok as usize * 2 - 2) / self.total_batch_size_bytes;
        ntok
    }

    fn prepare_intra_shard_indices_(&mut self) {
        // shuffle the examples inside the shards
        // (in C this is freed and re-malloc'ed in case shards have different number of samples / sizes)
        self.intra_shard_indices = vec![0i32; self.shard_num_samples];
        init_identity_permutation(&mut self.intra_shard_indices);
        random_permutation(&mut self.intra_shard_indices, &mut self.shuffle_rng);
    }

    pub fn reset(&mut self) {
        self.current_shard_idx = 0;
        self.current_sample_idx = 0;

        if self.should_shuffle {
            // shuffle the shards
            random_permutation(&mut self.shard_indices, &mut self.shuffle_rng);
        }

        self.dataloader_load_shard_(self.current_shard_idx);

        if self.should_shuffle {
            self.prepare_intra_shard_indices_();
        }
    }

    fn dataloader_advance_(&mut self) {
        if self.current_shard_idx == self.shard_paths.len() - 1 {
            // if we are at the last shard, we reset the loader and start a new epoch
            self.reset();
            return;
        }

        // advance the loader by loading the next data shard and resetting the position
        self.current_shard_idx = (self.current_shard_idx + 1) % self.shard_paths.len();
        self.current_sample_idx = 0;
        self.dataloader_load_shard_(self.current_shard_idx);

        if self.should_shuffle {
            self.prepare_intra_shard_indices_();
        }
    }

    /// port of dataloader_init()
    pub fn init(
        filename_pattern: &str,
        B: usize,
        T: usize,
        process_rank: i32,
        num_processes: i32,
        should_shuffle: i32,
    ) -> DataLoader {
        let mut loader = DataLoader {
            process_rank,
            num_processes,
            B,
            T,
            num_tokens: 0,
            shard_num_samples: 0,
            shard_paths: Vec::new(),
            current_shard_idx: 0,
            current_sample_idx: 0,
            tokens_file: None,
            buffer: Vec::new(),
            inputs: Vec::new(),
            targets: Vec::new(),
            shuffle_rng: Mt19937State::default(),
            should_shuffle: should_shuffle != 0,
            shard_indices: Vec::new(),
            intra_shard_indices: Vec::new(),
            total_batch_size_bytes: 0,
            local_batch_offset_bytes: 0,
            header_bytes: HEADER_SIZE * 4,
            file_size_bytes: 0,
        };
        loader.total_batch_size_bytes = (num_processes as usize * (B * T)) * 2;
        loader.local_batch_offset_bytes = process_rank as usize * B * T * 2;

        // glob to get the list of files matching the pattern, these are our data shards
        let mut glob_result: Vec<PathBuf> = Vec::new();
        match glob::glob(filename_pattern) {
            Ok(paths) => {
                for path in paths.flatten() {
                    glob_result.push(path);
                }
            }
            Err(_) => {
                println!("Error: failed to glob pattern: {filename_pattern}");
                exit(1);
            }
        }
        if glob_result.is_empty() {
            println!("Error: no files found matching the pattern: {filename_pattern}");
            exit(1);
        }
        loader.shard_paths = glob_result;

        if loader.should_shuffle {
            manual_seed(&mut loader.shuffle_rng, (42 + process_rank) as u32);
            loader.shard_indices = vec![0i32; loader.shard_paths.len()];
            init_identity_permutation(&mut loader.shard_indices);
            // intra_shard_indices dynamically (re)allocated, allowing different shard sizes
        }

        // inspect and validate all shards so we don't get any runtime errors later
        // if too slow / too many shards, may wish to revisit later
        let mut ntok_total: i64 = 0;
        for shard_index in 0..loader.shard_paths.len() {
            let shard_ntok = loader.dataloader_load_shard_(shard_index);
            // we need at least one batch/shard, the way things are written right now.
            // can be relaxed a lot later.
            assert!(shard_ntok >= (num_processes as usize * B * T + 1) as i64);
            ntok_total += shard_ntok;
        }

        // allocate all the space we'll need
        loader.buffer = vec![0u16; B * T + 1];
        loader.inputs = vec![0i32; B * T];
        loader.targets = vec![0i32; B * T];
        loader.num_tokens = ntok_total as usize;

        // reset the loader, to initialize it
        loader.reset();
        loader
    }

    fn dataloader_load_batch(&mut self) {
        assert!(!self.should_shuffle || !self.intra_shard_indices.is_empty());
        assert!(self.current_sample_idx < self.shard_num_samples);
        let idx = if self.should_shuffle {
            self.intra_shard_indices[self.current_sample_idx] as usize
        } else {
            self.current_sample_idx
        };
        let global_batch_offset_bytes = idx * self.total_batch_size_bytes;
        let current_offset =
            (self.header_bytes + global_batch_offset_bytes + self.local_batch_offset_bytes) as i64;

        let B = self.B;
        let T = self.T;
        // read B*T+1 uint16_t tokens from the file into buffer
        let tokens_file = self.tokens_file.as_mut().unwrap();
        fseek_check(tokens_file, std::io::SeekFrom::Start(current_offset as u64));
        self.buffer = read_u16s(tokens_file, B * T + 1);
        // decode the buffer into inputs and targets (cast to int)
        for i in 0..B * T {
            self.inputs[i] = self.buffer[i] as i32;
            self.targets[i] = self.buffer[i + 1] as i32;
        }
    }

    pub fn next_batch(&mut self) {
        // if the next batch would go past the end of the file, advance the loader
        if self.current_sample_idx >= self.shard_num_samples {
            self.dataloader_advance_();
        }
        self.dataloader_load_batch();
        self.current_sample_idx += 1;
    }

    /// port of dataloader_resume(), used during model resumption
    #[allow(dead_code)]
    pub fn resume(&mut self, current_shard_idx: usize, current_sample_idx: usize) {
        self.current_shard_idx = current_shard_idx;
        self.current_sample_idx = current_sample_idx;
        self.dataloader_load_shard_(self.current_shard_idx);
    }
}

// dataloader_free() has no Rust equivalent: everything is freed when dropped
