/*
Mersenne Twister implementation, numerically identical to torch.
Ported 1:1 from llm.c/llmc/rand.h.

Example usage:

    let mut state = Mt19937State::default();
    manual_seed(&mut state, 137);
    println!("{}", randint32(&mut state));
    println!("{}", randint32(&mut state));
    println!("{}", randint32(&mut state));
    println!("{}", randint32(&mut state));
    println!("{}", randint32(&mut state));

PyTorch reference (producing identical results):

    import torch
    torch.manual_seed(137)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    ...

Both output:

    4053805790
    2173880614
    380293709
    1237255315
    2986595568
*/

#![allow(non_snake_case)]

const MERSENNE_STATE_M: usize = 397;
const MERSENNE_STATE_N: usize = 624;

const LMASK: u32 = 0x7fffffff;
const UMASK: u32 = 0x80000000;

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

#[derive(Clone)]
pub struct Mt19937State {
    // (the C struct also carries a seed_ field, which it never reads)
    left_: i32,
    next_: usize,
    state_: [u32; MERSENNE_STATE_N],
    MATRIX_A: [u32; 2],
}

impl Default for Mt19937State {
    fn default() -> Self {
        Mt19937State {
            left_: 1,
            next_: 0,
            state_: [0; MERSENNE_STATE_N],
            MATRIX_A: [0, 0],
        }
    }
}

pub fn manual_seed(state: &mut Mt19937State, seed: u32) {
    state.MATRIX_A[0] = 0x0;
    state.MATRIX_A[1] = 0x9908b0df;
    state.state_[0] = seed;
    for j in 1..MERSENNE_STATE_N {
        // C wraps at 32 bits; u32 arithmetic does the same
        state.state_[j] = 1812433253u32
            .wrapping_mul(state.state_[j - 1] ^ (state.state_[j - 1] >> 30))
            .wrapping_add(j as u32);
    }
    state.left_ = 1;
    state.next_ = 0;
}

fn next_state(state: &mut Mt19937State) {
    state.left_ = MERSENNE_STATE_N as i32;
    state.next_ = 0;
    let mut y: u32;
    let mut j: usize = 0;
    while j < MERSENNE_STATE_N - MERSENNE_STATE_M {
        y = (state.state_[j] & UMASK) | (state.state_[j + 1] & LMASK);
        state.state_[j] = state.state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state.MATRIX_A[(y & 0x1) as usize];
        j += 1;
    }
    while j < MERSENNE_STATE_N - 1 {
        y = (state.state_[j] & UMASK) | (state.state_[j + 1] & LMASK);
        // in C this is state_[j + (M - N)]; j >= N - M always holds here, so write it as j - (N - M)
        state.state_[j] =
            state.state_[j - (MERSENNE_STATE_N - MERSENNE_STATE_M)] ^ (y >> 1) ^ state.MATRIX_A[(y & 0x1) as usize];
        j += 1;
    }
    y = (state.state_[MERSENNE_STATE_N - 1] & UMASK) | (state.state_[0] & LMASK);
    state.state_[MERSENNE_STATE_N - 1] =
        state.state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state.MATRIX_A[(y & 0x1) as usize];
}

pub fn randint32(state: &mut Mt19937State) -> u32 {
    if state.MATRIX_A[0] != 0 || state.MATRIX_A[1] != 0x9908b0df {
        manual_seed(state, 5489); // auto-initialize
    }
    state.left_ -= 1;
    if state.left_ <= 0 {
        next_state(state);
    }
    let mut y: u32 = state.state_[state.next_];
    state.next_ += 1;
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    y
}

pub fn randint64(state: &mut Mt19937State) -> u64 {
    ((randint32(state) as u64) << 32) | randint32(state) as u64
}

pub fn randfloat32(state: &mut Mt19937State) -> f32 {
    (randint32(state) & ((1u32 << 24) - 1)) as f32 * (1.0f32 / (1u32 << 24) as f32)
}

pub fn randfloat64(state: &mut Mt19937State) -> f64 {
    (randint64(state) & ((1u64 << 53) - 1)) as f64 * (1.0f64 / (1u64 << 53) as f64)
}

pub fn uniform_(data: &mut [f32], from: f32, to: f32, state: &mut Mt19937State) {
    for t in 0..data.len() {
        data[t] = randfloat32(state) * (to - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
fn normal_fill_16(data: &mut [f32], mean: f32, std: f32) {
    const EPSILONE: f32 = 1e-12;
    for t in 0..8 {
        let u1 = 1.0 - data[t];
        let u2 = data[t + 8];
        let radius = (-2.0 * (u1 + EPSILONE).ln()).sqrt();
        let theta = (2.0 * std::f64::consts::PI * u2 as f64) as f32;
        data[t] = radius * theta.cos() * std + mean;
        data[t + 8] = radius * theta.sin() * std + mean;
    }
}

fn normal_fill(data: &mut [f32], mean: f32, std: f32, state: &mut Mt19937State) {
    let numel = data.len();
    for t in 0..numel {
        data[t] = randfloat32(state);
    }
    let mut i = 0;
    while i + 16 <= numel {
        normal_fill_16(&mut data[i..i + 16], mean, std);
        i += 16;
    }
    #[allow(clippy::manual_is_multiple_of)] // verbatim C condition
    if numel % 16 != 0 {
        // recompute the last 16 values
        let data = &mut data[numel - 16..];
        for i in 0..16 {
            data[i] = randfloat32(state);
        }
        normal_fill_16(data, mean, std);
    }
}

pub fn normal_(data: &mut [f32], mean: f32, std: f32, state: &mut Mt19937State) {
    const EPSILONE: f32 = 1e-12;
    let numel = data.len();
    if numel >= 16 {
        normal_fill(data, mean, std, state);
    } else {
        let mut next_double_normal_sample = 0.0f64;
        let mut has_next_double_normal_sample = false;
        for t in 0..numel {
            if has_next_double_normal_sample {
                data[t] = (next_double_normal_sample as f32) * std + mean;
                has_next_double_normal_sample = false;
                continue;
            }
            // for numel < 16 we draw a double (float64)
            let u1 = randfloat64(state) as f32;
            let u2 = randfloat64(state) as f32;
            let radius = (-2.0 * (1.0 - u2 + EPSILONE).ln()).sqrt();
            let theta = (2.0 * std::f64::consts::PI * u1 as f64) as f32;
            next_double_normal_sample = (radius * theta.sin()) as f64;
            has_next_double_normal_sample = true;
            data[t] = radius * theta.cos() * std + mean;
        }
    }
}

pub fn init_identity_permutation(data: &mut [i32]) {
    for (i, d) in data.iter_mut().enumerate() {
        *d = i as i32;
    }
}

pub fn random_permutation(data: &mut [i32], state: &mut Mt19937State) {
    let numel = data.len();
    for i in (1..numel).rev() {
        // pick an index j in [0, i] with equal probability
        let j = (randint32(state) as usize) % (i + 1);
        // swap i <-> j
        data.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_torch_reference() {
        // the exact sequence documented at the top of llm.c/llmc/rand.h,
        // verified identical to `torch.manual_seed(137); torch.randint(0, 0xFFFFFFFF, [1])`
        let mut state = Mt19937State::default();
        manual_seed(&mut state, 137);
        let expected = [4053805790u32, 2173880614, 380293709, 1237255315, 2986595568];
        for &e in &expected {
            assert_eq!(randint32(&mut state), e);
        }
    }

    #[test]
    fn random_permutation_is_deterministic_and_bijective() {
        let mut state = Mt19937State::default();
        manual_seed(&mut state, 42);
        let mut data = vec![0i32; 100];
        init_identity_permutation(&mut data);
        random_permutation(&mut data, &mut state);
        let mut sorted = data.clone();
        sorted.sort();
        for (i, &s) in sorted.iter().enumerate() {
            assert_eq!(s, i as i32);
        }
        // determinism: same seed, same permutation
        let mut state2 = Mt19937State::default();
        manual_seed(&mut state2, 42);
        let mut data2 = vec![0i32; 100];
        init_identity_permutation(&mut data2);
        random_permutation(&mut data2, &mut state2);
        assert_eq!(data, data2);
    }
}
