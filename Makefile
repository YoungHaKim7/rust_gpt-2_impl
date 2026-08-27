# Makefile for the Rust port of llm.c, mirroring the llm.c workflow.
# The binaries are named exactly like the C ones: train_gpt2, test_gpt2.

.PHONY: all train_gpt2 test_gpt2 gen_synth run test verify clean

all: train_gpt2 test_gpt2 gen_synth

train_gpt2:
	cargo build --release --bin train_gpt2

test_gpt2:
	cargo build --release --bin test_gpt2

gen_synth:
	cargo build --release --bin gen_synth

# convenience: run training on the data in the current directory
run: train_gpt2
	./target/release/train_gpt2

# unit tests (mt19937 vectors, matmul equivalence, finite-difference gradient check, ...)
test:
	cargo test --release

# the strongest check: build the original llm.c train_gpt2.c with gcc and compare
# this port against it step by step on identical synthetic data
verify:
	bash dev/verify_vs_c.sh

clean:
	cargo clean
