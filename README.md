# rust_gpt-2_impl
- 러스트언어로 바꾸기 | llm.c를 사용하여 GPT-2 모델을 저렴하게 재구현하기 
  - https://github.com/karpathy/llm.c
  - https://github.com/GaoYusong/llm.cpp
  - https://github.com/ToJen/llm.rs

- `enum` & `struct` version
  - https://github.com/YoungHaKim7/rust_gpt-2_struct_enum_version
  
## Rust port (this repo's implementation): `gpt2/`

`llm.c/`의 순수 CPU 참조 구현(`train_gpt2.c` + `llmc/*.h`)을 Rust로 1:1 포팅한 크레이트.
함수 이름, 루프 구조, 코멘트, 바이너리 포맷, 출력까지 원본과 동일하게 유지.

A faithful 1:1 Rust port of the original llm.c pure-CPU reference, living in [`gpt2/`](gpt2/):

```bash
git clone https://github.com/YoungHaKim7/rust_gpt-2_impl.git
cd rust_gpt-2_impl
git submodule update --init
make            # builds train_gpt2, test_gpt2, gen_synth (binaries named like the C ones)
make test       # unit tests (mt19937 vs torch, matmul equivalence, finite-difference gradient check)
make verify     # compiles the original llm.c with gcc and compares both step by step
```

`make verify` result: the Rust port reproduces the C loss trajectory **bit-for-bit**
(46/46 loss values identical, generation output byte-identical), and runs ~5x faster
than the single-threaded C build thanks to rayon. See [gpt2/README.md](benches/README.md).

