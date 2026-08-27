# rust_gpt-2_impl
러스트언어로 바꾸기 | llm.c를 사용하여 GPT-2 모델을 저렴하게 재구현하기  (github.com/karpathy)

# ▲llm.c를 사용하여 GPT-2 모델을 저렴하게 재구현하기  (github.com/karpathy)
12P by GN⁺ 2024-05-29 | ★ favorite | 댓글과 토론
llm.c를 사용하여 GPT-2 (124M) 모델을 90분 안에 $20로 재현하는 방법 설명
GPT-2 (124M)은 OpenAI가 2019년에 발표한 가장 작은 모델
Lambda에서 8X A100 80GB SXM 노드를 사용하면 시간당 약 $14, 총 비용은 약 $20
단일 GPU로도 훈련 가능하지만 시간이 더 오래 걸림 (4-
- https://news.hada.io/topic?id=15065
- LLM training in simple, raw C/CUDA
  - https://github.com/karpathy/llm.c

# RustGPT: Rust로 처음부터 완전히 구현한 순수 트랜스포머 LLM (github.com/tekaratzas)
23P by GN⁺ 25-10-01
- https://news.hada.io/topic?id=23106
- RustGPT는 외부 머신러닝 프레임워크 없이, 순수 Rust와 ndarray만으로 구현된 트랜스포머 기반 언어 모델
- 사전 학습(Pre-training) 과 지시 튜닝(Instruction tuning) 을 통해 사실 기반 지식과 대화형 패턴을 학습하도록 설계됨
- 구조는 토크나이저 → 임베딩 → 트랜스포머 블록 → 출력 프로젝션으로 이어지는 전형적인 LLM 아키텍처를 따름
- 모듈화된 소스 구조와 테스트 코드를 제공하여 학습, 추론, 최적화 과정을 세부적으로 이해할 수 있음
- 러스트 생태계에서 프레임워크 의존 없이 LLM을 처음부터 구현해보고 싶은 개발자나 학습자에게 중요한 참고 자료

# ▲LLaMA-rs - Rust로 구현한 LLaMA (github.com/setzer22)
10P by xguru 2023-03-17 | ★ favorite | 댓글 1개
- llama.cpp 를 Rust로 포팅한 프로젝트
- f16 또는 4-bit quntized 버전 모델 지원
- 원본과 같이 ggml 텐서 라이브러리를 그대로 이용해서 오리지널과 같은 퍼포먼스
- https://news.hada.io/topic?id=8727

