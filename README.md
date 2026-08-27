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
