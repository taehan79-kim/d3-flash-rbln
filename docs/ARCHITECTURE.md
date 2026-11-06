# 아키텍처와 컴파일 경계

## 목표

한 `d3-flash` 텍스트 Decision 모델을 ATOM MAX 논리 NPU 2개에 분할 컴파일하고, 원본 계산과
결정 결과가 유지되는지 검증한다.

```text
JSONL request
  │
  ├─ CPU: prompt rendering, tokenization, fixed-size left padding
  │
  ▼
TextDecision(input_ids [1,L], attention_mask [1,L])
  ├─ BF16 Qwen3.5 text backbone
  │    ├─ causal Gated DeltaNet layers
  │    └─ noncausal full-attention layers
  ├─ final-token pooling
  └─ FP32 [255, hidden] decision readout
  │
  ├─ RBLN compiler: one graph, num_devices=2
  │
  ▼
logits [1,255]
  │
  └─ CPU: option masking, temperature, softmax, typed answer
```

## 일반 Qwen 서빙 경로를 재사용하지 않는 이유

`d3-flash`는 일반 텍스트 생성 모델이 아니다.

- full-attention layer는 prompt의 모든 non-padding key를 본다.
- Gated DeltaNet layer는 causal recurrence를 유지한다.
- 마지막 hidden state를 FP32 255-way readout으로 점수화한다.
- 각 질문은 별도 forward이며 텍스트를 생성하지 않는다.

일반 causal `Qwen3.5ForConditionalGeneration`이나 KV-cache 생성 경로로 바꾸면 같은 모델
계산이라고 볼 수 없다. 이 프로젝트는 `Qwen3_5Model`의 text weights를 로드하되 위 Decision
계약을 tensor-only module로 명시한다.

## 고정 shape

첫 포트는 다음 shape만 지원한다.

| 입력 | shape | dtype |
| --- | --- | --- |
| `input_ids` | `[1, bucket]` | `int64` |
| `attention_mask` | `[1, bucket]` | `int64` |
| 출력 logits | `[1, 255]` | `float32` |

입력은 왼쪽 패딩하며 bucket을 넘으면 오류로 중단한다. 자동 truncate는 정확도와 속도 비교를
왜곡하므로 사용하지 않는다. 여러 bucket은 별도 결과 디렉터리에서 각각 컴파일한다.

## 두 장치 분할

컴파일 호출은 다음 의미를 가진다.

```python
rebel.compile_from_torch(
    model,
    example_inputs=[input_ids, attention_mask],
    npu="RBLN-CA25",
    num_devices=2,
    model_trace_method="export_strict",
)
```

- `num_devices=2`: 컴파일러에 논리 NPU 두 개로 모델을 분할하도록 요청한다.
- `export_strict`: graph capture 실패를 숨기지 않는다.
- 구체적인 tensor/pipeline 분할 전략은 컴파일러가 결정한다.
- 아티팩트를 저장한 뒤 `RBLNCompiledModel.inspect(path)` classmethod로 검사한다.
- 컴파일 리포트의 `num_devices` 및 compile 객체의 `get_alloc_per_node()` 결과가 장치 두 개를 가리켜야
  성공으로 기록한다.

`tensor parallel`이라는 표현은 컴파일러 검사 결과가 이를 명시할 때만 사용한다.

## 컴파일과 실행 분리

컴파일 산출물에는 다음을 기록한다.

- target NPU와 요청한 장치 수
- SDK 패키지 버전
- compiler `inspect()` 결과
- 장치별 할당 바이트
- host op 존재 여부
- `model.rbln` SHA-256
- `runtime_validated=false`, `numerical_validated=false`

런타임은 별도 단계다. 저장된 파일을
`rebel.Runtime(artifact_path, device=<int>, tensor_type="pt")`로 직접 연다. 물리 장치 두 개는
프로세스 시작 전 `RBLN_VISIBLE_DEVICES`로 제한한다. visible ID와 runtime ID의 매핑은
서버에서 검증해야 하며 이 프로젝트는 이를 추정하지 않는다. compile 객체를 다시
`RBLNCompiledModel(...)`로 구성하거나 `device=[ID_A, ID_B]` 목록을 넘기지 않는다.

## CPU 대조

컴파일 전 대조는 같은 CPU weights를 공유하는 두 실행 경로를 비교한다.

1. Transformers Qwen3.5 text model에 d3 mask mapping을 직접 전달하는 기준 경로
2. RBLN에 넘길 `TextDecision` graph

이 비교는 wrapper의 계산 동등성을 확인하지만, upstream 전체 `system_one` 서버와 9B
체크포인트의 독립적인 end-to-end 검증을 대체하지 않는다. NPU 검증은 저장된 CPU logits와
별도로 수행한다.

## 의도적인 제한

- text only
- choice/noul only
- batch 1
- one fixed bucket per artifact
- no KV cache or text generation
- no image/video tower
- no FLA, Triton, CUDA 전용 kernel
- no hidden eager/CPU fallback
- no API server or load balancing

제한을 넓힐 때는 각 입력 모드마다 CPU 기준과 NPU 수치 gate를 별도로 추가해야 한다.

## 실패를 결과로 보존

컴파일 또는 검사 실패 시 `failure.json`을 남기고 오류를 다시 발생시킨다. 실패를 eager
성공으로 바꾸거나 기존 결과를 덮어쓰지 않는다. 컴파일 성공도 런타임·수치·성능 성공으로
승격하지 않는다.

## 보안·재현성 경계

- model revision은 commit SHA로 고정한다.
- snapshot 파일별 SHA-256을 확인한다.
- snapshot의 Python remote code는 다운로드하거나 실행하지 않는다.
- 토큰, `.env`, SDK 자격정보, 모델 가중치, 컴파일 산출물은 Git에 넣지 않는다.
- SDK와 PyTorch 버전을 자동 변경하지 않는다.
