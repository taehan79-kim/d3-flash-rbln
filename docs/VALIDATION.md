# 검증 단계와 성공 기준

컴파일, 런타임, 수치, 성능을 서로 다른 gate로 판정한다. 앞 단계 통과가 다음 단계를
보장하지 않는다.

## Gate 0 — 환경

```bash
d3-rbln doctor
```

확인 사항:

- Python 3.12
- Transformers 5.17.0
- 대상 서버에 `rebel-compiler`, `torch-rbln`, 호환 PyTorch 존재
- target이 `RBLN-CA25`
- SDK 패키지 자동 업그레이드 없음

## Gate 1 — snapshot

성공 기준:

- model ID와 revision이 고정값과 일치
- 설정, 토크나이저, readout, weight index 존재
- index가 가리키는 모든 shard 존재
- 허용하지 않은 remote Python 파일 없음
- `snapshot.json`의 크기·SHA-256 일치

검사 실패 시 파일을 임의로 추가하거나 `main` revision으로 바꾸지 않는다.

## Gate 2 — CPU graph parity

`compile`은 실제 RBLN 컴파일 전에 같은 weights를 공유하는 두 CPU 경로의 logits를 비교한다.

```text
Transformers Qwen3.5 + explicit d3 masks
                    vs
TextDecision tensor-only graph
```

검사 대상:

- 출력 shape `[1,255]`
- NaN/Inf 없음
- max/mean absolute logit error
- 선택 option 및 option probability
- wheel에 포함된 기본 smoke examples 또는 명시한 `--data`의 모든 요청

이 gate는 wrapper 동등성 검사다. 전체 9B upstream `system_one` 서버와 독립 비교한 결과로
해석하지 않는다.

## Gate 3 — strict export 및 2-device compile

```bash
d3-rbln compile \
  --model-dir models/d3-flash \
  --bucket 256 \
  --output results/compile-256
```

성공 기준:

- strict export 성공
- RBLN 컴파일 성공
- 저장 후 `RBLNCompiledModel.inspect(path)`가 요청과 모순되는 장치 수를 보고하지 않음
- `get_alloc_per_node()`가 양수인 두 개의 할당값 반환
- `model.rbln`과 SHA-256 생성
- 결과 디렉터리에 `compile.json` 및 런타임용 샘플 기준 저장

`host_ops_detected=true`면 아티팩트 생성은 성공할 수 있지만 전체 NPU 가속 성공으로 판정하지
않는다. 결과가 불충분하면 `unknown`으로 유지한다.

실패하면 `failure.json`과 compiler 원문을 보존한다. 다른 backend나 eager로 자동 재시도하지
않는다.

## Gate 4 — 선택한 두 장치에서 runtime

`rbln-smi`로 비어 있는 두 논리 NPU를 선택한다. 기존 프로세스가 사용하는 장치는 선택하지
않는다.

```bash
RBLN_VISIBLE_DEVICES=ID_A,ID_B \
  d3-rbln validate \
  --artifact-dir results/compile-256 \
  --runtime-device LOGICAL_RUNTIME_ID
```

성공 기준:

- visible ID가 정확히 두 개
- operator가 서버 토폴로지에서 runtime ID 매핑 확인
- runtime 생성 및 warm-up 성공
- 실행 중 `rbln-smi`에서 의도한 두 장치의 context 확인
- 다른 서비스 장치의 할당 변화 없음

런타임은 `rebel.Runtime(artifact_path, device=<int>, tensor_type="pt")`로 연다. `device`에는
정수 하나만 전달하며 물리 ID 목록을 Python API에 넘기지 않는다.

## Gate 5 — NPU 수치 일치

각 샘플마다 다음을 기록한다.

- CPU 기준 logits
- 마지막 NPU logits
- warm-up과 timed repetition 각각의 수치 검사
- max/mean absolute error
- CPU/NPU 선택 option
- option probability 차이
- NaN/Inf 여부
- artifact/input SHA-256
- 컴파일 SDK 버전과 런타임 환경
- 컴파일 전에 고정한 수치 tolerance

최소 성공 기준:

1. 모든 샘플에 유한한 `[1,255]` logits
2. CPU와 NPU가 같은 option 선택
3. 정한 수치 tolerance 통과
4. 결과 JSON에 tolerance와 SDK·artifact hash 기록

tolerance는 첫 실행 전에 명시한다. 결과를 본 후 통과하도록 넓히지 않는다. BF16/FP32 혼합
경로이므로 exact bit equality를 기본 가정하지 않는다.

## Gate 6 — 지연 및 반복 안정성

warm-up과 timed run을 분리한다.

- 첫 실행·컴파일 시간과 steady-state 시간 분리
- 동일 입력으로 반복
- median 및 p50, nearest-rank p95, 평균, 최소·최대 기록
- CPU 전처리 및 후처리를 포함한 end-to-end 시간과 runtime 시간 구분
- 장치별 메모리와 host op evidence 보존

짧은 샘플의 latency만으로 동시 처리량이나 운영 SLA를 주장하지 않는다. API 서버·queue·부하
테스트는 이 프로젝트의 현재 범위 밖이다.

## 판정표

| 상태 | 의미 |
| --- | --- |
| `compile_failed` | strict graph 또는 compiler 실패 |
| `compiled_unverified` | 아티팩트는 있으나 NPU 실행 전 |
| `runtime_failed` | 대상 장치에서 runtime 생성/실행 실패 |
| `numerical_failed` | 실행됐지만 CPU 기준과 수치·결정 불일치 |
| `runtime_numerically_validated` | 지정 runtime에서 저장된 샘플의 실행·수치 gate 통과 |

CLI는 항상 `physical_acceleration_verified=false`를 기록한다. 런타임·수치 gate 통과만으로
선택한 물리 장치에서 모든 연산이 가속됐다고 자동 판정하지 않는다. operator가 `rbln-smi`
context와 compiler의 host-op evidence를 별도로 확인해야 한다. 이 결과는 다른 SDK, 다른
ATOM MAX 토폴로지, 다른 bucket, 이미지·영상, 실제 업무 데이터에 자동으로 일반화하지 않는다.
