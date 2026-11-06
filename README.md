# d3-flash RBLN 2-device compile harness

[`vllm-sr/d3-flash`](https://huggingface.co/vllm-sr/d3-flash)의 **텍스트 Decision 경로**를
ATOM MAX(`RBLN-CA25`) 논리 NPU 2개에 분할 컴파일하고 검증하기 위한 실험용 프로젝트다.

현재 목표는 **컴파일 가능성 및 수치 일치 검증**이다. API 서버나 운영 배포는 포함하지
않는다. 이 저장소의 존재, 로컬 테스트 통과, 컴파일 아티팩트 생성만으로 NPU 실행 성공이나
성능 향상을 주장하지 않는다.

## 범위

| 포함 | 제외 |
| --- | --- |
| 텍스트 `choice`·`noul` 결정 | 이미지·영상 입력 |
| batch 1, 고정 길이, 왼쪽 패딩 | 동적 길이·자동 잘라내기 |
| BF16 Qwen3.5 본체 + FP32 255-way readout | 텍스트 생성·KV cache |
| 비인과 full attention + causal Gated DeltaNet | 일반 causal Qwen 생성 그래프로 대체 |
| RBLN 2-device strict compile | CPU/eager 자동 폴백 |
| CPU 대조, 런타임 수치·지연 검증 | 운영 처리량·업무 정확도 보장 |

컴파일러가 두 장치 사이의 실제 분할 전략을 결정한다. `num_devices=2`는 장치 두 개 사용을
요청하지만 특정 tensor-parallel 전략을 보장하지 않는다.

## 고정 버전

- 모델: `vllm-sr/d3-flash`
- 모델 revision: `67e03e1bc157961c582ee7c6a35d9b15fada3dec`
- Transformers: `5.17.0`
- Python: `3.12`
- 컴파일 대상: `RBLN-CA25`, 논리 장치 2개

모델 다운로드는 약 16.8 GB다. `compile`은 모델을 자동 다운로드하지 않는다.

## 1. 로컬 준비 및 테스트

RBLN SDK가 없는 개발 머신에서도 CPU 테스트와 입력 준비 코드를 검증할 수 있다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.11.0'
python -m pip install -e '.[test]'
pytest -q
ruff check .
ruff format --check .
d3-rbln doctor
```

`doctor`는 환경을 점검할 뿐 SDK나 드라이버를 설치·업그레이드하지 않는다.
`torch`는 프로젝트 의존성에 넣지 않았다. NPU 서버의 vendor PyTorch를 일반 PyPI 패키지가
교체하지 않도록 한 의도적인 선택이다.

## 2. 연결된 머신에서 고정 snapshot 받기

```bash
d3-rbln fetch --output models/d3-flash
```

`fetch`는 고정 revision의 설정·토크나이저·가중치만 내려받고 파일별 SHA-256을
`models/d3-flash/snapshot.json`에 기록한다. Hugging Face 저장소의 Python 코드는 내려받거나
실행하지 않는다.

NPU 서버가 오프라인이면 [오프라인 반입 안내](docs/OFFLINE.md)를 따른다.

## 3. NPU 서버의 기존 SDK 환경 보존

드라이버와 RBLN SDK가 이미 설치된 **전용 Python 3.12 환경**을 사용한다. SDK의 PyTorch를
일반 PyPI 패키지로 교체하지 않는다.

```bash
source /path/to/vendor-sdk-venv/bin/activate
python -m pip install --no-index --find-links wheelhouse \
  --no-deps -r requirements-app-lock.txt
python -m pip install --no-index --find-links wheelhouse \
  --no-deps wheelhouse/d3_flash_rbln-*.whl
python -m pip check
d3-rbln doctor
```

`requirements-app-lock.txt`는 `torch`와 vendor SDK를 제외한 애플리케이션 의존성 전체를
고정한다. `--no-deps`를 제거하면 resolver가 일반 PyPI `torch`를 가져와 vendor PyTorch를
교체할 수 있으므로 제거하지 않는다. 설치 충돌이 나면 제약을 삭제해 우회하지 말고 환경을
분리한다. 자세한 절차는
[오프라인 반입 안내](docs/OFFLINE.md)를 참고한다.

## 4. 컴파일

컴파일 단계는 NPU를 할당하거나 런타임을 실행하지 않는다.
9B BF16 가중치 외에도 CPU 로딩·대조 실행과 compiler workspace가 필요하다. 실행 전에
호스트 RAM과 디스크 여유를 확인하고, 모델 파일 크기만으로 필요한 작업 메모리를 추정하지
않는다.

```bash
d3-rbln compile \
  --model-dir models/d3-flash \
  --bucket 256 \
  --output results/compile-256
```

`--data`를 생략하면 wheel에 포함된 smoke examples를 사용한다. 같은 명령의 편의 wrapper도
제공한다.

```bash
./run_compile.sh models/d3-flash results/compile-256 256
```

순서:

1. snapshot revision·해시 확인
2. 입력을 고정 길이로 준비하고 초과 입력 거부
3. CPU에서 원본 Qwen3.5 실행 경로와 tensor wrapper의 logits 비교
4. `export_strict`로 graph trace
5. `rebel.compile_from_torch(..., npu="RBLN-CA25", num_devices=2)` 실행
6. 아티팩트를 저장한 뒤 `RBLNCompiledModel.inspect(path)`로 검사
7. 컴파일 검사 결과·장치별 메모리 할당·아티팩트 해시 기록

성공하면 `compile.json`과 `model.rbln`이 생성된다. 실패하면 `failure.json`을 남기고 비정상
종료한다. 기존 결과 디렉터리를 덮어쓰지 않는다.

## 5. 선택한 두 장치에서 검증

먼저 `rbln-smi`로 비어 있는 논리 NPU 두 개와 토폴로지를 확인한다. 아래의 `ID_A,ID_B`를
직접 확인한 ID로 바꾼다.

```bash
RBLN_VISIBLE_DEVICES=ID_A,ID_B \
  d3-rbln validate \
  --artifact-dir results/compile-256 \
  --runtime-device LOGICAL_RUNTIME_ID
```

동일한 검증을 편의 wrapper로 실행할 수 있다.

```bash
./run_validate.sh results/compile-256 ID_A,ID_B LOGICAL_RUNTIME_ID
```

중요:

- `RBLN_VISIBLE_DEVICES`는 **프로세스를 시작하기 전에** 정확히 두 ID로 지정한다.
- `--runtime-device`는 정수 하나다. `[ID_A,ID_B]` 같은 목록을 넘기지 않는다.
- visible ID와 runtime ID의 대응은 SDK/호스트 토폴로지에 의존한다. 임의로 `0`을 선택하지
  말고 해당 서버에서 확인한다.
- 기존 모델이 사용하는 장치를 선점하거나 기존 서비스를 내리지 않는다.

검증은 컴파일 때 저장한 샘플로 NPU 출력을 CPU 기준과 비교하고 지연 시간을 기록한다. 모델
가중치 전체를 다시 로드하지 않는다. 성공 기준은 [검증 기준](docs/VALIDATION.md)을 따른다.
런타임은 저장된 아티팩트를 `rebel.Runtime(path, device=..., tensor_type="pt")`로 직접 연다.

## 결과 해석

| 결과 | 의미 |
| --- | --- |
| `compile.json` 생성 | 2-device 아티팩트를 생성하고 컴파일러 메타데이터를 확인함 |
| `host_ops_detected=true` | host op 증거가 있으므로 전체 NPU 가속을 주장할 수 없음 |
| runtime 실행 성공 | 선택한 장치에서 아티팩트가 실행됨 |
| 수치 gate 통과 | 저장된 CPU 기준과 허용 오차 안에서 일치함 |
| 짧은 지연 시간 | 해당 샘플·설정의 관측값일 뿐 운영 성능 보장은 아님 |

컴파일만 성공한 상태에서는 `runtime_validated=false`, `numerical_validated=false`가 정상이다.
검증 JSON의 `physical_acceleration_verified=false`도 의도된 값이다. CLI가 선택 장치의 실제
context와 연산 배치를 자동 증명하지 않으므로 operator가 `rbln-smi`와 compiler evidence를
별도로 확인해야 한다.

## 문서

- [아키텍처와 경계](docs/ARCHITECTURE.md)
- [오프라인 반입](docs/OFFLINE.md)
- [검증 및 성공 기준](docs/VALIDATION.md)

## 근거

- [d3-flash 모델과 실행 계약](https://huggingface.co/vllm-sr/d3-flash)
- [공개 GLiNER 포팅 실험 참고](https://github.com/seongyeon1/rebel100-gliner-decision)
- [RBLN SDK 0.11.2 Python API](https://docs.rbln.ai/v0.11.2/software/api/python/python_api.html)
- [RBLN SDK 0.11.2 장치 선택](https://docs.rbln.ai/v0.11.2/software/api/troubleshoot.html#device-selection)

## 라이선스

Apache License 2.0. 모델 가중치와 RBLN SDK는 이 저장소에 포함하지 않는다. vendored prompt
format 코드의 출처는 [NOTICE](NOTICE)에 기록했다.
