# 오프라인 NPU 서버 반입

모델 snapshot과 애플리케이션 wheel을 **연결된 Linux x86_64 / Python 3.12 환경**에서 준비한
뒤 NPU 서버로 옮긴다. 기존 RBLN SDK·PyTorch는 반입하거나 교체하지 않는다.

## 1. 연결된 머신에서 모델 snapshot 받기

```bash
python3.12 -m venv .venv-download
source .venv-download/bin/activate
python -m pip install build 'huggingface-hub>=1,<2'
python -m build --wheel
python -m pip install --no-deps dist/d3_flash_rbln-*.whl
d3-rbln fetch --output models/d3-flash
python -m tarfile -c d3-flash-model.tar.gz models/d3-flash
```

`fetch`는 고정 revision만 내려받고 `snapshot.json`에 파일 크기와 SHA-256을 기록한다.
모델은 약 16.8 GB이므로 임시 공간과 전송 공간을 모두 확인한다.

전체 홈 디렉터리, Hugging Face cache, `.env`, shell history를 압축하지 않는다.

## 2. 애플리케이션 wheelhouse 준비

RBLN SDK·`torch`·`torch-rbln`·`rebel-compiler`는 대상 서버의 설치본을 유지한다.
`requirements-app-lock.txt`는 이 네 패키지를 제외한 애플리케이션 의존성 전체를 고정한다.
다음 명령은 Python 3.12용 wheel을 의존성 재해석 없이 그대로 받는다.

```bash
python -m pip download \
  --dest wheelhouse \
  --no-deps \
  --platform manylinux_2_28_x86_64 \
  --platform manylinux_2_17_x86_64 \
  --platform manylinux2014_x86_64 \
  --python-version 312 \
  --implementation cp \
  --abi cp312 \
  --abi abi3 \
  --abi none \
  --only-binary=:all: \
  -r requirements-app-lock.txt

cp dist/d3_flash_rbln-*.whl wheelhouse/
python -m tarfile -c d3-flash-wheelhouse.tar.gz wheelhouse
```

lock 파일에 `torch`, `torch-rbln`, `rebel-compiler`, `optimum-rbln`이 없는지 먼저 확인한다.
`--no-deps`를 제거하면 `accelerate` 등의 dependency resolver가 일반 PyPI `torch`를 추가할 수
있으므로 제거하지 않는다. 여러 manylinux·ABI tag는 호환 가능한 wheel 탐색 범위를 넓히지만,
모든 Linux 배포판에서의 실행을 보장하지 않는다. `--only-binary=:all:`이 실패하면 source
build로 자동 전환하지 말고 대상 환경과 호환되는 wheel을 별도로 확인한다.

## 3. SHA-256 목록 만들기

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path

def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

names = ["d3-flash-model.tar.gz", "d3-flash-wheelhouse.tar.gz"]
with Path("SHA256SUMS").open("w", encoding="utf-8") as out:
    for name in names:
        out.write(f"{file_sha256(Path(name))}  {name}\n")
PY
```

대용량 파일을 위해 운영 반입 도구가 별도 해시 기능을 제공하면 그 기능을 우선한다.

## 4. NPU 서버에서 검증·설치

반입 파일을 전용 작업 디렉터리에 둔 뒤:

```bash
sha256sum -c SHA256SUMS
python -m tarfile -e d3-flash-model.tar.gz .
python -m tarfile -e d3-flash-wheelhouse.tar.gz .

source /path/to/vendor-sdk-venv/bin/activate
python -m pip install --no-index --find-links wheelhouse \
  --no-deps -r requirements-app-lock.txt
python -m pip install --no-index --find-links wheelhouse \
  --no-deps wheelhouse/d3_flash_rbln-*.whl
python -m pip check
d3-rbln doctor
```

설치 전후에 다음 버전을 보관한다.

```bash
python - <<'PY'
from importlib.metadata import version

for name in ("torch", "torch-rbln", "rebel-compiler", "transformers"):
    try:
        print(f"{name}={version(name)}")
    except Exception as exc:
        print(f"{name}=unavailable ({type(exc).__name__})")
PY
```

`pip check` 충돌을 제약 삭제나 강제 재설치로 우회하지 않는다. 특히
`transformers==5.17.0`과 서버의 다른 Optimum 작업이 요구하는 버전이 다르면 전용 가상환경을
사용한다.

컴파일 전에 `free -h`, `df -h` 등 운영 환경의 표준 도구로 호스트 RAM과 작업 디스크를
확인한다. 모델 파일 외에도 CPU 로딩·대조 실행과 compiler workspace가 필요하므로 모델 파일
크기만큼의 여유만으로는 충분하다고 판단하지 않는다.

## 5. snapshot 무결성 확인

`compile`이 자동으로 전체 해시를 확인하지만, 컴파일 전에 별도 점검할 수 있다.

```bash
d3-rbln doctor
d3-rbln compile \
  --model-dir models/d3-flash \
  --bucket 256 \
  --output results/compile-256
```

`--data`를 생략하면 설치된 wheel에 포함된 smoke examples를 사용한다. 서버의 작업 디렉터리에
저장소 루트 `examples.jsonl`이 있다고 가정하지 않는다.

컴파일 중 네트워크 다운로드가 발생하면 안 된다. 누락 파일은 연결된 머신에서 고정
snapshot을 다시 준비해 반입한다.

## 반입 금지 항목

- Hugging Face·GitHub·SDK 토큰
- `.env`와 shell 설정 파일
- 기존 서비스의 API 키나 인증서
- 전체 홈 디렉터리
- 다른 NPU 또는 SDK 버전에서 만든 컴파일 cache
