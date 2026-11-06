"""Pinned, code-free Hugging Face snapshot acquisition and integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .constants import MODEL_ID, MODEL_REVISION

MANIFEST_NAME = "snapshot.json"
INDEX_NAME = "model.safetensors.index.json"
ALLOW_PATTERNS = (
    "config.json",
    "decision_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "readout.safetensors",
    "model.safetensors",
    "model-*.safetensors",
    INDEX_NAME,
    "LICENSE",
    "NOTICE",
)
REQUIRED_FILES = (
    "config.json",
    "decision_config.json",
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    INDEX_NAME,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise ValueError(f"snapshot contains a symbolic link: {relative}")
        if path.is_file() and relative.as_posix() != MANIFEST_NAME:
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _allowed(relative: str) -> bool:
    name = Path(relative).name
    if relative != name:
        return False
    if name in ALLOW_PATTERNS:
        return True
    return name.startswith("model-") and name.endswith(".safetensors")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _referenced_shards(root: Path) -> list[str]:
    index = _load_json(root / INDEX_NAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model shard index has no non-empty weight_map")
    shards: set[str] = set()
    for value in weight_map.values():
        if not isinstance(value, str):
            raise ValueError("model shard index contains a non-string filename")
        path = Path(value)
        if path.is_absolute() or path.name != value or value in (".", ".."):
            raise ValueError(f"unsafe model shard filename: {value!r}")
        if not value.startswith("model-") or not value.endswith(".safetensors"):
            raise ValueError(f"unexpected model shard filename: {value!r}")
        shards.add(value)
    return sorted(shards)


def fetch_snapshot(
    destination: Path,
    *,
    downloader: Callable[..., str] | None = None,
) -> Path:
    """Download the pinned model data while excluding all remote Python code."""

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"snapshot destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    downloaded = Path(
        downloader(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(destination),
            allow_patterns=list(ALLOW_PATTERNS),
        )
    )
    if downloaded.resolve() != destination.resolve():
        raise RuntimeError("downloader returned a different snapshot directory")

    files = _snapshot_files(destination)
    unexpected = [path.relative_to(destination).as_posix() for path in files]
    unexpected = [relative for relative in unexpected if not _allowed(relative)]
    if unexpected:
        raise ValueError(f"snapshot contains disallowed files: {unexpected}")

    missing = [name for name in REQUIRED_FILES if not (destination / name).is_file()]
    if missing:
        raise FileNotFoundError(f"snapshot is missing required files: {missing}")
    shards = _referenced_shards(destination)
    missing_shards = [name for name in shards if not (destination / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(f"snapshot is missing referenced shards: {missing_shards}")

    provenance = {
        "files": {
            path.relative_to(destination).as_posix(): {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in files
        },
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }
    manifest = destination / MANIFEST_NAME
    temporary = destination / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
    validate_snapshot(destination)
    return destination


def validate_snapshot(root: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate pinning, shard references, file coverage, sizes, and hashes."""

    root = Path(root)
    manifest = _load_json(root / MANIFEST_NAME)
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("snapshot model_id does not match the pinned model")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError("snapshot revision does not match the pinned revision")
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("snapshot manifest has no file hashes")

    files = _snapshot_files(root)
    actual = {path.relative_to(root).as_posix(): path for path in files}
    recorded = set(records)
    if set(actual) != recorded:
        missing_hashes = sorted(set(actual) - recorded)
        missing_files = sorted(recorded - set(actual))
        raise ValueError(
            f"snapshot file coverage mismatch: missing_hashes={missing_hashes}, "
            f"missing_files={missing_files}"
        )
    if any(not _allowed(relative) for relative in actual):
        raise ValueError("snapshot manifest contains a disallowed file")

    missing = [name for name in REQUIRED_FILES if name not in actual]
    if missing:
        raise FileNotFoundError(f"snapshot is missing required files: {missing}")
    for shard in _referenced_shards(root):
        if shard not in actual:
            raise FileNotFoundError(f"snapshot is missing referenced shard: {shard}")

    for relative, path in actual.items():
        record = records[relative]
        if not isinstance(record, dict):
            raise ValueError(f"invalid snapshot record: {relative}")
        if record.get("size") != path.stat().st_size:
            raise ValueError(f"snapshot size mismatch: {relative}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"snapshot has invalid SHA-256: {relative}")
        if verify_hashes and _sha256(path) != expected_hash:
            raise ValueError(f"snapshot hash mismatch: {relative}")
    return manifest
