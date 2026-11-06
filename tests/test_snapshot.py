from __future__ import annotations

import json
from pathlib import Path

import pytest

from d3_rbln.constants import MODEL_ID, MODEL_REVISION
from d3_rbln.snapshot import fetch_snapshot, validate_snapshot


def _populate(root: Path, *, shard_name: str = "model-00001-of-00001.safetensors") -> None:
    files = {
        "config.json": "{}",
        "decision_config.json": "{}",
        "readout.safetensors": "readout",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors.index.json": json.dumps({"weight_map": {"backbone.weight": shard_name}}),
        shard_name: "weights",
        "LICENSE": "Apache-2.0",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


class FakeDownloader:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        root = Path(kwargs["local_dir"])
        _populate(root)
        (root / ".cache").mkdir()
        (root / ".cache" / "download-state").write_text("ignored")
        return str(root)


def test_fetch_uses_pinned_revision_and_records_all_file_hashes(tmp_path):
    downloader = FakeDownloader()
    root = tmp_path / "snapshot"

    result = fetch_snapshot(root, downloader=downloader)

    assert result == root
    assert downloader.kwargs is not None
    assert downloader.kwargs["repo_id"] == MODEL_ID
    assert downloader.kwargs["revision"] == MODEL_REVISION
    assert "*.py" not in downloader.kwargs["allow_patterns"]
    manifest = validate_snapshot(root)
    assert manifest["revision"] == MODEL_REVISION
    assert "model-00001-of-00001.safetensors" in manifest["files"]
    assert "snapshot.json" not in manifest["files"]
    assert all(len(record["sha256"]) == 64 for record in manifest["files"].values())


def test_fetch_refuses_a_nonempty_destination(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "keep").write_text("existing")

    with pytest.raises(FileExistsError, match="not empty"):
        fetch_snapshot(root, downloader=FakeDownloader())

    assert (root / "keep").read_text() == "existing"


def test_fetch_rejects_remote_python_even_if_downloader_returns_it(tmp_path):
    def downloader(**kwargs):
        root = Path(kwargs["local_dir"])
        _populate(root)
        (root / "modeling_remote.py").write_text("raise RuntimeError('do not execute')")
        return str(root)

    with pytest.raises(ValueError, match="disallowed"):
        fetch_snapshot(tmp_path / "snapshot", downloader=downloader)


def test_path_traversal_in_shard_index_is_rejected(tmp_path):
    def downloader(**kwargs):
        root = Path(kwargs["local_dir"])
        _populate(root, shard_name="model-00001-of-00001.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "../model-escape.safetensors"}})
        )
        return str(root)

    with pytest.raises(ValueError, match="unsafe model shard"):
        fetch_snapshot(tmp_path / "snapshot", downloader=downloader)


def test_missing_referenced_shard_is_rejected(tmp_path):
    def downloader(**kwargs):
        root = Path(kwargs["local_dir"])
        _populate(root)
        (root / "model-00001-of-00001.safetensors").unlink()
        return str(root)

    with pytest.raises(FileNotFoundError, match="referenced shards"):
        fetch_snapshot(tmp_path / "snapshot", downloader=downloader)


def test_changed_file_hash_is_rejected(tmp_path):
    root = fetch_snapshot(tmp_path / "snapshot", downloader=FakeDownloader())
    (root / "readout.safetensors").write_text("changed")

    with pytest.raises(ValueError, match="mismatch"):
        validate_snapshot(root)


def test_missing_hash_record_is_rejected(tmp_path):
    root = fetch_snapshot(tmp_path / "snapshot", downloader=FakeDownloader())
    manifest_path = root / "snapshot.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"]["readout.safetensors"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing_hashes"):
        validate_snapshot(root)


def test_wrong_revision_is_rejected(tmp_path):
    root = fetch_snapshot(tmp_path / "snapshot", downloader=FakeDownloader())
    manifest_path = root / "snapshot.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["revision"] = "main"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="pinned revision"):
        validate_snapshot(root)


def test_hash_verification_can_be_skipped_but_size_is_still_checked(tmp_path):
    root = fetch_snapshot(tmp_path / "snapshot", downloader=FakeDownloader())
    path = root / "LICENSE"
    path.write_text("Not-Apache")  # same byte count as the original content

    manifest = validate_snapshot(root, verify_hashes=False)

    assert manifest["model_id"] == MODEL_ID
