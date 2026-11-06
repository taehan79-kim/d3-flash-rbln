from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from d3_rbln import cli
from d3_rbln.constants import MODEL_REVISION


def test_doctor_reports_availability_without_installing_or_allocating(monkeypatch, capsys):
    monkeypatch.setattr(cli, "environment", lambda: {"rebel_import_available": False})

    assert cli.main(["doctor"]) == 0

    assert json.loads(capsys.readouterr().out) == {"rebel_import_available": False}


@pytest.mark.parametrize(
    ("arguments", "option"),
    [
        (["compile", "--model-dir", "m", "--output", "o", "--bucket", "0"], "bucket"),
        (
            ["compile", "--model-dir", "m", "--output", "o", "--cpu-threads", "0"],
            "cpu-threads",
        ),
        (
            ["validate", "--artifact-dir", "a", "--runtime-device", "0", "--warmup", "0"],
            "warmup",
        ),
        (
            ["validate", "--artifact-dir", "a", "--runtime-device", "0", "--repeats", "0"],
            "repeats",
        ),
    ],
)
def test_cli_rejects_non_positive_size_and_iteration_arguments(arguments, option, capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.parser().parse_args(arguments)

    assert option in capsys.readouterr().err


def test_selected_devices_accepts_exactly_two_distinct_numeric_ids():
    assert cli.selected_devices({"RBLN_VISIBLE_DEVICES": "20, 21"}) == [20, 21]


@pytest.mark.parametrize("value", ["", "20", "20,21,22", "20,twenty-one", "20,20"])
def test_selected_devices_rejects_missing_ambiguous_or_duplicate_selection(value):
    with pytest.raises(ValueError):
        cli.selected_devices({"RBLN_VISIBLE_DEVICES": value})


def test_selected_devices_rejects_conflicting_visibility_aliases():
    with pytest.raises(ValueError, match="Conflicting"):
        cli.selected_devices({"RBLN_VISIBLE_DEVICES": "20,21", "RBLN_DEVICES": "22,23"})


def test_compile_without_sdk_fails_before_model_load_and_records_evidence(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "compile"
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)

    code = cli.main(
        [
            "compile",
            "--model-dir",
            str(tmp_path / "missing-model"),
            "--data",
            str(tmp_path / "missing-data.jsonl"),
            "--output",
            str(output),
        ]
    )

    assert code == 1
    failure = json.loads((output / "failure.json").read_text())
    assert failure["stage"] == "preflight"
    assert failure["compilation_succeeded"] is False
    assert failure["error_type"] == "RuntimeError"
    assert "RBLN SDK is absent" in capsys.readouterr().err
    assert not (output / "compile.json").exists()


def test_compile_output_collision_preserves_existing_evidence(tmp_path, capsys):
    output = tmp_path / "compile"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do-not-overwrite")

    code = cli.main(["compile", "--model-dir", "unused", "--output", str(output)])

    assert code == 1
    assert sentinel.read_text() == "do-not-overwrite"
    assert "FileExistsError" in capsys.readouterr().err
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


def test_cpu_parity_failure_blocks_vendor_compiler(tmp_path, monkeypatch):
    from d3_rbln import compiler, model, snapshot, validation

    class FakeModel:
        def __call__(self, *_):
            return torch.zeros(1, 255)

        def reference(self, *_):
            return torch.zeros(1, 255)

    prepared = SimpleNamespace(
        tensors=[torch.ones(1, 4, dtype=torch.int64)] * 2,
        input_ids=torch.ones(1, 4, dtype=torch.int64),
        attention_mask=torch.ones(1, 4, dtype=torch.int64),
        token_count=4,
        labels=["a", "b"],
    )
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(snapshot, "validate_snapshot", lambda path: {"revision": MODEL_REVISION})
    monkeypatch.setattr(
        model,
        "load_bundle",
        lambda path: SimpleNamespace(
            model=FakeModel(), tokenizer=object(), decision={"temperature": 1.0}
        ),
    )
    monkeypatch.setattr(model, "prepare", lambda *args: prepared)
    monkeypatch.setattr(cli, "read_rows", lambda path: [{"question": {}}])
    monkeypatch.setattr(
        validation,
        "compare_logits",
        lambda *args, **kwargs: {"passed": False, "finite": True},
    )
    compile_calls = []
    monkeypatch.setattr(compiler, "compile_model", lambda *a, **k: compile_calls.append((a, k)))
    args = argparse.Namespace(
        output=tmp_path / "compile",
        model_dir=tmp_path / "model",
        data=tmp_path / "data.jsonl",
        bucket=16,
        cpu_threads=1,
        atol=0.1,
        rtol=0.01,
        probability_atol=0.02,
    )

    with pytest.raises(RuntimeError, match="CPU reference/padding parity failed"):
        cli.compile_command(args)

    assert compile_calls == []
    assert json.loads((args.output / "failure.json").read_text())["stage"] == (
        "cpu_reference_parity"
    )
    assert (args.output / "cpu-parity-failure.json").exists()
    assert not (args.output / "compile.json").exists()


def _validation_fixture(root):
    artifact = root / "model.rbln"
    artifact.write_bytes(b"artifact")
    inputs = root / "validation-inputs.safetensors"
    save_file(
        {
            "ids_0": torch.ones(1, 4, dtype=torch.int64),
            "mask_0": torch.ones(1, 4, dtype=torch.int64),
            "reference_0": torch.zeros(1, 255),
        },
        str(inputs),
    )
    (root / "compile.json").write_text(
        json.dumps(
            {
                "compilation_succeeded": True,
                "requested_num_devices": 2,
                "artifact_sha256": cli.sha256(artifact),
                "host_ops_detected": False,
            }
        )
    )
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "model_revision": MODEL_REVISION,
                "inputs_sha256": cli.sha256(inputs),
                "temperature": 1.0,
                "thresholds": {"atol": 0.1, "rtol": 0.01, "probability_atol": 0.02},
                "cases": [{"index": 0, "options": 2}],
            }
        )
    )


def test_numerical_failure_is_recorded_without_claiming_success(tmp_path, monkeypatch):
    from d3_rbln import compiler, validation

    root = tmp_path / "compiled"
    root.mkdir()
    _validation_fixture(root)
    monkeypatch.setenv("RBLN_VISIBLE_DEVICES", "20,21")
    monkeypatch.delenv("RBLN_DEVICES", raising=False)
    monkeypatch.setattr(compiler, "load_runtime", lambda *a, **k: lambda *x: torch.ones(1, 255))
    monkeypatch.setattr(
        validation,
        "compare_logits",
        lambda *args, **kwargs: {
            "passed": False,
            "finite": True,
            "decision_agreement": False,
        },
    )
    report_path = tmp_path / "validation.json"
    args = argparse.Namespace(
        artifact_dir=root,
        runtime_device=0,
        warmup=1,
        repeats=1,
        output=report_path,
    )

    with pytest.raises(RuntimeError, match="numerical validation failed"):
        cli.validate_command(args)

    report = json.loads(report_path.read_text())
    assert report["runtime_validated"] is True
    assert report["numerical_validated"] is False
    assert report["physical_acceleration_verified"] is False
    assert report["visible_devices"] == [20, 21]
    assert report["cases"][0]["passed"] is False


@pytest.mark.parametrize("intermittent_failure", [False, True])
def test_validation_checks_every_repetition_and_records_evidence(
    tmp_path, monkeypatch, intermittent_failure
):
    from d3_rbln import compiler

    root = tmp_path / "compiled"
    root.mkdir()
    _validation_fixture(root)
    monkeypatch.setenv("RBLN_VISIBLE_DEVICES", "8,9")
    monkeypatch.delenv("RBLN_DEVICES", raising=False)
    calls = []

    def runtime(*inputs):
        calls.append(inputs)
        # A failing warmup followed by matching outputs must not be hidden.
        if intermittent_failure and len(calls) == 1:
            return torch.ones(1, 255) * 100
        return torch.zeros(1, 255)

    monkeypatch.setattr(compiler, "load_runtime", lambda *a, **k: runtime)
    output = tmp_path / "validation.json"
    args = argparse.Namespace(
        artifact_dir=root, runtime_device=0, warmup=1, repeats=2, output=output
    )
    if intermittent_failure:
        with pytest.raises(RuntimeError, match="numerical validation failed"):
            cli.validate_command(args)
    else:
        result = cli.validate_command(args)
        assert result["numerical_validated"]
        assert not result["production_ready"]
    report = json.loads(output.read_text())
    assert len(calls) == 3
    assert report["numerical_validated"] is not intermittent_failure
    assert report["artifact_sha256"] == cli.sha256(root / "model.rbln")
    assert report["inputs_sha256"] == cli.sha256(root / "validation-inputs.safetensors")
    assert report["thresholds"]["atol"] == 0.1
    assert report["latency_ms"]["samples"] == 2
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["min"]
    assert len(report["cases"][0]["checks"]) == 3
    assert len(report["cases"][0]["cpu_reference_logits"][0]) == 255


def test_default_data_is_packaged_and_matches_repo_examples():
    args = cli.parser().parse_args(["compile", "--model-dir", "unused", "--output", "unused"])
    assert args.data.is_file()
    repo = Path(__file__).resolve().parents[1]
    assert args.data.read_bytes() == (repo / "examples.jsonl").read_bytes()
