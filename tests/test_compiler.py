from __future__ import annotations

import json
from pathlib import Path

import pytest

from d3_rbln.compiler import compile_model, load_runtime


class FakeCompiled:
    def __init__(self, *, allocations=(100, 200), inspected_devices=2, host_ops=False):
        self.allocations = allocations
        self.inspected_devices = inspected_devices
        self.host_ops = host_ops

    def inspection(self):
        reasons = {"node": {"aten::item": "unsupported"}} if self.host_ops else {}
        return {
            "num_devices": self.inspected_devices,
            "subgraph": {"stats": [{"host_op_reason": reasons}]},
        }

    def get_alloc_per_node(self):
        return self.allocations

    def save(self, path):
        Path(path).write_bytes(b"compiled-rbln-artifact")


class FakeCompiler:
    def __init__(self, compiled=None, error=None, *, inspection_as_json=False):
        self.compiled = compiled or FakeCompiled()
        self.error = error
        self.calls = []
        self.inspect_paths = []
        owner = self

        class Inspector:
            @classmethod
            def inspect(cls, path):
                owner.inspect_paths.append(path)
                value = owner.compiled.inspection()
                return json.dumps(value) if inspection_as_json else value

        self.RBLNCompiledModel = Inspector

    def compile_from_torch(self, model, **kwargs):
        self.calls.append((model, kwargs))
        if self.error:
            raise self.error
        return self.compiled


def test_compile_records_strict_two_device_evidence(tmp_path):
    compiler = FakeCompiler()

    report = compile_model(object(), ["ids", "mask"], tmp_path, compiler=compiler)

    _, kwargs = compiler.calls[0]
    assert kwargs == {
        "example_inputs": ["ids", "mask"],
        "npu": "RBLN-CA25",
        "num_devices": 2,
        "model_trace_method": "export_strict",
    }
    assert report["per_device_allocation_bytes"] == [100, 200]
    assert report["runtime_validated"] is False
    assert report["numerical_validated"] is False
    assert report["execution_claim"] == "unverified"
    assert len(report["artifact_sha256"]) == 64
    assert json.loads((tmp_path / "compile.json").read_text()) == report
    assert (tmp_path / "model.rbln").read_bytes() == b"compiled-rbln-artifact"
    assert compiler.inspect_paths == [str(tmp_path / "model.rbln")]
    assert not (tmp_path / "failure.json").exists()


def test_host_ops_are_reported_without_claiming_acceleration(tmp_path):
    report = compile_model(
        object(),
        ["ids"],
        tmp_path,
        compiler=FakeCompiler(FakeCompiled(host_ops=True)),
    )

    assert report["host_ops_detected"] is True
    assert report["execution_claim"] == "unverified"


def test_json_inspection_text_is_parsed_and_validated(tmp_path):
    report = compile_model(
        object(),
        ["ids"],
        tmp_path,
        compiler=FakeCompiler(inspection_as_json=True),
    )

    assert report["inspect"]["num_devices"] == 2
    assert report["host_ops_detected"] is False


def test_invalid_json_inspection_fails_after_preserving_artifact(tmp_path):
    compiler = FakeCompiler()
    setattr(compiler.RBLNCompiledModel, "inspect", classmethod(lambda cls, path: "not-json"))

    with pytest.raises(ValueError, match="invalid JSON"):
        compile_model(object(), ["ids"], tmp_path, compiler=compiler)

    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["artifact_present"] is True
    assert not (tmp_path / "compile.json").exists()


def test_missing_host_op_metadata_is_reported_as_unknown(tmp_path):
    compiled = FakeCompiled()
    compiled.inspection = lambda: {"num_devices": 2}

    report = compile_model(object(), ["ids"], tmp_path, compiler=FakeCompiler(compiled))

    assert report["host_ops_detected"] is None


@pytest.mark.parametrize(
    ("compiled", "message"),
    [
        (FakeCompiled(allocations=(100,)), "covers 1 devices"),
        (FakeCompiled(allocations=(100, 200), inspected_devices=4), "reports 4 devices"),
        (FakeCompiled(allocations=(100, 0)), "non-positive"),
    ],
)
def test_invalid_compiler_device_evidence_fails_closed(tmp_path, compiled, message):
    with pytest.raises(RuntimeError, match=message):
        compile_model(object(), ["ids"], tmp_path, compiler=FakeCompiler(compiled))

    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["compilation_succeeded"] is False
    assert failure["artifact_present"] is True
    assert (tmp_path / "model.rbln").exists()
    assert not (tmp_path / "compile.json").exists()


def test_compile_failure_is_recorded_and_raised(tmp_path):
    compiler = FakeCompiler(error=RuntimeError("unsupported operation"))

    with pytest.raises(RuntimeError, match="unsupported operation"):
        compile_model(object(), ["ids"], tmp_path, compiler=compiler)

    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["error_type"] == "RuntimeError"
    assert failure["error_message"] == "unsupported operation"
    assert failure["compilation_succeeded"] is False


def test_prior_evidence_is_never_overwritten(tmp_path):
    previous = tmp_path / "compile.json"
    previous.write_text("keep me")

    with pytest.raises(FileExistsError, match="compile.json"):
        compile_model(object(), ["ids"], tmp_path, compiler=FakeCompiler())

    assert previous.read_text() == "keep me"
    assert not (tmp_path / "failure.json").exists()


@pytest.mark.parametrize("num_devices", [0, 1, 3, 8])
def test_package_refuses_non_two_device_compilation(tmp_path, num_devices):
    with pytest.raises(ValueError, match="exactly 2 devices"):
        compile_model(object(), ["ids"], tmp_path, num_devices=num_devices, compiler=FakeCompiler())


class FakeSDK:
    @staticmethod
    def Runtime(path, **kwargs):
        return {"path": path, **kwargs}


def test_runtime_uses_one_logical_start_device_not_a_device_list(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")

    runtime = load_runtime(artifact, device=3, sdk=FakeSDK)

    assert runtime == {"path": str(artifact), "device": 3, "tensor_type": "pt"}


def test_runtime_rejects_a_physical_device_list(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")

    with pytest.raises(ValueError, match="logical device integer"):
        load_runtime(artifact, device=[20, 21], sdk=FakeSDK)  # type: ignore[arg-type]
