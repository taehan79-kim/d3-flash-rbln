"""Strict, evidence-producing RBLN compilation helpers.

The module deliberately imports the vendor SDK lazily so that package tests and
CPU-side preparation do not require an RBLN installation.  Device visibility is
an external process concern (for example ``RBLN_VISIBLE_DEVICES``); this module
never guesses physical device identifiers.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable

ARTIFACT_NAME = "model.rbln"
REPORT_NAME = "compile.json"
FAILURE_NAME = "failure.json"
_SDK_PACKAGES = ("rebel-compiler", "torch-rbln", "torch")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _SDK_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _json_safe(value: Any) -> Any:
    """Convert SDK inspection results to deterministic JSON-compatible data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    raise FileNotFoundError(f"compiled artifact was not created: {path}")


def _find_num_devices(value: Any) -> int | None:
    if isinstance(value, dict):
        candidate = value.get("num_devices")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        for child in value.values():
            found = _find_num_devices(child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_num_devices(child)
            if found is not None:
                return found
    return None


def _host_ops_status(value: Any) -> bool | None:
    """Return True/False when inspect metadata contains host-op evidence, else None."""

    found = False
    detected = False

    def walk(current: Any) -> None:
        nonlocal found, detected
        if isinstance(current, dict):
            if "host_op_reason" in current:
                found = True
                reasons = current["host_op_reason"]
                if isinstance(reasons, dict) and bool(reasons):
                    detected = True
            for child in current.values():
                walk(child)
        elif isinstance(current, (list, tuple)):
            for child in current:
                walk(child)

    walk(value)
    return detected if found else None


def _inspection_data(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("RBLN inspection returned invalid JSON text") from exc
    value = _json_safe(value)
    if isinstance(value, dict):
        return value
    raise TypeError("RBLN inspection did not return an object")


def _compiler_callable(compiler: Any | None) -> Callable[..., Any]:
    compile_from_torch = getattr(compiler, "compile_from_torch", compiler)
    if not callable(compile_from_torch):
        raise TypeError("compiler must be callable or expose compile_from_torch")
    return compile_from_torch


def compile_model(
    model: Any,
    example_inputs: Iterable[Any],
    output_dir: Path,
    *,
    target: str = "RBLN-CA25",
    num_devices: int = 2,
    compiler: Any | None = None,
) -> dict[str, Any]:
    """Strictly compile one D3 graph across exactly two logical NPUs.

    Success only means that the vendor compiler produced and described an
    artifact.  Runtime execution, numerical parity, and physical acceleration
    remain explicitly unvalidated until target-hardware checks are performed.
    """

    if num_devices != 2:
        raise ValueError("this package scopes one d3-flash model to exactly 2 devices")
    if not target.strip():
        raise ValueError("target must be a non-empty NPU identifier")

    output_dir = Path(output_dir)
    artifact = output_dir / ARTIFACT_NAME
    report_path = output_dir / REPORT_NAME
    failure_path = output_dir / FAILURE_NAME
    collisions = [path for path in (artifact, report_path, failure_path) if path.exists()]
    if collisions:
        names = ", ".join(path.name for path in collisions)
        raise FileExistsError(f"refusing to overwrite prior compilation evidence: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = list(example_inputs)
    if not inputs:
        raise ValueError("example_inputs must contain at least one tensor")

    try:
        sdk = compiler if compiler is not None else importlib.import_module("rebel")
        compile_from_torch = _compiler_callable(sdk)
        compiled = compile_from_torch(
            model,
            example_inputs=inputs,
            npu=target,
            num_devices=num_devices,
            model_trace_method="export_strict",
        )

        compiled.save(str(artifact))
        artifact_hash = _sha256(artifact)

        compiled_type = getattr(sdk, "RBLNCompiledModel", None)
        if compiled_type is None:
            raise AttributeError("RBLN SDK does not expose RBLNCompiledModel.inspect")
        inspect_method = getattr(compiled_type, "inspect", None)
        if not callable(inspect_method):
            raise AttributeError("RBLNCompiledModel does not expose inspect(path)")
        inspect_data = _inspection_data(inspect_method(str(artifact)))
        allocation_raw = compiled.get_alloc_per_node()
        if not isinstance(allocation_raw, (list, tuple)):
            raise TypeError("get_alloc_per_node() did not return a per-device sequence")
        allocation = [int(value) for value in allocation_raw]
        if len(allocation) != num_devices:
            raise RuntimeError(
                f"compiled allocation covers {len(allocation)} devices, expected {num_devices}"
            )
        if any(value <= 0 for value in allocation):
            raise RuntimeError("compiled allocation contains a non-positive device allocation")

        inspected_devices = _find_num_devices(inspect_data)
        if inspected_devices is not None and inspected_devices != num_devices:
            raise RuntimeError(
                f"compiler inspection reports {inspected_devices} devices, expected {num_devices}"
            )

        host_ops_detected = _host_ops_status(inspect_data)
        report: dict[str, Any] = {
            "artifact": ARTIFACT_NAME,
            "artifact_sha256": artifact_hash,
            "compilation_succeeded": True,
            "execution_claim": "unverified",
            "host_ops_detected": host_ops_detected,
            "inspect": inspect_data,
            "npu_target": target,
            "numerical_validated": False,
            "per_device_allocation_bytes": allocation,
            "requested_num_devices": num_devices,
            "runtime_validated": False,
            "sdk_versions": _package_versions(),
            "trace_method": "export_strict",
        }
        _write_json(report_path, report)
        return report
    except Exception as exc:
        failure = {
            "artifact_present": artifact.exists(),
            "compilation_succeeded": False,
            "error_message": str(exc),
            "error_type": type(exc).__name__,
            "npu_target": target,
            "requested_num_devices": num_devices,
            "sdk_versions": _package_versions(),
            "trace_method": "export_strict",
        }
        _write_json(failure_path, failure)
        raise


def load_runtime(
    artifact: Path,
    *,
    device: int = 0,
    sdk: Any | None = None,
) -> Any:
    """Load an artifact using one logical runtime start device.

    Physical device selection must be configured before process start using the
    target SDK's visibility mechanism.  A list of physical device IDs is not a
    supported argument to ``Runtime``.
    """

    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a non-negative logical device integer")
    artifact = Path(artifact)
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    if sdk is None:
        sdk = importlib.import_module("rebel")
    runtime_type = getattr(sdk, "Runtime", None)
    if runtime_type is None:
        raise AttributeError("RBLN SDK does not expose Runtime")
    return runtime_type(str(artifact), device=device, tensor_type="pt")
