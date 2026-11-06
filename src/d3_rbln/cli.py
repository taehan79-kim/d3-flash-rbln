"""Compile on the SDK host, then explicitly validate on two operator-selected chips."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from .constants import MODEL_ID, MODEL_REVISION, NUM_DEVICES, TARGET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def environment() -> dict:
    versions = {}
    for name in (
        "torch",
        "torch-rbln",
        "rebel-compiler",
        "optimum-rbln",
        "transformers",
        "safetensors",
        "huggingface-hub",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
        "rebel_import_available": importlib.util.find_spec("rebel") is not None,
        "rbln_smi_available": shutil.which("rbln-smi") is not None,
        "note": "Presence/version information only; no ABI, device readiness or compilation claim.",
    }


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict) or "question" not in row:
                raise ValueError("Each JSONL row must contain a question object")
            rows.append(row)
    if not 1 <= len(rows) <= 100:
        raise ValueError("Use 1–100 validation examples")
    return rows


def compile_command(args: argparse.Namespace) -> dict:
    from .compiler import compile_model

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = "preflight"
    try:
        # Fail before loading a 9B checkpoint if no vendor compiler is available.
        if importlib.util.find_spec("rebel") is None:
            raise RuntimeError(
                "RBLN SDK is absent; run compilation in the isolated SDK environment"
            )
        import torch
        from safetensors.torch import save_file

        from .model import load_bundle, prepare
        from .snapshot import validate_snapshot
        from .validation import compare_logits

        torch.set_num_threads(args.cpu_threads)
        stage = "snapshot_validation"
        provenance = validate_snapshot(args.model_dir)
        stage = "load_model"
        bundle = load_bundle(args.model_dir)
        rows = read_rows(args.data)
        tensors, cases = {}, []
        prepared_cases = []
        stage = "cpu_reference_parity"
        with torch.inference_mode():
            for i, row in enumerate(rows):
                prepared = prepare(bundle.tokenizer, bundle.decision, row, args.bucket)
                fixed = bundle.model(*prepared.tensors)
                reference = bundle.model.reference(*prepared.tensors)
                same_shape = compare_logits(
                    fixed,
                    reference,
                    options=len(prepared.labels),
                    temperature=bundle.decision["temperature"],
                    atol=1e-5,
                    rtol=1e-5,
                    probability_atol=1e-5,
                )
                # Fixed left padding must not materially change the unpadded decision.
                n = prepared.token_count
                natural = bundle.model.reference(
                    prepared.input_ids[:, -n:], prepared.attention_mask[:, -n:]
                )
                padding = compare_logits(
                    fixed,
                    natural,
                    options=len(prepared.labels),
                    temperature=bundle.decision["temperature"],
                    atol=args.atol,
                    rtol=args.rtol,
                    probability_atol=args.probability_atol,
                )
                cases.append(
                    {
                        "index": i,
                        "token_count": n,
                        "options": len(prepared.labels),
                        "cpu_reference_parity": same_shape,
                        "padding_parity": padding,
                    }
                )
                if not same_shape["passed"] or not padding["passed"]:
                    write_json(output / "cpu-parity-failure.json", {"cases": cases})
                    raise RuntimeError(f"CPU reference/padding parity failed at example {i}")
                tensors[f"ids_{i}"] = prepared.input_ids.contiguous()
                tensors[f"mask_{i}"] = prepared.attention_mask.contiguous()
                tensors[f"reference_{i}"] = reference.detach().cpu().contiguous()
                prepared_cases.append(prepared)
        save_file(tensors, str(output / "validation-inputs.safetensors"))
        write_json(
            output / "preparation.json",
            {
                "model_id": MODEL_ID,
                "model_revision": provenance["revision"],
                "bucket": args.bucket,
                "batch_size": 1,
                "text_only": True,
                "backbone_dtype": "bfloat16",
                "readout_dtype": "float32",
                "temperature": bundle.decision["temperature"],
                "cases": cases,
                "data_sha256": sha256(args.data),
                "inputs_sha256": sha256(output / "validation-inputs.safetensors"),
                "thresholds": {
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "probability_atol": args.probability_atol,
                },
                "reference_scope": "HF text forward with d3 masks/readout; not full upstream API",
                "environment": environment(),
            },
        )
        stage = "strict_rbln_compilation"
        with torch.inference_mode():
            report = compile_model(
                bundle.model,
                prepared_cases[0].tensors,
                output,
                target=TARGET,
                num_devices=NUM_DEVICES,
            )
        return {
            "output": str(output),
            "compilation_succeeded": report["compilation_succeeded"],
            "runtime_validated": False,
            "numerical_validated": False,
            "next": "Validate the saved artifact explicitly on two selected devices.",
        }
    except Exception as exc:
        if not (output / "failure.json").exists():
            write_json(
                output / "failure.json",
                {
                    "stage": stage,
                    "compilation_succeeded": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "environment": environment(),
                },
            )
        raise


def selected_devices(env: dict[str, str]) -> list[int]:
    raw = env.get("RBLN_VISIBLE_DEVICES", "")
    if not raw:
        raise ValueError(
            "Set RBLN_VISIBLE_DEVICES to exactly two chosen physical IDs before launch"
        )
    parts = raw.split(",")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError("RBLN_VISIBLE_DEVICES must contain exactly two numeric IDs, e.g. X,Y")
    devices = [int(p.strip()) for p in parts]
    if len(set(devices)) != 2:
        raise ValueError("Choose two distinct visible devices")
    if env.get("RBLN_DEVICES") and env["RBLN_DEVICES"] != raw:
        raise ValueError("Conflicting RBLN_DEVICES and RBLN_VISIBLE_DEVICES")
    return devices


def validate_command(args: argparse.Namespace) -> dict:
    devices = selected_devices(dict(os.environ))
    import torch
    from safetensors.torch import load_file

    from .compiler import load_runtime
    from .validation import compare_logits

    root = args.artifact_dir.resolve()
    compiled = json.loads((root / "compile.json").read_text())
    prepared = json.loads((root / "preparation.json").read_text())
    if prepared.get("model_revision") != MODEL_REVISION:
        raise ValueError("Artifact preparation does not match the pinned model revision")
    if not compiled.get("compilation_succeeded") or compiled.get("requested_num_devices") != 2:
        raise ValueError("A successful two-device compile report is required")
    if sha256(root / "model.rbln") != compiled["artifact_sha256"]:
        raise ValueError("Artifact hash mismatch")
    if sha256(root / "validation-inputs.safetensors") != prepared["inputs_sha256"]:
        raise ValueError("Validation input hash mismatch")
    output = args.output or root / f"validation-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    if output.exists():
        raise FileExistsError(output)
    report = {
        "runtime_validated": False,
        "numerical_validated": False,
        "model_revision": MODEL_REVISION,
        "artifact_sha256": compiled["artifact_sha256"],
        "inputs_sha256": prepared["inputs_sha256"],
        "thresholds": prepared["thresholds"],
        "compile_sdk_versions": compiled.get("sdk_versions"),
        "visible_devices": devices,
        "runtime_device_argument": args.runtime_device,
        "device_mapping_note": "Operator must verify SDK mapping and rbln-smi contexts.",
        "environment": environment(),
        "cases": [],
        "host_ops_detected": compiled.get("host_ops_detected"),
        "physical_acceleration_verified": False,
    }
    try:
        runtime = load_runtime(root / "model.rbln", device=args.runtime_device)
        tensors = load_file(str(root / "validation-inputs.safetensors"), device="cpu")
        timings = []
        with torch.inference_mode():
            for case in prepared["cases"]:
                i = case["index"]
                ids, mask = tensors[f"ids_{i}"], tensors[f"mask_{i}"]
                actual = None
                checks = []
                for repetition in range(args.warmup + args.repeats):
                    started = time.perf_counter()
                    value = runtime(ids, mask)
                    if isinstance(value, (list, tuple)) and len(value) == 1:
                        value = value[0]
                    actual = torch.as_tensor(value).detach().cpu()
                    if repetition >= args.warmup:
                        timings.append((time.perf_counter() - started) * 1000)
                    check = compare_logits(
                        actual,
                        tensors[f"reference_{i}"],
                        options=case["options"],
                        temperature=prepared["temperature"],
                        **prepared["thresholds"],
                    )
                    checks.append({"repetition": repetition, **check})
                assert actual is not None
                report["cases"].append(
                    {
                        "index": i,
                        "passed": all(check["passed"] for check in checks),
                        "checks": checks,
                        "cpu_reference_logits": tensors[f"reference_{i}"].tolist(),
                        "last_actual_logits": [
                            [v if math.isfinite(v) else None for v in row]
                            for row in actual.tolist()
                        ],
                    }
                )
        report["runtime_validated"] = True
        report["numerical_validated"] = bool(report["cases"]) and all(
            c["passed"] for c in report["cases"]
        )
        report["latency_ms"] = {
            "median": statistics.median(timings),
            "p50": statistics.median(timings),
            "p95": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
            "percentile_method": "p50 median; p95 nearest-rank",
            "min": min(timings),
            "max": max(timings),
            "mean": statistics.mean(timings),
            "samples": len(timings),
            "scope": "synchronous runtime call plus output transfer; not full API or pure NPU time",
        }
        write_json(output, report)
        if not report["numerical_validated"]:
            raise RuntimeError(f"NPU numerical validation failed; see {output}")
        return {"report": str(output), "numerical_validated": True, "production_ready": False}
    except Exception as exc:
        if not output.exists():
            report.update(error_type=type(exc).__name__, error_message=str(exc))
            write_json(output, report)
        raise


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Read dependency availability; no installs or device allocation"
    )
    fetch = commands.add_parser(
        "fetch", help="Explicitly download the pinned ~16.8GB model snapshot"
    )
    fetch.add_argument("--output", type=Path, required=True)
    compile_ = commands.add_parser(
        "compile", help="CPU parity and strict two-device compile; NO runtime"
    )
    compile_.add_argument("--model-dir", type=Path, required=True)
    compile_.add_argument("--data", type=Path, default=Path(__file__).with_name("examples.jsonl"))
    compile_.add_argument("--bucket", type=positive, default=256)
    compile_.add_argument("--output", type=Path, required=True)
    compile_.add_argument("--cpu-threads", type=positive, default=4)
    compile_.add_argument("--atol", type=float, default=0.1)
    compile_.add_argument("--rtol", type=float, default=0.01)
    compile_.add_argument("--probability-atol", type=float, default=0.02)
    validate = commands.add_parser(
        "validate", help="Explicit target-side runtime and numerical check"
    )
    validate.add_argument("--artifact-dir", type=Path, required=True)
    validate.add_argument("--runtime-device", type=int, required=True)
    validate.add_argument("--warmup", type=positive, default=2)
    validate.add_argument("--repeats", type=positive, default=5)
    validate.add_argument("--output", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = environment()
        elif args.command == "fetch":
            from .snapshot import fetch_snapshot

            result = {
                "model_directory": str(fetch_snapshot(args.output)),
                "revision": MODEL_REVISION,
            }
        elif args.command == "compile":
            result = compile_command(args)
        else:
            result = validate_command(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
