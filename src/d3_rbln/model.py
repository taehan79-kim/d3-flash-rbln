"""Tensor-only d3 text graph; no generation, KV cache, vision, or GPU-only kernels.

Full-attention blocks see all non-padding keys; Gated DeltaNet blocks retain their
causal recurrence. This is NOT the standard causal Qwen generation graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import vendor_d3_format as fmt
from .constants import FORMAT_ID, TRANSFORMERS_VERSION


def validate_decision_config(config: dict[str, Any]) -> None:
    expected = {
        "format_version": 1,
        "format_id": FORMAT_ID,
        "base_model": "Qwen/Qwen3.5-9B",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "readout_dtype": "float32",
        "prompt": "d3",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported decision_config {key}: {config.get(key)!r}")
    codes, token_ids = config.get("codes", []), config.get("token_ids", [])
    if len(codes) != 255 or len(token_ids) != 255:
        raise ValueError("Expected 255 answer codes and token IDs")
    if len(set(codes)) != 255 or len(set(token_ids)) != 255:
        raise ValueError("Answer codes and token IDs must be unique")
    temperature = config.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")


class TextDecision(nn.Module):
    """One model graph distributed by the RBLN compiler across two devices."""

    def __init__(self, language_model: Any, readout: torch.Tensor):
        super().__init__()
        if readout.ndim != 2 or readout.shape != (255, language_model.config.hidden_size):
            raise ValueError("Expected readout shape [255, hidden_size]")
        self.language_model = language_model
        self.register_buffer("readout", readout.detach().to(torch.float32))
        self.layer_types = tuple(language_model.config.layer_types)
        if len(self.layer_types) != len(language_model.layers):
            raise ValueError("Layer type count does not match the text model")
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError("Unsupported text layer type")
        self.eval().requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        model = self.language_model
        hidden = model.embed_tokens(input_ids)
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).view(1, 1, -1)
        positions = positions.expand(4, batch, -1)
        rope = model.rotary_emb(hidden, positions[1:])
        # No bool inversion (~), causal mask, or data-dependent all-ones branch.
        keys = attention_mask.to(torch.bool)[:, None, None, :]
        recurrent = attention_mask.contiguous()
        for layer, kind in zip(model.layers, self.layer_types, strict=True):
            hidden = layer(
                hidden,
                position_embeddings=rope,
                attention_mask=keys if kind == "full_attention" else recurrent,
                position_ids=positions[0],
                past_key_values=None,
                use_cache=False,
            )
        last = model.norm(hidden)[:, -1, :]
        return last.float() @ self.readout.T

    @torch.inference_mode()
    def reference(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """HF model's complete forward with d3 mask semantics, sharing CPU weights.

        Independent traversal from our tensor wrapper. Not a measured comparison
        against the full 9B upstream system_one server; that is a separate gate.
        """
        output = self.language_model(
            input_ids=input_ids,
            attention_mask={
                "full_attention": attention_mask.bool()[:, None, None, :],
                "linear_attention": attention_mask,
            },
            use_cache=False,
        )
        return output.last_hidden_state[:, -1, :].float() @ self.readout.T


@dataclass
class Prepared:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_count: int
    question: dict[str, Any]
    labels: list[str]

    @property
    def tensors(self) -> list[torch.Tensor]:
        return [self.input_ids, self.attention_mask]


def prepare(tokenizer: Any, decision: dict, row: dict, bucket: int) -> Prepared:
    if bucket < 2:
        raise ValueError("bucket must be >= 2")
    if any(row.get(k) for k in ("images", "videos")):
        raise ValueError("This first port supports text only, not images/videos")
    question = row["question"]
    if question.get("type") not in {"choice", "noul"}:
        raise ValueError("Only choice and noul questions are supported in this first port")
    if question["type"] == "choice":
        criteria = question.get("criteria")
        if (
            not isinstance(criteria, dict)
            or not criteria
            or not all(isinstance(k, str) for k in criteria)
        ):
            raise ValueError("choice criteria must be a non-empty mapping of string labels")
    prompt = fmt.render(tokenizer, row.get("state"), question, decision["codes"])
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    maximum = decision.get("max_length")
    if not ids or len(ids) > bucket or (maximum is not None and len(ids) > maximum):
        raise ValueError(f"Input has {len(ids)} tokens; bucket={bucket}; no truncation allowed")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    gap = bucket - len(ids)
    return Prepared(
        input_ids=torch.tensor([[tokenizer.pad_token_id] * gap + ids], dtype=torch.int64),
        attention_mask=torch.tensor([[0] * gap + [1] * len(ids)], dtype=torch.int64),
        token_count=len(ids),
        question=question,
        labels=fmt.options(question)[0],
    )


def answer(logits: torch.Tensor, prepared: Prepared, decision: dict) -> dict:
    if logits.shape != (1, 255) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite logits with shape [1,255]")
    probs = torch.softmax(
        logits[0, : len(prepared.labels)].float() / decision["temperature"], dim=-1
    ).tolist()
    return fmt.to_answer(prepared.question, probs)


@dataclass
class Bundle:
    model: TextDecision
    tokenizer: Any
    decision: dict


def load_bundle(path: str | Path) -> Bundle:
    """Load built-in HF classes locally; never execute snapshot Python code."""
    from importlib.metadata import version

    if version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Expected transformers {TRANSFORMERS_VERSION}; isolate the environment")
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5Model
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        function = getattr(qwen, name)
        if function.__module__.startswith(("fla", "causal_conv1d", "kernels")):
            raise RuntimeError(
                "Use an isolated environment without GPU-only FLA/causal-conv kernels"
            )
    root = Path(path).resolve()
    decision = json.loads((root / "decision_config.json").read_text())
    validate_decision_config(decision)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("This port is only for the pinned Qwen3.5 d3-flash checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    codes, ids = fmt.answer_codes(tokenizer)
    if codes != decision["codes"] or ids != decision["token_ids"]:
        raise ValueError("Tokenizer and decision answer-code vocabulary do not match")
    backbone = Qwen3_5Model.from_pretrained(
        root,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    readout = load_file(str(root / "readout.safetensors"), device="cpu")["weight"]
    model = TextDecision(backbone.language_model, readout)
    # Do not retain the unused vision tower in the compilation model.
    del backbone
    return Bundle(model, tokenizer, decision)
