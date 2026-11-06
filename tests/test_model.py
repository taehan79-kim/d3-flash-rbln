import copy
from typing import Any

import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from d3_rbln.model import TextDecision, answer, prepare, validate_decision_config


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
        },
    )
    config._attn_implementation = "sdpa"
    language: Any = Qwen3_5TextModel(config)
    language = language.to(dtype=torch.bfloat16).eval()
    return TextDecision(language, torch.randn(255, 32))


def inputs(padding=4):
    ids = torch.arange(1, 17).view(1, -1)
    mask = torch.tensor([[0] * padding + [1] * (16 - padding)])
    ids[:, :padding] = 0
    return ids, mask


@pytest.mark.parametrize("padding", [0, 1, 4, 12])
def test_real_hybrid_qwen_cpu_reference_parity(tiny, padding):
    ids, mask = inputs(padding)
    with torch.inference_mode():
        actual = tiny(ids, mask)
        reference = tiny.reference(ids, mask)
    assert actual.shape == (1, 255)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_strict_export_accepts_different_padding_without_retracing(tiny):
    with torch.inference_mode():
        exported = torch.export.export(tiny, inputs(4), strict=True).module()
        for padding in (0, 1, 4, 12):
            ids, mask = inputs(padding)
            torch.testing.assert_close(exported(ids, mask), tiny(ids, mask), atol=0, rtol=0)


def test_full_attention_is_not_silently_replaced_with_causal_attention(tiny):
    ids, mask = inputs(4)
    with torch.inference_mode():
        d3 = tiny(ids, mask)
        causal = tiny.language_model(input_ids=ids, attention_mask=mask, use_cache=False)
        causal_logits = causal.last_hidden_state[:, -1].float() @ tiny.readout.T
    assert not torch.allclose(d3, causal_logits, atol=0.01, rtol=0.01)


def test_invalid_readout_rejected(tiny):
    with pytest.raises(ValueError, match="readout"):
        TextDecision(tiny.language_model, torch.zeros(254, 32))


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[-1]["content"]

    def encode(self, prompt, add_special_tokens=False):
        assert "State:" in prompt and "A: report" in prompt
        return [11, 12, 13]


@pytest.fixture
def decision():
    return {
        "format_version": 1,
        "format_id": "d3-code-readout-v1",
        "base_model": "Qwen/Qwen3.5-9B",
        "prompt": "d3",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "readout_dtype": "float32",
        "codes": ["A"] + [f"C{i}" for i in range(254)],
        "token_ids": list(range(255)),
        "temperature": 1.0,
    }


@pytest.fixture
def row():
    return {"state": "Example", "question": {"type": "choice", "criteria": {"report": None}}}


def test_left_padding_and_no_truncation(decision, row):
    p = prepare(Tokenizer(), decision, row, 8)
    assert p.input_ids.tolist() == [[0] * 5 + [11, 12, 13]]
    assert p.attention_mask.tolist() == [[0] * 5 + [1] * 3]
    assert p.token_count == 3
    with pytest.raises(ValueError, match="no truncation"):
        prepare(Tokenizer(), decision, row, 2)


def test_rejects_unsupported_request_types(decision, row):
    for patch in ({"images": ["example.png"]}, {"videos": ["example.mp4"]}):
        with pytest.raises(ValueError, match="text only"):
            prepare(Tokenizer(), decision, row | patch, 8)
    with pytest.raises(ValueError, match="Only choice and noul"):
        prepare(Tokenizer(), decision, {"question": {"type": "score"}}, 8)


def test_config_rejects_causal_and_fp16_changes(decision):
    validate_decision_config(decision)
    for patch in (
        {"attention_mode": "causal"},
        {"readout_dtype": "float16"},
        {"codes": ["A"] * 255},
        {"temperature": float("nan")},
    ):
        with pytest.raises(ValueError):
            validate_decision_config(decision | patch)


def test_answer_uses_only_declared_options(decision, row):
    p = prepare(Tokenizer(), decision, row, 8)
    logits = torch.zeros(1, 255)
    logits[0, 254] = 1000  # unused answer codes must not affect probabilities
    assert answer(logits, p, decision)["probabilities"] == {"report": 1.0}
    logits[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        answer(logits, p, decision)


def test_unknown_layer_type_rejected(tiny):
    config = copy.copy(tiny.language_model.config)
    config.layer_types = ["unsupported"] * 4
    language = copy.copy(tiny.language_model)
    language.config = config
    with pytest.raises(ValueError, match="Unsupported text layer"):
        TextDecision(language, tiny.readout)
