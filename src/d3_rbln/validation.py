"""Numerical gates, separate from compilation and application accuracy."""

from __future__ import annotations

import math

import torch


def compare_logits(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    options: int,
    temperature: float = 1.0,
    atol: float = 0.1,
    rtol: float = 0.01,
    probability_atol: float = 0.02,
) -> dict:
    if not 1 <= options <= 255:
        raise ValueError("options must be in [1,255]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    for tolerance in (atol, rtol, probability_atol):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerances must be finite and non-negative")
    if actual.shape != (1, 255) or reference.shape != actual.shape:
        raise ValueError("Expected actual and reference shapes [1,255]")
    actual, reference = actual.detach().float().cpu(), reference.detach().float().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"passed": False, "finite": False, "decision_agreement": False}
    difference = (actual - reference).abs()
    a = torch.softmax(actual[:, :options] / temperature, dim=-1)
    b = torch.softmax(reference[:, :options] / temperature, dim=-1)
    agrees = int(a.argmax()) == int(b.argmax())
    logits_close = bool(torch.allclose(actual, reference, atol=atol, rtol=rtol))
    probability_error = float((a - b).abs().max())
    return {
        "passed": agrees and logits_close and probability_error <= probability_atol,
        "finite": True,
        "decision_agreement": agrees,
        "logits_close": logits_close,
        "max_logit_error": float(difference.max()),
        "mean_logit_error": float(difference.mean()),
        "max_probability_error": probability_error,
        "actual_choice_index": int(a.argmax()),
        "reference_choice_index": int(b.argmax()),
    }
