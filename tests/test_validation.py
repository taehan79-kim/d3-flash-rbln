import pytest
import torch

from d3_rbln.validation import compare_logits


def test_equal_logits_pass():
    a = torch.randn(1, 255)
    assert compare_logits(a, a, options=3)["passed"]


def test_wrong_decision_fails_even_when_loose_logit_tolerance():
    a, b = torch.zeros(1, 255), torch.zeros(1, 255)
    a[0, 0], b[0, 1] = 0.001, 0.001
    result = compare_logits(a, b, options=2, atol=1, probability_atol=1)
    assert result["logits_close"] and not result["decision_agreement"]
    assert not result["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_fail(value):
    a = torch.zeros(1, 255)
    a[0, 0] = value
    assert not compare_logits(a, torch.zeros(1, 255), options=2)["passed"]


@pytest.mark.parametrize("kwargs", [{"options": 0}, {"atol": -1}, {"rtol": float("nan")}])
def test_invalid_thresholds_rejected(kwargs):
    a = torch.zeros(1, 255)
    with pytest.raises(ValueError):
        compare_logits(a, a, **({"options": 2} | kwargs))


def test_unselected_logit_error_is_still_reported():
    a, b = torch.zeros(1, 255), torch.zeros(1, 255)
    a[0, 254] = 100
    result = compare_logits(a, b, options=2)
    assert result["decision_agreement"]
    assert result["max_probability_error"] == 0
    assert not result["passed"]
