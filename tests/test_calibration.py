import torch

from pet_breed_mlops.calibration import (
    abstention_stats,
    expected_calibration_error,
    fit_temperature,
)


def _overconfident(n=2000, k=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, k, (n,), generator=g)
    logits = torch.randn(n, k, generator=g)
    right = torch.rand(n, generator=g) < 0.6          # model is right only ~60% of the time
    pred = torch.where(right, labels, torch.randint(0, k, (n,), generator=g))
    logits[torch.arange(n), pred] += 12.0              # ...but always very confident
    return logits, labels


def test_temperature_reduces_overconfidence_and_ece() -> None:
    logits, labels = _overconfident()
    t = fit_temperature(logits, labels)
    assert t > 1.5
    assert expected_calibration_error(logits, labels, t) < expected_calibration_error(logits, labels)


def test_temperature_never_changes_predictions() -> None:
    logits, labels = _overconfident()
    t = fit_temperature(logits, labels)
    assert torch.equal(logits.argmax(1), (logits / t).argmax(1))


def test_higher_threshold_lowers_coverage() -> None:
    logits, labels = _overconfident()
    t = fit_temperature(logits, labels)
    lo = abstention_stats(logits, labels, t, 0.2)
    hi = abstention_stats(logits, labels, t, 0.8)
    assert hi.coverage <= lo.coverage