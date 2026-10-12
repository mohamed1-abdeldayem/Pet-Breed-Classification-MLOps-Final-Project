import pytest

from pet_breed_mlops.monitoring import calculate_psi


def test_psi_is_zero_for_same_distribution() -> None:
    baseline = {"cat": 50, "dog": 50}
    current = ["cat"] * 50 + ["dog"] * 50

    assert calculate_psi(baseline, current) == pytest.approx(0.0)


def test_psi_increases_when_distribution_shifts() -> None:
    baseline = {"cat": 50, "dog": 50}
    current = ["cat"] * 80 + ["dog"] * 20

    assert calculate_psi(baseline, current) > 0.0


def test_psi_rejects_empty_current_data() -> None:
    with pytest.raises(ValueError, match="current observations"):
        calculate_psi({"cat": 10}, [])

def test_psi_rejects_empty_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        calculate_psi({}, ["cat"])


def test_psi_rejects_negative_baseline_counts() -> None:
    with pytest.raises(ValueError, match="negative"):
        calculate_psi({"cat": 10, "dog": -1}, ["cat"])


def test_psi_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        calculate_psi({"cat": 10}, ["cat"], epsilon=0)

def test_psi_handles_new_category() -> None:
    baseline = {"cat": 50, "dog": 50}
    current = ["cat"] * 50 + ["dog"] * 40 + ["bird"] * 10

    score = calculate_psi(baseline, current)

    assert score > 0.0
    assert score < float("inf")


def test_psi_is_finite_for_missing_category() -> None:
    baseline = {"cat": 50, "dog": 50}
    current = ["cat"] * 100

    score = calculate_psi(baseline, current)

    assert score >= 0.0
    assert score < float("inf")