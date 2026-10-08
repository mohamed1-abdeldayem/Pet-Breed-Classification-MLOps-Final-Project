import numpy as np
import pytest
from PIL import Image

from pet_breed_mlops.corruptions import CORRUPTIONS, SEVERITIES, apply


@pytest.fixture(scope="module")
def img() -> Image.Image:
    # gradient + noise: يُظهر أثر الـ blur والـ compression، عكس لون واحد
    rng = np.random.default_rng(0)
    x = np.linspace(0, 255, 200, dtype=np.float32)[None, :, None] * np.ones((150, 1, 3))
    arr = np.clip(x + rng.normal(0, 25, x.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _diff(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


@pytest.mark.parametrize("name", list(CORRUPTIONS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_keeps_size_and_rgb(img, name, severity) -> None:
    out = apply(img, name, severity)
    assert out.size == img.size
    assert out.mode == "RGB"


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_severity_is_monotonic(img, name) -> None:
    diffs = [_diff(img, apply(img, name, s)) for s in SEVERITIES]
    assert diffs[0] < diffs[1] < diffs[2], diffs


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_is_deterministic(img, name) -> None:
    assert _diff(apply(img, name, 2), apply(img, name, 2)) == 0.0


def test_brightness_directions(img) -> None:
    base = np.asarray(img, np.float32).mean()
    assert np.asarray(apply(img, "brightness_up", 3), np.float32).mean() > base
    assert np.asarray(apply(img, "brightness_down", 3), np.float32).mean() < base


def test_unknown_corruption_and_severity_rejected(img) -> None:
    with pytest.raises(ValueError):
        apply(img, "nope", 1)
    with pytest.raises(ValueError):
        apply(img, "jpeg", 4)