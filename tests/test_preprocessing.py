import io

import pytest
import torch
from PIL import Image

from pet_breed_mlops.preprocessing import (
    EvalTransformConfig,
    build_eval_transform,
    load_image,
    preprocess,
)


def _png_bytes(mode: str) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (300, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_output_shape_and_dtype() -> None:
    img = Image.new("RGB", (394, 500), color=(120, 80, 40))
    out = preprocess(img, build_eval_transform())
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_is_deterministic() -> None:
    img = Image.new("RGB", (320, 240), color=(10, 200, 90))
    t = build_eval_transform()
    assert torch.equal(preprocess(img, t), preprocess(img, t))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_all_channel_modes_load_as_rgb(mode: str) -> None:
    img = load_image(_png_bytes(mode))
    assert img.mode == "RGB"


def test_non_image_raises_value_error() -> None:
    with pytest.raises(ValueError):
        load_image(b"this is not an image")


def test_config_roundtrip(tmp_path) -> None:
    cfg = EvalTransformConfig()
    cfg.save(tmp_path / "t.json")
    assert EvalTransformConfig.load(tmp_path / "t.json") == cfg