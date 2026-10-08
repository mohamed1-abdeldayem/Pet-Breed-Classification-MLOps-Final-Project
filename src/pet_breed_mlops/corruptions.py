from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SEVERITIES = (1, 2, 3)


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def low_res(img: Image.Image, side: int) -> Image.Image:
    """Downscale to side x side, then upscale back (old handset)."""
    return img.resize((side, side), Image.BILINEAR).resize(img.size, Image.BILINEAR)


def motion_blur(img: Image.Image, length: int) -> Image.Image:
    """Horizontal motion blur: average `length` shifted copies."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    pad = length // 2
    p = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = sum(p[:, i : i + a.shape[1]] for i in range(length)) / length
    return Image.fromarray(out.round().astype(np.uint8))


# name -> (function, parameter for severity 1, 2, 3). Severity 3 is the worst.
CORRUPTIONS: dict[str, tuple[Callable[[Image.Image, float], Image.Image], tuple]] = {
    "gaussian_blur": (gaussian_blur, (1.0, 2.0, 4.0)),          # radius in px
    "brightness_up": (brightness, (1.3, 1.6, 2.0)),             # direct sun
    "brightness_down": (brightness, (0.7, 0.5, 0.3)),           # indoor evening
    "jpeg": (jpeg_compress, (60, 30, 10)),                      # messaging re-encode
    "low_res": (low_res, (160, 96, 64)),                        # old handset
    "motion_blur": (motion_blur, (5, 11, 21)),                  # kernel length in px
}


def apply(img: Image.Image, name: str, severity: int) -> Image.Image:
    """Apply one corruption at severity 1-3. Returns an RGB image of the same size."""
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{name}'. Supported: {list(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be one of {SEVERITIES}, got {severity}")
    fn, params = CORRUPTIONS[name]
    return fn(img.convert("RGB"), params[severity - 1]).convert("RGB")