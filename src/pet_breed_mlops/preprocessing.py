from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class EvalTransformConfig:
    """Everything needed to rebuild the eval transform. Saved next to the weights."""

    resize: int = 256
    crop: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvalTransformConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["mean"] = tuple(data["mean"])
        data["std"] = tuple(data["std"])
        return cls(**data)


def build_eval_transform(config: EvalTransformConfig | None = None) -> transforms.Compose:
    """Deterministic eval transform. No augmentation here, ever."""
    cfg = config or EvalTransformConfig()
    return transforms.Compose(
        [
            transforms.Resize(cfg.resize),
            transforms.CenterCrop(cfg.crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def to_rgb(image: Image.Image) -> Image.Image:
    """Handle CMYK, greyscale, palette and RGBA (4-channel PNG) uniformly."""
    return image.convert("RGB")


def load_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB PIL image. Raises ValueError if not an image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # PIL raises several different error types
        raise ValueError("Uploaded file is not a valid image") from exc
    return to_rgb(image)


def preprocess(image: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    """PIL image -> normalized tensor of shape (3, crop, crop)."""
    return transform(to_rgb(image))