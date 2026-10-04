from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .backbones import build_backbone
from .labels import load_classes, species_of
from .preprocessing import EvalTransformConfig, build_eval_transform, preprocess

CHECKPOINT_NAME = "model.pt"
TRANSFORM_NAME = "eval_transform.json"


@dataclass(frozen=True)
class Prediction:
    breed: str
    species: str
    confidence: float
    top_3: list[tuple[str, float]]
    decision: str  # "confident" | "uncertain"
    model_version: str


def save_checkpoint(
    model_dir: str | Path,
    model: torch.nn.Module,
    backbone: str,
    num_classes: int,
    temperature: float = 1.0,
    model_version: str = "v0",
    transform_config: EvalTransformConfig | None = None,
) -> None:
    """Weights + metadata + eval transform, saved together so they can never drift apart."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone,
            "num_classes": num_classes,
            "temperature": float(temperature),
            "model_version": model_version,
            "state_dict": model.state_dict(),
        },
        model_dir / CHECKPOINT_NAME,
    )
    (transform_config or EvalTransformConfig()).save(model_dir / TRANSFORM_NAME)


class PetBreedClassifier:
    """Loads weights, label map and eval transform once; predicts one PIL image at a time."""

    def __init__(
        self,
        model_dir: str | Path,
        classes: list[str] | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        model_dir = Path(model_dir)
        ckpt = torch.load(model_dir / CHECKPOINT_NAME, map_location=device, weights_only=True)

        self.classes = classes if classes is not None else load_classes()
        if len(self.classes) != ckpt["num_classes"]:
            raise ValueError(
                f"Label map has {len(self.classes)} classes, "
                f"checkpoint expects {ckpt['num_classes']}"
            )

        self.device = device
        self.threshold = threshold
        self.temperature: float = ckpt["temperature"]
        self.model_version: str = ckpt["model_version"]
        self.transform = build_eval_transform(EvalTransformConfig.load(model_dir / TRANSFORM_NAME))

        self.model = build_backbone(ckpt["backbone"], ckpt["num_classes"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @torch.inference_mode()
    def logits(self, image: Image.Image) -> torch.Tensor:
        """Raw logits, shape (num_classes,). Used by the train-vs-API agreement test."""
        x = preprocess(image, self.transform).unsqueeze(0).to(self.device)
        return self.model(x)[0]

    def predict(self, image: Image.Image) -> Prediction:
        probs = torch.softmax(self.logits(image) / self.temperature, dim=0)
        top_p, top_i = torch.topk(probs, k=3)
        top_3 = [(self.classes[i], p) for i, p in zip(top_i.tolist(), top_p.tolist())]
        breed, confidence = top_3[0]
        return Prediction(
            breed=breed,
            species=species_of(breed),
            confidence=confidence,
            top_3=top_3,
            decision="confident" if confidence >= self.threshold else "uncertain",
            model_version=self.model_version,
        )