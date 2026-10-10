from __future__ import annotations

from pathlib import Path

import bentoml
import numpy as np
import torch
from PIL import Image

from pet_breed_mlops.inference import CHECKPOINT_NAME, TRANSFORM_NAME
from pet_breed_mlops.labels import load_classes, species_of
from pet_breed_mlops.preprocessing import (
    EvalTransformConfig,
    build_eval_transform,
    preprocess,
)

MODEL_REF = bentoml.models.BentoModel("pet_breed_onnx:latest")


@bentoml.service(workers=1)
class OnnxRunner:
    """ONNX Runtime CPU. batchable=True: BentoML merges concurrent requests."""

    model_ref = MODEL_REF

    def __init__(self) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(
            str(Path(self.model_ref.path) / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=30)
    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(["logits"], {"image": x.astype(np.float32)})[0]


@bentoml.service(workers=1)
class PetBreed:
    runner = bentoml.depends(OnnxRunner)
    model_ref = MODEL_REF

    def __init__(self) -> None:
        d = Path(self.model_ref.path)
        ckpt = torch.load(d / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
        self.temperature: float = ckpt["temperature"]
        self.threshold: float = ckpt["threshold"]
        self.model_version: str = ckpt["model_version"]
        del ckpt  # the weights live in the ONNX file, free the memory
        self.classes = load_classes()
        self.transform = build_eval_transform(EvalTransformConfig.load(d / TRANSFORM_NAME))


    @bentoml.api
    def predict(self, image: Image.Image) -> dict:
        x = preprocess(image, self.transform).unsqueeze(0).numpy()
        z = self.runner.logits(x)[0] / self.temperature
        e = np.exp(z - z.max())
        p = e / e.sum()
        top = np.argsort(p)[::-1][:3]
        breed = self.classes[int(top[0])]
        conf = float(p[top[0]])

        return {
            "breed": breed,
            "species": species_of(breed),
            "confidence": conf,
            "top_3": [
                {
                    "breed": self.classes[int(i)],
                    "probability": float(p[i]),
                }
                for i in top
            ],
            "decision": "confident" if conf >= self.threshold else "uncertain",
            "model_version": self.model_version,
        }

    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy", "model_version": self.model_version,
                "num_classes": len(self.classes)}