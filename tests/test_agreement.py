import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from pet_breed_mlops.backbones import build_backbone
from pet_breed_mlops.config import Settings
from pet_breed_mlops.inference import CHECKPOINT_NAME, TRANSFORM_NAME, PetBreedClassifier
from pet_breed_mlops.labels import load_classes
from pet_breed_mlops.main import create_app
from pet_breed_mlops.preprocessing import EvalTransformConfig, build_eval_transform

ATOL = 1e-4


@pytest.fixture(scope="module")
def png_bytes() -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(400, 300, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def reference(model_dir):
    """The 'training path': load weights + transform directly, no PetBreedClassifier."""
    ckpt = torch.load(model_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    model = build_backbone(ckpt["backbone"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    transform = build_eval_transform(EvalTransformConfig.load(model_dir / TRANSFORM_NAME))
    return model, transform, ckpt["temperature"]


def _reference_logits(reference, png_bytes: bytes) -> torch.Tensor:
    model, transform, _ = reference
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    with torch.inference_mode():
        return model(transform(image).unsqueeze(0))[0]


def test_classifier_logits_match_training_path(model_dir, reference, png_bytes) -> None:
    from pet_breed_mlops.preprocessing import load_image

    clf = PetBreedClassifier(model_dir)
    api_logits = clf.logits(load_image(png_bytes))
    train_logits = _reference_logits(reference, png_bytes)
    assert torch.allclose(api_logits, train_logits, atol=ATOL)


def test_http_response_matches_training_path(model_dir, reference, png_bytes) -> None:
    app = create_app(Settings(model_dir=model_dir))
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.json()
    body = r.json()

    _, _, temperature = reference
    probs = torch.softmax(_reference_logits(reference, png_bytes) / temperature, dim=0)
    top_p, top_i = torch.topk(probs, k=3)
    classes = load_classes()

    assert [t["breed"] for t in body["top_3"]] == [classes[i] for i in top_i.tolist()]
    for got, want in zip(body["top_3"], top_p.tolist()):
        assert got["probability"] == pytest.approx(want, abs=ATOL)


def test_transform_matches_handwritten_imagenet_normalization(png_bytes) -> None:
    """Guards the constants themselves: resize 256 -> center-crop 224 -> ImageNet mean/std."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    got = build_eval_transform()(image)

    w, h = image.size  # 300 x 400, shorter side -> 256
    scale = 256 / min(w, h)
    resized = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    crop = resized.crop((left, top, left + 224, top + 224))

    x = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    want = (x - mean) / std

    assert got.shape == want.shape
    assert got.mean().item() == pytest.approx(want.mean().item(), abs=1e-2)
    assert got.std().item() == pytest.approx(want.std().item(), abs=1e-2)