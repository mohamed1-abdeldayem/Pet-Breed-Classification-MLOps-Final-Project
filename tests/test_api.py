import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pet_breed_mlops.config import Settings
from pet_breed_mlops.main import create_app


def _img_bytes(mode: str, fmt: str = "PNG", size=(300, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format=fmt)
    return buf.getvalue()

@pytest.fixture(scope="module")
def client(model_dir):
    app = create_app(Settings(model_dir=model_dir, max_upload_bytes=100_000))
    with TestClient(app) as c:
        yield c


def _post(client, data: bytes, name="x.png", ctype="image/png"):
    return client.post("/predict", files={"file": (name, data, ctype)})


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_version": "test", "num_classes": 37}


def test_predict_jpeg_returns_full_schema(client) -> None:
    r = _post(client, _img_bytes("RGB", "JPEG"), "x.jpg", "image/jpeg")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"breed", "species", "confidence", "top_3", "decision", "model_version"}
    assert len(body["top_3"]) == 3
    assert body["decision"] in {"confident", "uncertain"}


@pytest.mark.parametrize(
    "mode, fmt",
    [
        ("RGBA", "PNG"),
        ("L", "PNG"),
        ("P", "PNG"),
        ("CMYK", "JPEG"),
        ("CMYK", "TIFF"),
    ],
)
def test_unusual_channel_modes_return_200(client, mode, fmt) -> None:
    data = _img_bytes(mode, fmt, size=(64, 64))
    assert _post(client, data).status_code == 200

def test_non_image_is_422(client) -> None:
    assert _post(client, b"this is not an image", "x.txt", "text/plain").status_code == 422


def test_empty_file_is_422(client) -> None:
    assert _post(client, b"").status_code == 422


def test_oversized_upload_is_422(client) -> None:
    r = _post(client, b"0" * 200_000)
    assert r.status_code == 422
    assert "too large" in r.json()["detail"]


def test_missing_file_field_is_422(client) -> None:
    assert client.post("/predict").status_code == 422