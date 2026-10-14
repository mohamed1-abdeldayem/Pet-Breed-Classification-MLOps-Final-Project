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


def test_prometheus_metrics_update_after_predictions(client) -> None:
    import re

    def metric_value(metrics: str, name: str) -> float:
        match = re.search(
            rf"^{re.escape(name)}\s+([0-9.eE+-]+)$",
            metrics,
            re.MULTILINE,
        )
        return float(match.group(1)) if match else 0.0

    before = client.get("/metrics")
    assert before.status_code == 200
    before_text = before.text

    before_success = metric_value(
        before_text,
        'pet_breed_prediction_requests_total{status="success"}',
    )
    before_failure = metric_value(
        before_text,
        'pet_breed_prediction_requests_total{status="failure"}',
    )
    before_duration_count = metric_value(
        before_text,
        "pet_breed_prediction_duration_seconds_count",
    )
    before_confidence_count = metric_value(
        before_text,
        "pet_breed_prediction_confidence_count",
    )

    success_response = _post(client, _img_bytes("RGB", "JPEG"), "x.jpg", "image/jpeg")
    assert success_response.status_code == 200

    failure_response = _post(client, b"not an image", "x.txt", "text/plain")
    assert failure_response.status_code == 422

    after = client.get("/metrics")
    assert after.status_code == 200
    after_text = after.text

    assert metric_value(
        after_text,
        'pet_breed_prediction_requests_total{status="success"}',
    ) == before_success + 1

    assert metric_value(
        after_text,
        'pet_breed_prediction_requests_total{status="failure"}',
    ) == before_failure + 1

    assert metric_value(
        after_text,
        "pet_breed_prediction_duration_seconds_count",
    ) == before_duration_count + 2

    assert metric_value(
        after_text,
        "pet_breed_prediction_confidence_count",
    ) == before_confidence_count + 1

def test_drift_metrics_update_after_full_window(client) -> None:
    import re

    def metric_value(metrics: str, name: str) -> float:
        match = re.search(
            rf"^{re.escape(name)}\s+([0-9.eE+-]+)$",
            metrics,
            re.MULTILINE,
        )
        return float(match.group(1)) if match else 0.0

    for _ in range(100):
        response = _post(
            client,
            _img_bytes("RGB", "JPEG", size=(64, 64)),
            "drift-test.jpg",
            "image/jpeg",
        )
        assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200

    assert metric_value(
        metrics.text,
        "pet_breed_drift_window_ready",
    ) == 1.0

    drift_score = metric_value(
        metrics.text,
        "pet_breed_drift_score",
    )
    assert drift_score > 0.0

def test_prometheus_failure_metric_updates_on_prediction_exception(
    model_dir, monkeypatch
) -> None:
    import re

    app = create_app(Settings(model_dir=model_dir, max_upload_bytes=100_000))

    with TestClient(app, raise_server_exceptions=False) as client:
        before = client.get("/metrics")
        assert before.status_code == 200

        def metric_value(metrics: str, name: str) -> float:
            match = re.search(
                rf"^{re.escape(name)}\s+([0-9.eE+-]+)$",
                metrics,
                re.MULTILINE,
            )
            return float(match.group(1)) if match else 0.0

        before_failure = metric_value(
            before.text,
            'pet_breed_prediction_requests_total{status="failure"}',
        )

        def raise_unexpected_error(self, image):
            raise RuntimeError("simulated prediction failure")

        monkeypatch.setattr(
            "pet_breed_mlops.main.PetBreedClassifier.predict",
            raise_unexpected_error,
        )

        response = _post(
            client,
            _img_bytes("RGB", "JPEG"),
            "x.jpg",
            "image/jpeg",
        )
        assert response.status_code == 500

        after = client.get("/metrics")
        assert after.status_code == 200

        after_failure = metric_value(
            after.text,
            'pet_breed_prediction_requests_total{status="failure"}',
        )
        assert after_failure == before_failure + 1
