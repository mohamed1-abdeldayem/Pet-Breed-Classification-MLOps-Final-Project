
import numpy as np
import pytest

from pet_breed_mlops.bento_service import PetBreed as PetBreedService

PetBreed = PetBreedService.inner

@pytest.fixture
def service():
    """Create the service without running its real model-loading initializer."""
    obj = PetBreed.__new__(PetBreed)
    obj.temperature = 1.0
    obj.threshold = 0.7
    obj.model_version = "test"
    obj.classes = ["Abyssinian", "Bengal", "Egyptian Mau"] + [
        f"class_{i}" for i in range(34)
    ]
    obj.transform = None

    class FakeRunner:
        def logits(self, x):
            assert x.shape == (1, 3, 224, 224)
            logits = np.full((1, 37), -5.0, dtype=np.float32)
            logits[0, :3] = [5.0, 2.0, 1.0]
            return logits

    obj.runner = FakeRunner()
    return obj


def test_health_contract(service):
    result = service.health()

    assert result == {
        "status": "healthy",
        "model_version": "test",
        "num_classes": 37,
    }


def test_predict_returns_expected_schema(service, monkeypatch):
    import pet_breed_mlops.bento_service as module

    monkeypatch.setattr(
        module,
        "preprocess",
        lambda image, transform: __import__("torch").zeros(3, 224, 224),
    )

    result = service.predict(object())

    assert set(result) == {
        "breed",
        "species",
        "confidence",
        "top_3",
        "decision",
        "model_version",
    }
    assert result["breed"] == "Abyssinian"
    assert result["species"] == "cat"
    assert result["model_version"] == "test"
    assert len(result["top_3"]) == 3
    assert all(
        set(item) == {"breed", "probability"} for item in result["top_3"]
    )


def test_probabilities_are_valid_and_sorted(service, monkeypatch):
    import pet_breed_mlops.bento_service as module

    monkeypatch.setattr(
        module,
        "preprocess",
        lambda image, transform: __import__("torch").zeros(3, 224, 224),
    )

    result = service.predict(object())
    probabilities = [item["probability"] for item in result["top_3"]]

    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1.0 + 1e-6


def test_temperature_is_applied(service, monkeypatch):
    import pet_breed_mlops.bento_service as module

    monkeypatch.setattr(
        module,
        "preprocess",
        lambda image, transform: __import__("torch").zeros(3, 224, 224),
    )

    service.temperature = 2.0
    result = service.predict(object())

    assert result["confidence"] < 1.0
    assert result["top_3"][0]["probability"] == pytest.approx(
        result["confidence"]
    )