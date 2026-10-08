import pytest
import torch
from PIL import Image

from pet_breed_mlops.inference import PetBreedClassifier
from pet_breed_mlops.labels import CAT_BREEDS, load_classes, species_of


@pytest.fixture(scope="module")
def clf(model_dir):
    return PetBreedClassifier(model_dir)


@pytest.fixture
def img():
    return Image.new("RGB", (394, 500), (120, 80, 40))


def test_cat_breeds_all_exist_in_label_map() -> None:
    assert CAT_BREEDS <= set(load_classes())
    assert sum(species_of(c) == "cat" for c in load_classes()) == 12


def test_prediction_structure(clf, img) -> None:
    p = clf.predict(img)
    assert p.breed in clf.classes
    assert p.species in {"cat", "dog"}
    assert len(p.top_3) == 3
    assert p.top_3[0][0] == p.breed
    assert 0.0 <= p.confidence <= 1.0
    assert p.model_version == "test"


def test_top3_probabilities_sorted_and_bounded(clf, img) -> None:
    probs = [p for _, p in clf.predict(img).top_3]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0 + 1e-6


def test_same_image_same_answer(clf, img) -> None:
    assert torch.equal(clf.logits(img), clf.logits(img))


def test_abstains_below_threshold(model_dir, img) -> None:
    strict = PetBreedClassifier(model_dir, threshold=1.01)
    assert strict.predict(img).decision == "uncertain"
    lax = PetBreedClassifier(model_dir, threshold=0.0)
    assert lax.predict(img).decision == "confident"


def test_wrong_label_map_size_is_rejected(model_dir) -> None:
    with pytest.raises(ValueError):
        PetBreedClassifier(model_dir, classes=["a", "b"])