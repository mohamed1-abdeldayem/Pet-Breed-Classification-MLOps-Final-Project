import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from pet_breed_mlops.labels import load_classes, species_of

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="run data/build_manifest.py first (or dvc pull)"
)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_every_path_exists_and_opens(records) -> None:
    for r in records:
        with Image.open(ROOT / r["path"]) as img:
            img.verify()


def test_no_image_in_two_splits(records) -> None:
    ids = [r["image_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_split_sizes(records) -> None:
    counts = Counter(r["split"] for r in records)
    assert counts["train"] + counts["val"] == 3680
    assert counts["test"] == 3669
    assert len(records) == 7349


def test_labels_match_label_map(records) -> None:
    classes = load_classes()
    assert len(classes) == 37
    assert {r["class_index"] for r in records} == set(range(37))
    assert all(classes[r["class_index"]] == r["breed"] for r in records)


def test_species_matches_label_map(records) -> None:
    assert all(species_of(r["breed"]) == r["species"] for r in records)


def test_every_class_has_at_least_50_train_images(records) -> None:
    per_class = Counter(r["class_index"] for r in records if r["split"] == "train")
    assert len(per_class) == 37
    assert min(per_class.values()) >= 50


def test_every_class_present_in_val_and_test(records) -> None:
    for split in ("val", "test"):
        assert {r["class_index"] for r in records if r["split"] == split} == set(range(37))


def test_no_image_under_32x32(records) -> None:
    assert min(min(r["width"], r["height"]) for r in records) >= 32


def test_clean_records_have_no_corruption(records) -> None:
    assert all(r["corruption"] is None and r["severity"] == 0 for r in records)