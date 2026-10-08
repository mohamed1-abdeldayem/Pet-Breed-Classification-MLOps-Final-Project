from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from pet_breed_mlops.labels import load_classes

logger = logging.getLogger(__name__)


class ManifestBuilder:
    """Build the per-image manifest and the train/val/test index files."""

    def __init__(
        self,
        root: str | Path = "data/raw/oxford-iiit-pet",
        output: str | Path = "data/manifest.json",
        splits_dir: str | Path = "data/splits",
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.output = Path(output)
        self.splits_dir = Path(splits_dir)
        self.val_fraction = val_fraction
        self.seed = seed
        self.classes = load_classes()

    def _read_annotations(self, name: str) -> list[tuple[str, int, str]]:
        """Rows of (image_id, class_index 0-36, species) from the official split file."""
        rows = []
        for line in (self.root / "annotations" / f"{name}.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            image_id, class_id, species_id, _ = line.split()
            rows.append((image_id, int(class_id) - 1, "cat" if species_id == "1" else "dog"))
        return rows

    def _val_ids(self, trainval: list[tuple[str, int, str]]) -> set[str]:
        """Stratified, seeded validation fold carved out of trainval."""
        by_class: dict[int, list[str]] = defaultdict(list)
        for image_id, class_index, _ in trainval:
            by_class[class_index].append(image_id)

        rng = random.Random(self.seed)
        val: set[str] = set()
        for class_index in sorted(by_class):
            ids = sorted(by_class[class_index])  # sorted first, so the shuffle is reproducible
            rng.shuffle(ids)
            n_val = max(1, round(len(ids) * self.val_fraction))
            val.update(ids[:n_val])
        return val

    def _check_label(self, image_id: str, class_index: int) -> None:
        """The breed in the filename must match the committed label map at that index."""
        raw = image_id.rsplit("_", 1)[0]
        from_name = " ".join(part.title() for part in raw.split("_"))
        if from_name != self.classes[class_index]:
            raise ValueError(
                f"{image_id}: filename says '{from_name}' but label map[{class_index}] "
                f"is '{self.classes[class_index]}'"
            )

    def _record(self, image_id: str, class_index: int, species: str, split: str) -> dict:
        self._check_label(image_id, class_index)
        path = self.root / "images" / f"{image_id}.jpg"
        with Image.open(path) as img:
            width, height = img.size
        return {
            "image_id": image_id,
            "path": path.as_posix(),
            "breed": self.classes[class_index],
            "species": species,
            "class_index": class_index,
            "split": split,
            "corruption": None,
            "severity": 0,
            "width": width,
            "height": height,
        }

    def build(self) -> list[dict]:
        trainval = self._read_annotations("trainval")
        test = self._read_annotations("test")
        val_ids = self._val_ids(trainval)

        records = [
            self._record(i, c, s, "val" if i in val_ids else "train") for i, c, s in trainval
        ]
        records += [self._record(i, c, s, "test") for i, c, s in test]

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(records, indent=1), encoding="utf-8")

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            ids = sorted(r["image_id"] for r in records if r["split"] == split)
            (self.splits_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
            logger.info("%s: %d images", split, len(ids))

        logger.info("Wrote %d records to %s", len(records), self.output)
        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ManifestBuilder().build()