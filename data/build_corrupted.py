from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from PIL import Image

from pet_breed_mlops.corruptions import CORRUPTIONS, SEVERITIES, apply

logger = logging.getLogger(__name__)


class CorruptedSetBuilder:
    """Write corrupted copies of a fixed test subset + a manifest describing each one."""

    def __init__(
        self,
        manifest: str | Path = "data/manifest.json",
        out_dir: str | Path = "data/corrupted",
        out_manifest: str | Path = "data/manifest_corrupted.json",
        subset_file: str | Path = "data/splits/drift_subset.txt",
        n_images: int | None = 300,
        seed: int = 42,
    ) -> None:
        self.manifest = Path(manifest)
        self.out_dir = Path(out_dir)
        self.out_manifest = Path(out_manifest)
        self.subset_file = Path(subset_file)
        self.n_images = n_images
        self.seed = seed

    def _subset(self) -> list[dict]:
        test = sorted(
            (r for r in json.loads(self.manifest.read_text()) if r["split"] == "test"),
            key=lambda r: r["image_id"],
        )
        if self.n_images is not None:
            test = random.Random(self.seed).sample(test, min(self.n_images, len(test)))
            test.sort(key=lambda r: r["image_id"])
        return test

    def build(self) -> list[dict]:
        subset = self._subset()
        self.subset_file.parent.mkdir(parents=True, exist_ok=True)
        self.subset_file.write_text("\n".join(r["image_id"] for r in subset) + "\n")

        records: list[dict] = []
        for name in CORRUPTIONS:
            for severity in SEVERITIES:
                folder = self.out_dir / name / f"s{severity}"
                folder.mkdir(parents=True, exist_ok=True)
                for r in subset:
                    with Image.open(r["path"]) as src:
                        out = apply(src.convert("RGB"), name, severity)
                    path = folder / f"{r['image_id']}.jpg"
                    out.save(path, format="JPEG", quality=95)
                    records.append(
                        {**r, "path": path.as_posix(), "corruption": name, "severity": severity}
                    )
                logger.info("%s severity %d: %d images", name, severity, len(subset))

        self.out_manifest.write_text(json.dumps(records, indent=1), encoding="utf-8")
        logger.info("Wrote %d records to %s", len(records), self.out_manifest)
        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    CorruptedSetBuilder().build()