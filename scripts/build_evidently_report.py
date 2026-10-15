from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from PIL import Image, UnidentifiedImageError

from pet_breed_mlops.corruptions import apply
from pet_breed_mlops.inference import PetBreedClassifier

VAL_FILE = Path("data/splits/val.txt")
MANIFEST_FILE = Path("data/manifest.json")
MODEL_DIR = Path("models/resnet50")
OUTPUT_DIR = Path("reports/monitoring")

CORRUPTION = "gaussian_blur"
SEVERITY = 2

DRIFT_COLUMNS = [
    "predicted_breed",
    "species",
    "confidence",
    "decision",
]


def score_images(
    classifier: PetBreedClassifier,
    records: list[dict],
    corrupted: bool,
) -> pd.DataFrame:
    rows: list[dict] = []

    for index, record in enumerate(records, start=1):
        image_path = Path(record["path"])

        try:
            with Image.open(image_path) as image:
                image.load()
                image = image.convert("RGB")

            if corrupted:
                image = apply(image, CORRUPTION, SEVERITY)

            prediction = classifier.predict(image)

            rows.append(
                {
                    "image_id": record["image_id"],
                    "true_breed": record["breed"],
                    "true_species": record["species"],
                    "predicted_breed": prediction.breed,
                    "species": prediction.species,
                    "confidence": prediction.confidence,
                    "decision": prediction.decision,
                    "model_version": prediction.model_version,
                    "corruption": CORRUPTION if corrupted else "none",
                    "severity": SEVERITY if corrupted else 0,
                }
            )
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Failed to score validation image {record['image_id']}: {exc}"
            ) from exc

        if index % 100 == 0 or index == len(records):
            print(
                f"{'Corrupted' if corrupted else 'Original'}: "
                f"{index}/{len(records)}"
            )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_ids = set(VAL_FILE.read_text(encoding="utf-8").splitlines())
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    records = sorted(
        (
            row
            for row in manifest
            if row.get("split") == "val" and row["image_id"] in val_ids
        ),
        key=lambda row: row["image_id"],
    )

    if len(records) != len(val_ids):
        raise SystemExit(
            f"Validation manifest mismatch: {len(records)} records "
            f"for {len(val_ids)} validation IDs."
        )

    if not records:
        raise SystemExit("No validation images found.")

    classifier = PetBreedClassifier(MODEL_DIR, device="cpu")

    reference = score_images(classifier, records, corrupted=False)
    current = score_images(classifier, records, corrupted=True)

    reference_path = OUTPUT_DIR / "evidently_reference.csv"
    current_path = OUTPUT_DIR / "evidently_current_gaussian_blur_s2.csv"
    report_path = OUTPUT_DIR / "evidently_gaussian_blur_s2.html"

    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    report = Report(
        metrics=[
            DataDriftPreset(
                columns=DRIFT_COLUMNS,
                include_tests=True,
            )
        ]
    )

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )
    snapshot.save_html(str(report_path))

    print("\nEvidently report generated.")
    print(f"Reference CSV: {reference_path}")
    print(f"Current CSV:   {current_path}")
    print(f"HTML report:   {report_path}")
    print(f"Rows per dataset: {len(reference)}")
    print(f"Compared columns: {', '.join(DRIFT_COLUMNS)}")


if __name__ == "__main__":
    main()
