from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pet_breed_mlops.inference import PetBreedClassifier

VAL_FILE = Path("data/splits/val.txt")
MANIFEST_FILE = Path("data/manifest.json")
MODEL_DIR = Path("models/resnet50")
OUTPUT_FILE = Path("reports/monitoring/baseline.json")

CONFIDENCE_BINS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.01),
]


def main() -> None:
    val_ids = set(VAL_FILE.read_text(encoding="utf-8").splitlines())
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    records = {
        row["image_id"]: row
        for row in manifest
        if row.get("split") == "val" and row["image_id"] in val_ids
    }

    missing_ids = sorted(val_ids - records.keys())
    if missing_ids:
        raise SystemExit(
            f"Missing validation images in manifest: {len(missing_ids)}"
        )

    classifier = PetBreedClassifier(MODEL_DIR, device="cpu")

    predicted_breeds: Counter[str] = Counter()
    confidence_bins: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    confidence_sum = 0.0
    successful = 0

    for index, image_id in enumerate(sorted(val_ids), start=1):
        image_path = Path(records[image_id]["path"])

        try:
            with Image.open(image_path) as image:
                image.load()
                prediction = classifier.predict(image.convert("RGB"))

            predicted_breeds[prediction.breed] += 1
            decisions[prediction.decision] += 1
            confidence_sum += prediction.confidence
            successful += 1

            for lower, upper in CONFIDENCE_BINS:
                if lower <= prediction.confidence < upper:
                    label = f"{lower:.1f}-{min(upper, 1.0):.1f}"
                    confidence_bins[label] += 1
                    break

        except (OSError, ValueError, UnidentifiedImageError) as exc:
            errors.append(
                {
                    "image_id": image_id,
                    "error": str(exc)[:300],
                }
            )

        if index % 100 == 0 or index == len(val_ids):
            print(f"Processed {index}/{len(val_ids)} images")

    if successful == 0:
        raise SystemExit("No validation images were scored successfully.")

    baseline = {
        "model_version": classifier.model_version,
        "backbone": MODEL_DIR.name,
        "split": "val",
        "validation_images": len(val_ids),
        "successful_predictions": successful,
        "failed_predictions": len(errors),
        "mean_confidence": round(confidence_sum / successful, 6),
        "decision_counts": dict(decisions),
        "predicted_breed_distribution": dict(predicted_breeds),
        "confidence_distribution": dict(confidence_bins),
        "errors": errors,
        "note": (
            "Monitoring reference built from validation predictions only. "
            "Not a test-set evaluation or a measured drift score."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(baseline, indent=2),
        encoding="utf-8",
    )

    print(f"\nBaseline saved to: {OUTPUT_FILE}")
    print(json.dumps(
        {
            "validation_images": len(val_ids),
            "successful_predictions": successful,
            "failed_predictions": len(errors),
            "mean_confidence": baseline["mean_confidence"],
            "decision_counts": baseline["decision_counts"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
