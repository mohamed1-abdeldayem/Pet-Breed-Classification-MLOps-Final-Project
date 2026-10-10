from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pet_breed_mlops.inference import PetBreedClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score Oxford-IIIT Pet images.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/oxford-iiit-pet/images"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/resnet50"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/batch_scoring"))
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    files = sorted(
        p for p in args.input.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(files) < args.limit:
        raise SystemExit(
            f"Need at least {args.limit} images, but found {len(files)} in {args.input}"
        )

    files = files[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.json"

    # Load weights and preprocessing exactly once.
    classifier = PetBreedClassifier(args.model_dir, device="cpu")
    counts: Counter[str] = Counter()
    failed = 0
    started = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "breed",
                "species",
                "confidence",
                "decision",
                "model_version",
                "error",
            ],
        )
        writer.writeheader()

        for index, path in enumerate(files, start=1):
            try:
                with Image.open(path) as image:
                    image.load()
                    prediction = classifier.predict(image.convert("RGB"))

                writer.writerow(
                    {
                        "image": str(path),
                        "breed": prediction.breed,
                        "species": prediction.species,
                        "confidence": prediction.confidence,
                        "decision": prediction.decision,
                        "model_version": prediction.model_version,
                        "error": "",
                    }
                )
                counts[prediction.decision] += 1
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                failed += 1
                writer.writerow(
                    {
                        "image": str(path),
                        "breed": "",
                        "species": "",
                        "confidence": "",
                        "decision": "error",
                        "model_version": classifier.model_version,
                        "error": str(exc)[:300],
                    }
                )

            if index % 100 == 0 or index == len(files):
                print(f"Processed {index}/{len(files)} images")

    elapsed = time.perf_counter() - started
    successful = len(files) - failed
    summary = {
        "model_version": classifier.model_version,
            "backbone": args.model_dir.name,
        "input_dir": str(args.input),
        "requested_images": len(files),
        "successful_predictions": successful,
        "failed_images": failed,
        "decision_counts": dict(counts),
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(successful / elapsed, 3) if elapsed else None,
        "predictions_csv": str(csv_path),
        "note": "Operational batch-scoring report; not a test-set accuracy evaluation.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

