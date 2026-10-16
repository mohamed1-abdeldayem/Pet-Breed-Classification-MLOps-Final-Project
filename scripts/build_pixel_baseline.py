from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from pet_breed_mlops.pixel_monitoring import calculate_pixel_statistics

VAL_FILE = Path("data/splits/val.txt")
MANIFEST_FILE = Path("data/manifest.json")
OUTPUT_FILE = Path("reports/monitoring/pixel_baseline.json")

BASELINE_FRACTION = 0.5


def main() -> None:
    val_ids = set(VAL_FILE.read_text(encoding="utf-8").splitlines())
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    records = {
        row["image_id"]: row
        for row in manifest
        if row.get("split") == "val" and row["image_id"] in val_ids
    }

    missing = sorted(val_ids - records.keys())
    if missing:
        raise SystemExit(
            f"Missing validation images in manifest: {len(missing)}"
        )


    ordered_records = [records[key] for key in sorted(records)]

    rng = np.random.default_rng(42)
    rng.shuffle(ordered_records)

    if len(ordered_records) < 4:
        raise SystemExit("Not enough validation images to split.")

    baseline_count = int(len(ordered_records) * BASELINE_FRACTION)
    baseline_records = ordered_records[:baseline_count]

    mean_rgb: list[list[float]] = []
    std_rgb: list[list[float]] = []
    mean_brightness: list[float] = []
    edge_strength: list[float] = []
    jpeg_blockiness: list[float] = []
    errors: list[dict[str, str]] = []

    for index, record in enumerate(baseline_records, start=1):
        try:
            with Image.open(record["path"]) as image:
                image.load()
                stats = calculate_pixel_statistics(image)

            mean_rgb.append(list(stats.mean_rgb))
            std_rgb.append(list(stats.std_rgb))
            mean_brightness.append(stats.mean_brightness)
            edge_strength.append(stats.edge_strength)
            jpeg_blockiness.append(stats.jpeg_blockiness)

        except (OSError, ValueError, UnidentifiedImageError) as exc:
            errors.append(
                {
                    "image_id": record["image_id"],
                    "error": str(exc)[:300],
                }
            )

        if index % 100 == 0 or index == len(baseline_records):
            print(
                f"Processed {index}/{len(baseline_records)} baseline images"
            )

    if errors:
        raise SystemExit(
            f"Failed to process {len(errors)} images. Baseline was not written."
        )

    baseline = {
        "split": "val",
        "images": len(mean_brightness),
        "baseline_image_ids": [
            record["image_id"] for record in baseline_records
        ],
        "pixel_value_range": [0.0, 1.0],
        "statistics": {
            "jpeg_blockiness": {
                "mean": float(np.mean(jpeg_blockiness)),
                "std": float(np.std(jpeg_blockiness)),
            },
            "edge_strength": {
                "mean": float(np.mean(edge_strength)),
                "std": float(np.std(edge_strength)),
            },
            "mean_rgb": {
                "mean": np.mean(mean_rgb, axis=0).tolist(),
                "std": np.std(mean_rgb, axis=0).tolist(),
            },
            "std_rgb": {
                "mean": np.mean(std_rgb, axis=0).tolist(),
                "std": np.std(std_rgb, axis=0).tolist(),
            },
            "mean_brightness": {
                "mean": float(np.mean(mean_brightness)),
                "std": float(np.std(mean_brightness)),
            },
        },
        "note": (
            "Pixel statistics baseline computed from a fixed subset of "
            "validation images only. Remaining validation images are reserved "
            "for calibration and evaluation."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(baseline, indent=2),
        encoding="utf-8",
    )

    print(f"\nPixel baseline saved to: {OUTPUT_FILE}")
    print(f"Baseline images: {len(mean_brightness)}")
    print(json.dumps(baseline["statistics"], indent=2))


if __name__ == "__main__":
    main()