from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from pet_breed_mlops.corruptions import CORRUPTIONS, SEVERITIES, apply
from pet_breed_mlops.pixel_monitoring import (
    PixelStatistics,
    calculate_pixel_statistics,
    compare_pixel_statistics,
)

VAL_FILE = Path("data/splits/val.txt")
MANIFEST_FILE = Path("data/manifest.json")
BASELINE_FILE = Path("reports/monitoring/pixel_baseline.json")
OUTPUT_DIR = Path("reports/monitoring")

BATCH_SIZE = 50
CALIBRATION_FRACTION = 0.5
RANDOM_SEED = 42


def load_validation_records() -> list[dict]:
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

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    baseline_ids = set(baseline.get("baseline_image_ids", []))

    if not baseline_ids:
        raise SystemExit(
            "Baseline has no baseline_image_ids. Rebuild it first."
        )

    if not baseline_ids.issubset(records):
        raise SystemExit(
            "Some baseline image IDs are missing from validation records."
        )

    remaining = [
        records[key]
        for key in sorted(records)
        if key not in baseline_ids
    ]

    if len(remaining) < 2 * BATCH_SIZE:
        raise SystemExit(
            "Not enough remaining validation images for calibration "
            "and evaluation."
        )

    return remaining


def score_images(
    records: list[dict],
    corruption: str | None = None,
    severity: int = 0,
) -> list[PixelStatistics]:
    statistics: list[PixelStatistics] = []

    for index, record in enumerate(records, start=1):
        try:
            with Image.open(record["path"]) as image:
                image.load()
                image = image.convert("RGB")

            if corruption is not None:
                image = apply(image, corruption, severity)

            statistics.append(calculate_pixel_statistics(image))

        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Failed on {record['image_id']}, "
                f"corruption={corruption}, severity={severity}: {exc}"
            ) from exc

        if index % 100 == 0 or index == len(records):
            condition = (
                "clean"
                if corruption is None
                else f"{corruption}, severity={severity}"
            )
            print(f"{condition}: {index}/{len(records)}")

    return statistics


def make_batches(
    values: list[PixelStatistics],
) -> list[list[PixelStatistics]]:
    shuffled = values.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(shuffled)

    full_count = len(shuffled) // BATCH_SIZE
    return [
        shuffled[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        for i in range(full_count)
    ]


def main() -> None:
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    records = load_validation_records()

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_records = records.copy()
    rng.shuffle(shuffled_records)

    calibration_count = int(
        len(shuffled_records) * CALIBRATION_FRACTION
    )
    calibration_records = shuffled_records[:calibration_count]
    evaluation_records = shuffled_records[calibration_count:]

    calibration_statistics = score_images(calibration_records)
    calibration_batches = make_batches(calibration_statistics)

    evaluation_statistics = score_images(evaluation_records)
    evaluation_batches = make_batches(evaluation_statistics)

    if not calibration_batches or not evaluation_batches:
        raise SystemExit(
            "Need at least one full batch for both calibration and evaluation."
        )

    calibration_scores = [
        float(compare_pixel_statistics(baseline, batch)["score"])
        for batch in calibration_batches
    ]

    threshold = float(max(calibration_scores))

    rows: list[dict] = []

    def record_batches(
        batches: list[list[PixelStatistics]],
        condition: str,
        severity: int,
        subset: str,
    ) -> None:
        for batch_index, batch in enumerate(batches, start=1):
            comparison = compare_pixel_statistics(baseline, batch)
            score = float(comparison["score"])

            rows.append(
                {
                    "subset": subset,
                    "condition": condition,
                    "severity": severity,
                    "batch": batch_index,
                    "sample_count": len(batch),
                    "pixel_drift_score": score,
                    "threshold": threshold,
                    "drift_detected": score > threshold,
                }
            )

    record_batches(
        calibration_batches, "none", 0, "calibration"
    )
    record_batches(
        evaluation_batches, "none", 0, "evaluation"
    )

    for corruption in sorted(CORRUPTIONS):
        for severity in SEVERITIES:
            statistics = score_images(
                evaluation_records, corruption, severity
            )
            record_batches(
                make_batches(statistics),
                corruption,
                severity,
                "evaluation",
            )

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "pixel_drift_scorecard.csv"
    json_path = OUTPUT_DIR / "pixel_drift_scorecard.json"
    results.to_csv(csv_path, index=False)

    clean_calibration = results[
        (results["condition"] == "none")
        & (results["subset"] == "calibration")
    ]
    clean_evaluation = results[
        (results["condition"] == "none")
        & (results["subset"] == "evaluation")
    ]

    false_alarm_rate = float(
        clean_evaluation["drift_detected"].mean()
    )

    condition_summary = []
    corrupted_results = results[results["condition"] != "none"]

    for (condition, severity), group in corrupted_results.groupby(
        ["condition", "severity"],
        sort=True,
    ):
        condition_summary.append(
            {
                "corruption": condition,
                "severity": int(severity),
                "batches": len(group),
                "detection_rate": float(group["drift_detected"].mean()),
                "mean_score": float(group["pixel_drift_score"].mean()),
                "max_score": float(group["pixel_drift_score"].max()),
            }
        )

    summary = {
        "split": "val",
        "baseline_images": baseline["images"],
        "calibration_images": len(calibration_records),
        "evaluation_images": len(evaluation_records),
        "batch_size": BATCH_SIZE,
        "calibration_batches": len(clean_calibration),
        "evaluation_clean_batches": len(clean_evaluation),
        "unused_calibration_images": len(calibration_records) % BATCH_SIZE,
        "unused_evaluation_images": len(evaluation_records) % BATCH_SIZE,
        "threshold_method": (
            "Maximum score among clean calibration batches"
        ),
        "pixel_drift_threshold": threshold,
        "clean_false_alarm_rate": false_alarm_rate,
        "conditions_tested": len(condition_summary),
        "corruption_results": condition_summary,
        "note": (
            "Validation-only pixel drift experiment. Baseline, calibration, "
            "and evaluation image IDs are disjoint. Corruption detection rates "
            "are batch-level estimates from the evaluation subset. A small "
            "number of clean calibration batches does not guarantee a "
            "production false-alarm rate."
        ),
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPixel drift scorecard generated.")
    print(f"Baseline images: {baseline['images']}")
    print(f"Calibration images: {len(calibration_records)}")
    print(f"Evaluation images: {len(evaluation_records)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Clean false-alarm rate: {false_alarm_rate:.3f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()