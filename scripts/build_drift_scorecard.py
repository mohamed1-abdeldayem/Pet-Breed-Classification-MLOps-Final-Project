from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from pet_breed_mlops.corruptions import CORRUPTIONS, SEVERITIES, apply
from pet_breed_mlops.inference import PetBreedClassifier
from pet_breed_mlops.monitoring import calculate_psi

VAL_FILE = Path("data/splits/val.txt")
MANIFEST_FILE = Path("data/manifest.json")
BASELINE_FILE = Path("reports/monitoring/baseline.json")
MODEL_DIR = Path("models/resnet50")
OUTPUT_DIR = Path("reports/monitoring")

BATCH_SIZE = 100
CALIBRATION_SIMULATIONS = 1000
THRESHOLD_PERCENTILE = 95
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

    if not records:
        raise SystemExit("No validation images found.")

    return [records[key] for key in sorted(records)]


def predict_breeds(
    classifier: PetBreedClassifier,
    records: list[dict],
    corruption: str | None = None,
    severity: int = 0,
) -> list[str]:
    predictions: list[str] = []

    for index, record in enumerate(records, start=1):
        try:
            with Image.open(record["path"]) as image:
                image.load()
                image = image.convert("RGB")

            if corruption is not None:
                image = apply(image, corruption, severity)

            predictions.append(classifier.predict(image).breed)

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

    return predictions



def make_batches(values: list[str]) -> list[list[str]]:
    shuffled = values.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(shuffled)

    full_count = len(shuffled) // BATCH_SIZE
    return [
        shuffled[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        for i in range(full_count)
    ]



def calibrate_threshold(
    baseline: dict[str, int],
    batch_size: int,
) -> tuple[float, list[float]]:
    total = sum(baseline.values())
    if total <= 0:
        raise ValueError("Monitoring baseline has no predictions.")

    categories = sorted(baseline)
    probabilities = np.array(
        [baseline[category] / total for category in categories],
        dtype=float,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    simulated_scores: list[float] = []

    for _ in range(CALIBRATION_SIMULATIONS):
        counts = rng.multinomial(batch_size, probabilities)
        sample = [
            category
            for category, count in zip(categories, counts)
            for _ in range(int(count))
        ]
        simulated_scores.append(calculate_psi(baseline, sample))

    threshold = float(
        np.percentile(simulated_scores, THRESHOLD_PERCENTILE)
    )
    return threshold, simulated_scores


def main() -> None:
    baseline_data = json.loads(
        BASELINE_FILE.read_text(encoding="utf-8")
    )
    baseline_breeds = baseline_data["predicted_breed_distribution"]

    records = load_validation_records()
    classifier = PetBreedClassifier(MODEL_DIR, device="cpu")

    # Score original images once, then reuse the predictions for clean batches.
    clean_predictions = predict_breeds(classifier, records)
    clean_batches = make_batches(clean_predictions)

    if len(clean_batches) < 2:
        raise SystemExit(
            f"Need at least 2 full batches of {BATCH_SIZE} validation images."
        )

    threshold, simulated_scores = calibrate_threshold(
        baseline_breeds,
        BATCH_SIZE,
    )

    rows: list[dict] = []

    def record_batches(
        batches: list[list[str]],
        corruption: str,
        severity: int,
    ) -> None:
        for batch_index, batch in enumerate(batches, start=1):
            psi = calculate_psi(baseline_breeds, batch)
            rows.append(
                {
                    "condition": corruption,
                    "severity": severity,
                    "batch": batch_index,
                    "sample_count": len(batch),
                    "psi": psi,
                    "threshold": threshold,
                    "drift_detected": psi > threshold,
                }
            )

    record_batches(clean_batches, "none", 0)

    for corruption in sorted(CORRUPTIONS):
        for severity in SEVERITIES:
            predictions = predict_breeds(
                classifier,
                records,
                corruption=corruption,
                severity=severity,
            )
            record_batches(
                make_batches(predictions),
                corruption,
                severity,
            )

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "drift_scorecard.csv"
    json_path = OUTPUT_DIR / "drift_scorecard.json"
    results.to_csv(csv_path, index=False)

    clean_results = results[results["condition"] == "none"]
    clean_false_alarm_rate = float(
        clean_results["drift_detected"].mean()
    )

    condition_summary = []
    corrupted_results = results[results["condition"] != "none"]

    for (corruption, severity), group in corrupted_results.groupby(
        ["condition", "severity"],
        sort=True,
    ):
        condition_summary.append(
            {
                "corruption": corruption,
                "severity": int(severity),
                "batches": len(group),
                "detection_rate": float(group["drift_detected"].mean()),
                "mean_psi": float(group["psi"].mean()),
                "max_psi": float(group["psi"].max()),
            }
        )

    summary = {
        "model_version": classifier.model_version,
        "split": "val",
        "baseline_images": baseline_data["validation_images"],
        "batch_size": BATCH_SIZE,
        "full_batches_per_condition": len(clean_batches),
        "unused_remainder_images": len(records) % BATCH_SIZE,
        "threshold_method": (
            f"{THRESHOLD_PERCENTILE}th percentile of "
            f"{CALIBRATION_SIMULATIONS} simulated clean batches"
        ),
        "psi_threshold": threshold,
        "calibration_seed": RANDOM_SEED,
        "simulated_clean_psi_p95": float(
            np.percentile(simulated_scores, 95)
        ),
        "clean_false_alarm_rate": clean_false_alarm_rate,
        "clean_batches": len(clean_results),
        "conditions_tested": len(condition_summary),
        "corruption_results": condition_summary,
        "note": (
            "Validation-only detector evaluation based on predicted-breed "
            "distribution PSI. Detection rates use full batches only. "
            "Clean false-alarm rate is estimated from a small number of "
            "validation batches; it is not a production guarantee."
        ),
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nDrift scorecard generated.")
    print(f"PSI threshold: {threshold:.6f}")
    print(f"Clean false-alarm rate: {clean_false_alarm_rate:.3f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
