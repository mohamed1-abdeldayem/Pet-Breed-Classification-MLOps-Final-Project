from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PixelStatistics:
    mean_rgb: tuple[float, float, float]
    std_rgb: tuple[float, float, float]
    mean_brightness: float
    edge_strength: float
    jpeg_blockiness: float


def calculate_pixel_statistics(image: Image.Image) -> PixelStatistics:
    """Calculate RGB, brightness, edge, and JPEG blockiness statistics."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    if rgb.size == 0:
        raise ValueError("Image must not be empty")

    mean_rgb = tuple(float(value) for value in rgb.mean(axis=(0, 1)))
    std_rgb = tuple(float(value) for value in rgb.std(axis=(0, 1)))

    brightness = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )

    horizontal_diff = np.abs(np.diff(brightness, axis=1))
    vertical_diff = np.abs(np.diff(brightness, axis=0))

    edge_values = []
    if horizontal_diff.size:
        edge_values.append(float(horizontal_diff.mean()))
    if vertical_diff.size:
        edge_values.append(float(vertical_diff.mean()))

    edge_strength = float(np.mean(edge_values)) if edge_values else 0.0

    block_differences = []
    non_block_differences = []

    if horizontal_diff.size:
        columns = np.arange(1, brightness.shape[1])
        block_mask = columns % 8 == 0

        if block_mask.any():
            block_differences.append(
                float(horizontal_diff[:, block_mask].mean())
            )
        if (~block_mask).any():
            non_block_differences.append(
                float(horizontal_diff[:, ~block_mask].mean())
            )

    if vertical_diff.size:
        rows = np.arange(1, brightness.shape[0])
        block_mask = rows % 8 == 0

        if block_mask.any():
            block_differences.append(
                float(vertical_diff[block_mask, :].mean())
            )
        if (~block_mask).any():
            non_block_differences.append(
                float(vertical_diff[~block_mask, :].mean())
            )

    if block_differences and non_block_differences:
        jpeg_blockiness = max(
            0.0,
            float(np.mean(block_differences) - np.mean(non_block_differences)),
        )
    else:
        jpeg_blockiness = 0.0

    return PixelStatistics(
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        mean_brightness=float(brightness.mean()),
        edge_strength=edge_strength,
        jpeg_blockiness=jpeg_blockiness,
    )


def statistics_to_dict(stats: PixelStatistics) -> dict[str, object]:
    """Convert pixel statistics to JSON-compatible values."""
    return asdict(stats)


def compare_pixel_statistics(
    baseline: dict[str, object],
    current: list[PixelStatistics],
    epsilon: float = 1e-6,
) -> dict[str, object]:
    """Compare batch pixel statistics with the validation baseline."""
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")

    if not current:
        raise ValueError("Current observations must not be empty")

    feature_values: dict[str, list[float]] = {
        "mean_rgb": [],
        "std_rgb": [],
        "mean_brightness": [],
        "edge_strength": [],
        "jpeg_blockiness": [],
    }

    for stats in current:
        feature_values["mean_rgb"].extend(stats.mean_rgb)
        feature_values["std_rgb"].extend(stats.std_rgb)
        feature_values["mean_brightness"].append(stats.mean_brightness)
        feature_values["edge_strength"].append(stats.edge_strength)
        feature_values["jpeg_blockiness"].append(stats.jpeg_blockiness)

    baseline_features = baseline["statistics"]
    if not isinstance(baseline_features, dict):
        raise TypeError("Baseline must contain a statistics mapping")

    results: dict[str, object] = {}
    scores: list[float] = []

    for feature, values in feature_values.items():
        reference = baseline_features.get(feature)

        if not isinstance(reference, dict):
            raise KeyError(f"Baseline is missing feature: {feature}")

        reference_mean = reference.get("mean")
        reference_std = reference.get("std")

        if feature in {"mean_rgb", "std_rgb"}:
            if (
                not isinstance(reference_mean, list)
                or not isinstance(reference_std, list)
                or len(reference_mean) != 3
                or len(reference_std) != 3
            ):
                raise ValueError(f"Invalid RGB baseline for {feature}")

            current_mean = np.asarray(values, dtype=float).reshape(
                -1, 3
            ).mean(axis=0)
            means = np.asarray(reference_mean, dtype=float)
            stds = np.asarray(reference_std, dtype=float)
            feature_scores = (
                np.abs(current_mean - means) / np.maximum(stds, epsilon)
            )

            results[feature] = {
                "baseline_mean": means.tolist(),
                "current_mean": current_mean.tolist(),
                "standardized_difference": feature_scores.tolist(),
            }
            scores.extend(feature_scores.tolist())
        else:
            if not isinstance(reference_mean, (int, float)) or not isinstance(
                reference_std, (int, float)
            ):
                raise ValueError(f"Invalid baseline for {feature}")

            current_mean = float(np.mean(values))
            score = abs(current_mean - reference_mean) / max(
                reference_std, epsilon
            )
            results[feature] = {
                "baseline_mean": float(reference_mean),
                "current_mean": current_mean,
                "standardized_difference": score,
            }
            scores.append(score)

    return {
        "image_count": len(current),
        "score": float(max(scores)),
        "features": results,
        "note": (
            "Score is the maximum standardized difference across available "
            "features. It is not a calibrated drift decision until a "
            "threshold is set."
        ),
    }
