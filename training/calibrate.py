from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pet_breed_mlops.calibration import (
    abstention_stats,
    expected_calibration_error,
    fit_temperature,
    pick_threshold,
    reliability_bins,
)

logger = logging.getLogger(__name__)


def calibrate(run_dir: str | Path, target_accuracy: float = 0.97, plot: str | Path | None = None, write_checkpoint=False) -> dict:
    run_dir = Path(run_dir)
    data = torch.load(run_dir / "val_logits.pt", weights_only=True)
    logits, labels = data["logits"], data["labels"]

    t = fit_temperature(logits, labels)
    stats = pick_threshold(logits, labels, t, target_accuracy)
    result = {
        "temperature": t,
        "ece_before": expected_calibration_error(logits, labels, 1.0),
        "ece_after": expected_calibration_error(logits, labels, t),
        "threshold": stats.threshold,
        "coverage": stats.coverage,
        "selective_accuracy": stats.selective_accuracy,
        "target_accuracy": target_accuracy,
    }
    (run_dir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if write_checkpoint:
        write_to_checkpoint(run_dir, result)
    logger.info("%s -> %s", run_dir, result)
    return result


def _plot(logits, labels, t, result, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, temp, title, ece in (
        (axes[0], 1.0, "Before (T=1)", result["ece_before"]),
        (axes[1], t, f"After (T={t:.2f})", result["ece_after"]),
    ):
        bins = reliability_bins(logits, labels, temp)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.bar([b[0] for b in bins], [b[1] for b in bins], width=0.08, alpha=0.7, label="accuracy")
        ax.set_title(f"{title}  ECE={ece:.3f}")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)

def write_to_checkpoint(run_dir: str | Path, result: dict) -> None:
    """Ship the calibration with the weights, so the API can never serve raw softmax."""
    path = Path(run_dir) / "model.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt["temperature"] = result["temperature"]
    ckpt["threshold"] = result["threshold"]
    torch.save(ckpt, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--target-accuracy", type=float, default=0.97)
    p.add_argument("--plot", default=None)
    p.add_argument("--write-checkpoint", action="store_true")
    a = p.parse_args()
    calibrate(a.run_dir, a.target_accuracy, a.plot, a.write_checkpoint)