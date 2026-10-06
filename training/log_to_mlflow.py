from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

EXPERIMENT = "pet-breed"
MODEL_NAME = "PetBreedClassifier"
ALIAS = "Production"


def load_run(d: Path) -> dict | None:
    train, calib = d / "metrics.json", d / "calibration.json"
    if not (train.exists() and calib.exists()):
        return None
    return {
        "dir": d,
        "name": d.name,
        "train": json.loads(train.read_text(encoding="utf-8")),
        "calib": json.loads(calib.read_text(encoding="utf-8")),
    }


def log_run(run: dict, is_best: bool) -> str:
    t, c, d = run["train"], run["calib"], run["dir"]
    with mlflow.start_run(run_name=run["name"]) as active:
        mlflow.log_params({k: t[k] for k in ("backbone", "lr", "batch_size", "epochs", "seed", "device")})
        mlflow.log_metrics(
            {
                "top1": t["top1"],
                "f1_macro": t["f1_macro"],
                "ece": c["ece_after"],
                "ece_before": c["ece_before"],
                "temperature": c["temperature"],
                "threshold": c["threshold"],
                "coverage": c["coverage"],
                "selective_accuracy": c["selective_accuracy"],
            }
        )
        mlflow.log_artifact(str(d / "metrics.json"))
        mlflow.log_artifact(str(d / "calibration.json"))
        plot = Path("reports") / f"calibration_{run['name']}.png"
        if plot.exists():
            mlflow.log_artifact(str(plot), "calibration")
        if is_best:  # الأوزان (~100 MB للـ ResNet-50) بنرفعها للأحسن بس
            for f in ("model.pt", "eval_transform.json", "calibration.json"):
                mlflow.log_artifact(str(d / f), "model")
        return active.info.run_id


def register_and_promote(run_id: str, top1: float) -> None:
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException:
        pass  # موجود بالفعل
    source = f"{mlflow.get_run(run_id).info.artifact_uri}/model"
    mv = client.create_model_version(MODEL_NAME, source=source, run_id=run_id, tags={"top1": f"{top1:.4f}"})
    client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    logger.info("registered %s v%s and set alias '%s'", MODEL_NAME, mv.version, ALIAS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="models")
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    a = p.parse_args()

    runs = [r for d in sorted(Path(a.models_dir).iterdir()) if d.is_dir() and (r := load_run(d))]
    if not runs:
        raise SystemExit("No run folders with metrics.json + calibration.json found")
    best = max(runs, key=lambda r: r["train"]["top1"])

    mlflow.set_tracking_uri(a.tracking_uri)
    exp = mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    for run in runs:
        found = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{run['name']}'")
        if found:
            logger.info("skip %s (already logged)", run["name"])
            continue
        is_best = run is best
        run_id = log_run(run, is_best)
        logger.info("logged %s top1=%.4f", run["name"], run["train"]["top1"])
        if is_best:
            register_and_promote(run_id, run["train"]["top1"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()