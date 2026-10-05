from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pet_breed_mlops.backbones import SUPPORTED_BACKBONES, build_backbone
from pet_breed_mlops.inference import save_checkpoint
from pet_breed_mlops.labels import load_classes
from pet_breed_mlops.preprocessing import (
    EvalTransformConfig,
    build_eval_transform,
    build_train_transform,
    to_rgb,
)

logger = logging.getLogger(__name__)


class ManifestDataset(Dataset):
    """Reads images listed in the manifest. Labels come from the committed label map."""

    def __init__(self, records: list[dict], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        with Image.open(r["path"]) as img:
            x = self.transform(to_rgb(img))
        return x, r["class_index"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> dict:
    """Top-1 and macro-F1, plus raw logits/labels (reused later for calibration)."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).float().cpu())
        all_labels.append(y)
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    preds = logits.argmax(1)

    cm = torch.bincount(labels * num_classes + preds, minlength=num_classes**2)
    cm = cm.view(num_classes, num_classes).float()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    return {
        "top1": (preds == labels).float().mean().item(),
        "f1_macro": f1.mean().item(),
        "logits": logits,
        "labels": labels,
    }


def fit(
    backbone: str,
    lr: float = 5e-4,
    batch_size: int = 64,
    epochs: int = 8,
    seed: int = 42,
    manifest: str | Path = "data/manifest.json",
    out_dir: str | Path = "models/run",
    num_workers: int = 2,
    model_version: str = "v1",
) -> dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = load_classes()
    num_classes = len(classes)

    records = json.loads(Path(manifest).read_text(encoding="utf-8"))
    train_rec = [r for r in records if r["split"] == "train"]
    val_rec = [r for r in records if r["split"] == "val"]  # test is never touched here

    cfg = EvalTransformConfig()
    train_dl = DataLoader(
        ManifestDataset(train_rec, build_train_transform(cfg)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=device == "cuda", drop_last=True,
    )
    val_dl = DataLoader(
        ManifestDataset(val_rec, build_eval_transform(cfg)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    model = build_backbone(backbone, num_classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    best, best_state = {"top1": -1.0}, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        m = evaluate(model, val_dl, device, num_classes)
        logger.info(
            "epoch %d/%d loss=%.4f val_top1=%.4f val_f1=%.4f",
            epoch, epochs, running / len(train_dl), m["top1"], m["f1_macro"],
        )
        if m["top1"] > best["top1"]:
            best, best_state = m, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    out_dir = Path(out_dir)
    save_checkpoint(out_dir, model.cpu(), backbone, num_classes, model_version=model_version,
                    transform_config=cfg)
    torch.save({"logits": best["logits"], "labels": best["labels"]}, out_dir / "val_logits.pt")

    metrics = {
        "backbone": backbone, "lr": lr, "batch_size": batch_size, "epochs": epochs,
        "seed": seed, "top1": best["top1"], "f1_macro": best["f1_macro"], "device": device,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("saved %s -> %s", metrics, out_dir)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default="resnet18")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--model-version", default="v1")
    a = p.parse_args()
    fit(
        a.backbone, a.lr, a.batch_size, a.epochs, a.seed,
        out_dir=a.out_dir or f"models/{a.backbone}", model_version=a.model_version,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()