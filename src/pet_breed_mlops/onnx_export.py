from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pet_breed_mlops.backbones import build_backbone
from pet_breed_mlops.inference import CHECKPOINT_NAME, TRANSFORM_NAME
from pet_breed_mlops.preprocessing import EvalTransformConfig, build_eval_transform, to_rgb

ATOL = 1e-4


def load_torch_model(run_dir: str | Path) -> torch.nn.Module:
    ckpt = torch.load(Path(run_dir) / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    model = build_backbone(ckpt["backbone"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def export_onnx(run_dir: str | Path, out: str | Path | None = None, opset: int = 17) -> Path:
    """Export raw logits with a dynamic batch dimension."""
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir / "model.onnx"
    model = load_torch_model(run_dir)
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        str(out),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return out


def ort_session(onnx_path: str | Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def compare_logits(model: torch.nn.Module, onnx_path: str | Path, batch: torch.Tensor) -> dict:
    """Max absolute logit difference and top-1 agreement between PyTorch and ONNX Runtime."""
    with torch.inference_mode():
        ref = model(batch).numpy()
    got = ort_session(onnx_path).run(["logits"], {"image": batch.numpy()})[0]
    return {
        "n_images": int(batch.shape[0]),
        "max_abs_diff": float(np.abs(ref - got).max()),
        "top1_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
    }


def val_batch(run_dir: str | Path, manifest: str | Path, n: int = 200) -> torch.Tensor:
    """First n val images (sorted by id, so the set is fixed), through the shared eval transform."""
    records = sorted(
        (r for r in json.loads(Path(manifest).read_text(encoding="utf-8")) if r["split"] == "val"),
        key=lambda r: r["image_id"],
    )[:n]
    transform = build_eval_transform(EvalTransformConfig.load(Path(run_dir) / TRANSFORM_NAME))
    tensors = []
    for r in records:
        with Image.open(r["path"]) as img:
            tensors.append(transform(to_rgb(img)))
    return torch.stack(tensors)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--report", default="reports/onnx_agreement.json")
    a = p.parse_args()

    onnx_path = export_onnx(a.run_dir)
    result = compare_logits(load_torch_model(a.run_dir), onnx_path, val_batch(a.run_dir, a.manifest, a.n))
    result.update(atol=ATOL, passed=result["max_abs_diff"] <= ATOL, onnx_mb=onnx_path.stat().st_size / 1e6)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()