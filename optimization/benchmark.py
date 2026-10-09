from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from pet_breed_mlops.onnx_export import load_torch_model, ort_session, val_batch

Backend = Callable[[np.ndarray], np.ndarray]  # (B,3,224,224) float32 -> (B,37) logits


def make_pytorch(run_dir: Path) -> Backend:
    model = load_torch_model(run_dir)

    def run(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(x)).numpy()

    return run


def make_ort(onnx_path: Path) -> Backend:
    sess = ort_session(onnx_path)
    return lambda x: sess.run(["logits"], {"image": x})[0]


def make_openvino(onnx_path: Path) -> Backend:
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(onnx_path))
    model.reshape({"image": [1, 3, 224, 224]})
    compiled = core.compile_model(
        model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
    )
    out = compiled.output(0)

    def run(x: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.asarray(compiled(x[i : i + 1])[out]) for i in range(len(x))]
        )

    return run
def time_single(fn: Backend, x: np.ndarray, warmup: int, runs: int) -> np.ndarray:
    """Per-call latency in seconds for one image at a time (model only, no HTTP/decode)."""
    for _ in range(warmup):
        fn(x)
    out = np.empty(runs)
    for i in range(runs):
        t = time.perf_counter()
        fn(x)
        out[i] = time.perf_counter() - t
    return out


def top1(fn: Backend, images: np.ndarray, labels: np.ndarray, bs: int = 32) -> float:
    preds = np.concatenate([fn(images[i : i + bs]).argmax(1) for i in range(0, len(images), bs)])
    return float((preds == labels).mean())


def val_labels(manifest: Path, n: int) -> np.ndarray:
    recs = sorted(
        (r for r in json.loads(manifest.read_text(encoding="utf-8")) if r["split"] == "val"),
        key=lambda r: r["image_id"],
    )[:n]
    return np.array([r["class_index"] for r in recs])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--label", default="laptop-cpu", help="hardware tag, must match hardware.md")
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--n-val", type=int, default=736)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default="reports/benchmark_results.json")
    a = p.parse_args()

    run_dir = Path(a.run_dir)
    onnx_path = run_dir / "model.onnx"
    images = val_batch(run_dir, a.manifest, a.n_val).numpy()
    labels = val_labels(Path(a.manifest), a.n_val)
    one = images[:1]

    backends: dict[str, tuple[Backend, Path]] = {
        "pytorch": (make_pytorch(run_dir), run_dir / "model.pt"),
        "onnxruntime": (make_ort(onnx_path), onnx_path),
        "openvino": (make_openvino(onnx_path), onnx_path),
    }

    rows = []
    for name, (fn, weights) in backends.items():
        t = time_single(fn, one, a.warmup, a.runs) * 1000
        row = {
            "backend": name,
            "hardware": a.label,
            "batch": 1,
            "runs": a.runs,
            "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99)),
            "throughput_img_s": float(1000 / t.mean()),
            "top1_val": top1(fn, images, labels),
            "n_val": len(labels),
            "size_mb": round(weights.stat().st_size / 1e6, 1),
        }
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    out = Path(a.out)
    old = json.loads(out.read_text()) if out.exists() else []
    keep = [r for r in old if not (r["hardware"] == a.label and r["backend"] in backends)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(keep + rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()