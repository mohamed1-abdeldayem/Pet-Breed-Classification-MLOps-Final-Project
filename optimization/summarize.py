import json
from pathlib import Path

import numpy as np

HW = "laptop-cpu"
files = sorted(Path("reports").glob("benchmark_r[0-9].json"))
runs = [json.loads(p.read_text()) for p in files]


def row(run, backend):
    return next(r for r in run if r["backend"] == backend and r["hardware"] == HW)


print(f"{len(runs)} rounds: {[p.name for p in files]}")
for metric in ("p50_ms", "p95_ms"):
    print(f"\n{metric}")
    for b in ("pytorch", "onnxruntime", "openvino"):
        v = [row(r, b)[metric] for r in runs]
        print(f"  {b:12s} median={np.median(v):6.1f}  range={min(v):.1f}-{max(v):.1f}")
    pt = [row(r, "pytorch")[metric] for r in runs]
    ort = [row(r, "onnxruntime")[metric] for r in runs]
    ov = [row(r, "openvino")[metric] for r in runs]
    print("  ORT speedup vs PyTorch per round:", [f"{a / b:.2f}x" for a, b in zip(pt, ort)])
    print("  OpenVINO / ORT per round:        ", [f"{b / a:.2f}" for a, b in zip(ort, ov)])