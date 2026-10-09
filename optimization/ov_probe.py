import time
from pathlib import Path

import numpy as np
import openvino as ov

ONNX = Path("models/resnet50/model.onnx")
x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)

CONFIGS = {
    "baseline (LATENCY, f32)": {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"},
    "no hints": {},
    "1 stream, 4 threads": {"NUM_STREAMS": "1", "INFERENCE_NUM_THREADS": "4"},
    "1 stream, 8 threads": {"NUM_STREAMS": "1", "INFERENCE_NUM_THREADS": "8"},
}

core = ov.Core()
for name, cfg in CONFIGS.items():
    try:
        model = core.read_model(str(ONNX))
        model.reshape({"image": [1, 3, 224, 224]})
        compiled = core.compile_model(model, "CPU", cfg)
        for _ in range(20):
            compiled(x)
        t = []
        for _ in range(100):
            s = time.perf_counter()
            compiled(x)
            t.append((time.perf_counter() - s) * 1000)
        print(f"{name:28s} p50={np.percentile(t, 50):6.1f}  p95={np.percentile(t, 95):6.1f} ms")
    except Exception as e:
        print(f"{name:28s} FAILED: {e!r}")