from __future__ import annotations

import argparse
import os
import platform
import subprocess
from datetime import date
from pathlib import Path

import torch


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "unknown"


def ram_gb() -> str:
    try:
        import psutil

        return f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    except ImportError:
        return "unknown (pip install psutil)"


def build(label: str) -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return f"""# Hardware: {label}

Measured on {date.today().isoformat()}. Every number tagged `{label}` in the
benchmark table was produced on this machine with these settings.

| Item | Value |
|---|---|
| CPU | {cpu_name()} |
| Physical/logical cores | {os.cpu_count()} logical |
| RAM | {ram_gb()} |
| GPU | {gpu} |
| OS | {platform.platform()} |
| Python | {platform.python_version()} |
| PyTorch | {torch.__version__} |
| torch threads | {torch.get_num_threads()} |
| Run mode | native process (no Docker), CPU |

## Load test settings

`locust -u 10 -r 2 -t 90s`, 50 fixed images (seed 0), POST /predict : GET /health = 20 : 1.

## Rule

Do not mix numbers from different hardware in the same column of a comparison table.
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="laptop-cpu")
    p.add_argument("--out", default="reports/hardware.md")
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(build(a.label), encoding="utf-8")
    print(f"wrote {a.out}")