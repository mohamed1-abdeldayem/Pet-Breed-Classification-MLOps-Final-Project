from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def passes(baseline_top1: float, candidate_top1: float, tolerance: float = 0.01) -> bool:
    """True if the candidate is not worse than the baseline by more than `tolerance`."""
    return candidate_top1 >= baseline_top1 - tolerance


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="reports/baseline.json")
    p.add_argument("--candidate", default="reports/latest_metrics.json")
    p.add_argument("--tolerance", type=float, default=0.01)
    a = p.parse_args(argv)

    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))["top1"]
    cand = json.loads(Path(a.candidate).read_text(encoding="utf-8"))["top1"]
    ok = passes(base, cand, a.tolerance)
    print(f"baseline top1={base:.4f} candidate top1={cand:.4f} tolerance={a.tolerance} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())