from __future__ import annotations

import argparse
import json
from pathlib import Path

from pet_breed_mlops.quality_gate import main as quality_gate_main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    # Runtime errors propagate and keep the Airflow task failed.
    exit_code = quality_gate_main(
        [
            "--baseline", args.baseline,
            "--candidate", args.candidate,
            "--tolerance", str(args.tolerance),
        ]
    )

    result = {
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
    }

    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "passed" if exit_code == 0 else "failed",
                "exit_code": exit_code,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(result))

    # A quality rejection is a completed check, not a runtime error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
