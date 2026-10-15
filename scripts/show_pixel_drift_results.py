import json
from pathlib import Path

path = Path("reports/monitoring/pixel_drift_scorecard.json")
data = json.loads(path.read_text(encoding="utf-8"))

print("Threshold:", data["pixel_drift_threshold"])
print("Clean false-alarm rate:", data["clean_false_alarm_rate"])
print("\nCorruption results:")

for result in data["corruption_results"]:
    print(
        f'{result["corruption"]:16} '
        f'severity={result["severity"]} '
        f'detection_rate={result["detection_rate"]:.3f} '
        f'mean_score={result["mean_score"]:.3f} '
        f'max_score={result["max_score"]:.3f}'
    )
