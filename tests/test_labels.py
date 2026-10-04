import json
from pathlib import Path

CONFIG = Path("configs/label_map.json")

def _load(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))["classes"]


def test_label_map_has_37_unique_classes() -> None:
    classes = _load(CONFIG)
    assert len(classes) == 37
    assert len(set(classes)) == 37