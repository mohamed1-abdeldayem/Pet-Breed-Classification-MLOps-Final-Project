import json
from pathlib import Path
import os

DEFAULT_LABEL_MAP = Path(
    os.environ.get(
        "PET_LABEL_MAP",
        Path(__file__).resolve().parents[2] / "configs" / "label_map.json",
    )
)

def load_classes(path: Path = DEFAULT_LABEL_MAP) -> list[str]:
    return json.loads(path.read_text())["classes"]


CAT_BREEDS = frozenset(
    {
        "Abyssinian",
        "Bengal",
        "Birman",
        "Bombay",
        "British Shorthair",
        "Egyptian Mau",
        "Maine Coon",
        "Persian",
        "Ragdoll",
        "Russian Blue",
        "Siamese",
        "Sphynx",
    }
)


def species_of(breed: str) -> str:
    return "cat" if breed in CAT_BREEDS else "dog"
