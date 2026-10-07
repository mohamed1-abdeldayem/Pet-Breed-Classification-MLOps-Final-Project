import tarfile
from pathlib import Path

ARCHIVE = Path("data/archive")
OUT = Path("data/raw/oxford-iiit-pet")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("images.tar.gz", "annotations.tar.gz"):
        with tarfile.open(ARCHIVE / name) as tar:
            tar.extractall(OUT, filter="data")
    print(f"extracted into {OUT}")


if __name__ == "__main__":
    main()