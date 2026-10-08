import json
import logging
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

logger = logging.getLogger(__name__)
class LabelMapBuilder:
    """Build and save the Oxford-IIIT Pet dataset label map."""

    EXPECTED_NUM_CLASSES = 37

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        output_path: str | Path = "configs/label_map.json",
    ) -> None:
        self.data_root = Path(data_root)
        self.output_path = Path(output_path)

    def load_classes(self) -> list[str]:
        """Load the official class names from Oxford-IIIT Pet."""
        dataset = OxfordIIITPet(
            root=self.data_root,
            split="trainval",
            download=True,
        )

        classes = list(dataset.classes)

        return classes

    def save(self, classes: list[str]) -> None:
        """Save the class names to a JSON label map."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.output_path.write_text(
            json.dumps({"classes": classes}, indent=2),
            encoding="utf-8",
        )

    def build(self) -> None:
        """Build and save the label map."""
        classes = self.load_classes()
        self.save(classes)

        logger.info(
            "Wrote %d classes to %s",
            len(classes),
            self.output_path,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LabelMapBuilder().build()