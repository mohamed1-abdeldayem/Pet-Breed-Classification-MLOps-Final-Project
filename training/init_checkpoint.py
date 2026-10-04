from pet_breed_mlops.backbones import build_backbone
from pet_breed_mlops.inference import save_checkpoint
from pet_breed_mlops.labels import load_classes
import logging

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    n = len(load_classes())
    model = build_backbone("resnet18", n)
    save_checkpoint("models", model, "resnet18", n, model_version="v0-untrained")
    logger.info(
        "Saved models/model.pt and models/eval_transform.json"
    )