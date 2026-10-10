import shutil
from pathlib import Path

import bentoml

from pet_breed_mlops.inference import CHECKPOINT_NAME, TRANSFORM_NAME

RUN_DIR = Path("models/resnet50")

with bentoml.models.create(name="pet_breed_onnx") as m:
    for f in ("model.onnx", CHECKPOINT_NAME, TRANSFORM_NAME):
        shutil.copy(RUN_DIR / f, Path(m.path) / f)
    print("registered:", m.tag)