import pytest

from pet_breed_mlops.backbones import build_backbone
from pet_breed_mlops.inference import save_checkpoint
from pet_breed_mlops.labels import load_classes


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    n = len(load_classes())
    save_checkpoint(d, build_backbone("resnet18", n), "resnet18", n, model_version="test")
    return d