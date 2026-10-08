import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from pet_breed_mlops.backbones import build_backbone  # noqa: E402
from pet_breed_mlops.inference import save_checkpoint  # noqa: E402
from pet_breed_mlops.onnx_export import ATOL, compare_logits, export_onnx, load_torch_model  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    torch.manual_seed(0)
    save_checkpoint(d, build_backbone("resnet18", 37), "resnet18", 37)
    return d, export_onnx(d)


def test_onnx_logits_match_pytorch(exported) -> None:
    d, onnx_path = exported
    batch = torch.randn(8, 3, 224, 224)
    r = compare_logits(load_torch_model(d), onnx_path, batch)
    assert r["max_abs_diff"] <= ATOL, r
    assert r["top1_agreement"] == 1.0


def test_dynamic_batch_size(exported) -> None:
    d, onnx_path = exported
    for n in (1, 3):
        r = compare_logits(load_torch_model(d), onnx_path, torch.randn(n, 3, 224, 224))
        assert r["n_images"] == n and r["max_abs_diff"] <= ATOL