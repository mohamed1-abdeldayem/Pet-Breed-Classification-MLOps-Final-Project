from __future__ import annotations

from torch import nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet50", "resnet18", "mobilenet_v3_small")


def build_backbone(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Torchvision backbone with the classification head replaced by `num_classes` outputs."""
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone '{name}'. Supported: {SUPPORTED_BACKBONES}")
    return model