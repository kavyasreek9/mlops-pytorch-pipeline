"""Model definitions for the MLOps PyTorch pipeline.

Supports a small custom CNN and a fine-tuned torchvision ResNet-18,
selectable via the `architecture` field in configs/training_config.yaml.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """A small CNN suitable for CIFAR-10 / Fashion-MNIST sized (32x32 or 28x28) images."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

# pretrained=False by default: Kubernetes pods in this cluster have no outbound
# DNS resolution, so downloading ImageNet weights from download.pytorch.org
# fails. Training from scratch works fine since conv1/fc are replaced anyway
# for CIFAR-10's 32x32 images and 10 classes.
def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Fine-tuned torchvision ResNet-18, adapted for small images and num_classes."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # Adapt first conv + drop maxpool for small (32x32) inputs like CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "resnet18":
        return get_resnet18(num_classes=num_classes)
    if architecture in ("cnn", "simplecnn"):
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")
