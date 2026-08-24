import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, get_model, get_resnet18  # noqa: E402


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_resnet18_output_shape():
    model = get_resnet18(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_get_model_factory():
    model = get_model("resnet18", num_classes=10)
    assert model is not None

    model = get_model("cnn", num_classes=10)
    assert isinstance(model, SimpleCNN)


def test_get_model_invalid_architecture_raises():
    try:
        get_model("not-a-real-architecture", num_classes=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
