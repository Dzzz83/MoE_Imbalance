import torch
from torch import nn

from pipeline.model import create_model
from pipeline.runtime import seed_everything


def test_model_contract_initialization_and_backward():
    seed_everything(12)
    model = create_model()
    seed_everything(12)
    same_initialization = create_model()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, same_initialization.state_dict()[name], rtol=0, atol=0)
    assert sum(isinstance(module, nn.Conv2d) for module in model.modules()) == 31
    images = torch.randn(2, 3, 32, 32)
    features = model.backbone(images)
    assert features.shape == (2, 64)
    logits = model(images)
    assert logits.shape == (2, 100)
    torch.testing.assert_close(logits, model.fc(features))
    nn.functional.cross_entropy(logits, torch.tensor([0, 99])).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
