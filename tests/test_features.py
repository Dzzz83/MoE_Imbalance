import numpy as np
import pytest
import torch

from pipeline.data import evaluation_transform, save_training_data
from pipeline.extract_features import extract_features, file_sha256, run_extraction
from pipeline.model import create_model
from pipeline.runtime import save_checkpoint


def test_extraction_preserves_features_order_and_backbone(training_data):
    model = create_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    features, labels = extract_features(model, training_data, "cpu", batch_size=3)
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert not features.requires_grad
    np.testing.assert_array_equal(labels.numpy(), training_data.labels)
    # Independently extract each image to detect ordering or preprocessing errors.
    with torch.no_grad():
        transform = evaluation_transform()
        from torchvision.transforms.functional import to_pil_image
        expected = torch.cat([
            model.backbone(transform(to_pil_image(image)).unsqueeze(0))
            for image in training_data.images
        ])
    torch.testing.assert_close(features, expected)
    repeated, _ = extract_features(model, training_data, "cpu", batch_size=3)
    torch.testing.assert_close(features, repeated, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_feature_cache_round_trip_and_input_checks(tmp_path, training_data):
    checkpoint = tmp_path / "final.pt"
    source = tmp_path / "train.npz"
    output = tmp_path / "features.pt"
    save_training_data(source, training_data)
    save_checkpoint(checkpoint, create_model(), {}, training_data)
    original = checkpoint.read_bytes()
    cache = run_extraction(checkpoint, source, output, batch_size=3)
    loaded = torch.load(output, weights_only=True)
    torch.testing.assert_close(loaded["features"], cache["features"])
    assert loaded["features"].shape == (7, 64)
    expected_scale = loaded["features"].double().square().sum(1).mean().sqrt().item()
    assert loaded["feature_scale"] == pytest.approx(expected_scale)
    assert loaded["checkpoint_sha256"] == file_sha256(checkpoint)
    assert loaded["train_data_sha256"] == file_sha256(source)
    assert checkpoint.read_bytes() == original
    with pytest.raises(FileExistsError):
        run_extraction(checkpoint, source, output)
    training_data.data_settings["data_seed"] = 10
    other = tmp_path / "other.npz"
    save_training_data(other, training_data)
    with pytest.raises(ValueError, match="settings"):
        run_extraction(checkpoint, other, tmp_path / "bad.pt")
