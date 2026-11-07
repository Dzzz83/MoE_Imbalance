"""Stage 4: continue model.fc training on cached raw-coordinate features."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.runtime import load_checkpoint, resolve_device, seed_everything, seed_worker


DEFAULTS = {
    "mode": "synthetic", "seed": 0, "target_count": 500,
    "epochs": 40, "batch_size": 128, "num_workers": 0,
    "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    paths = {"pretrained_checkpoint", "feature_cache", "synthetic_features", "output_dir"}
    unknown = config.keys() - DEFAULTS.keys() - paths
    if unknown:
        raise ValueError(f"Unknown classifier settings: {', '.join(sorted(unknown))}")
    result = DEFAULTS | {"synthetic_features": None} | config
    if result["mode"] not in {"real", "duplicate", "synthetic"}:
        raise ValueError("mode must be real, duplicate, or synthetic")
    for name in ("pretrained_checkpoint", "feature_cache", "output_dir"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"{name} must be a nonempty path string")
    if result["mode"] == "synthetic":
        if not isinstance(result["synthetic_features"], str) or not result["synthetic_features"]:
            raise ValueError("synthetic mode requires synthetic_features")
    elif result["synthetic_features"] is not None:
        raise ValueError("real and duplicate modes must not specify synthetic_features")
    for name, minimum in {
        "seed": 0, "target_count": 1, "epochs": 1, "batch_size": 1, "num_workers": 0
    }.items():
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if result["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    for name in ("learning_rate", "momentum", "weight_decay"):
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if result["learning_rate"] == 0 or result["momentum"] >= 1:
        raise ValueError("learning_rate must be positive; momentum must be less than one")
    return result


def validate_real_cache(cache, pretrained, checkpoint_hash):
    features, labels, counts = (cache[key] for key in ("features", "labels", "class_counts"))
    if (features.ndim != 2 or features.shape[1] != 64 or not len(features)
            or not features.is_floating_point() or not torch.isfinite(features).all()):
        raise ValueError("Expected finite raw features with shape (N, 64)")
    if labels.dtype != torch.int64 or labels.shape != (len(features),):
        raise ValueError("Expected aligned int64 labels")
    if (labels < 0).any() or (labels >= 100).any():
        raise ValueError("Labels must lie in [0, 99]")
    if counts.dtype != torch.int64 or counts.shape != (100,):
        raise ValueError("Expected 100 real class counts")
    if not torch.equal(torch.bincount(labels, minlength=100), counts):
        raise ValueError("Feature cache class counts do not match labels")
    if cache["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Feature cache was extracted from a different pretrained checkpoint")
    if not torch.equal(counts, pretrained["class_counts"].cpu()):
        raise ValueError("Pretrained checkpoint class counts do not match the feature cache")
    if cache["data_settings"] != pretrained["data_settings"]:
        raise ValueError("Pretrained checkpoint data settings do not match the feature cache")
    scale = cache["feature_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid feature coordinate scale")


def prepare_training_features(cache, config, cache_hash, synthetic=None):
    """Return raw-coordinate CPU features; never divide by feature_scale here."""
    features = cache["features"].float().contiguous()
    labels = cache["labels"].clone()
    counts = cache["class_counts"]
    if config["mode"] == "real":
        return features, labels
    expected = (config["target_count"] - counts).clamp_min(0)
    if config["mode"] == "duplicate":
        generator = torch.Generator().manual_seed(config["seed"])
        extras = []
        for class_id, needed in enumerate(expected.tolist()):
            if needed:
                indices = torch.where(labels == class_id)[0]
                if not len(indices):
                    raise ValueError(f"Cannot duplicate absent class {class_id}")
                extras.append(indices[torch.randint(len(indices), (needed,), generator=generator)])
        if extras:
            indices = torch.cat(extras)
            return torch.cat([features, features[indices]]), torch.cat([labels, labels[indices]])
        return features, labels
    if synthetic is None:
        raise ValueError("Synthetic feature dataset is missing")
    generated, targets = synthetic["features"], synthetic["labels"]
    if (generated.ndim != 2 or generated.shape[1] != 64
            or not generated.is_floating_point() or not torch.isfinite(generated).all()):
        raise ValueError("Invalid synthetic features")
    if (targets.dtype != torch.int64 or targets.shape != (len(generated),)
            or (targets < 0).any() or (targets >= 100).any()):
        raise ValueError("Invalid synthetic labels")
    if not torch.equal(torch.bincount(targets, minlength=100), expected):
        raise ValueError("Synthetic counts do not cover all classes to the requested target")
    if not torch.equal(synthetic["real_class_counts"], counts):
        raise ValueError("Synthetic dataset has different real class counts")
    if not torch.equal(synthetic["synthetic_class_counts"], expected):
        raise ValueError("Synthetic count metadata is inconsistent")
    if synthetic["generation_config"]["feature_cache_sha256"] != cache_hash:
        raise ValueError("Synthetic features came from a different real feature cache")
    if synthetic["checkpoint_sha256"] != cache["checkpoint_sha256"]:
        raise ValueError("Synthetic dataset checkpoint provenance does not match")
    if synthetic["data_settings"] != cache["data_settings"]:
        raise ValueError("Synthetic dataset data settings do not match")
    if not math.isclose(synthetic["feature_scale"], cache["feature_scale"], rel_tol=1e-12):
        raise ValueError("Synthetic dataset coordinate scale does not match")
    # Stage 3 already multiplied synthetic working features by feature_scale.
    return torch.cat([features, generated.float()]), torch.cat([labels, targets])


def frozen_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.backbone.state_dict().items()}


def verify_backbone(model, reference):
    if model.backbone.training or any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("Backbone is not frozen in evaluation mode")
    for name, value in model.backbone.state_dict().items():
        if not torch.equal(value.detach().cpu(), reference[name]):
            raise RuntimeError(f"Backbone parameter or buffer changed: {name}")


def run_training(config, device="cpu", resume=False):
    config = resolve_config(config)
    device = resolve_device(device)
    output = Path(config["output_dir"])
    hashes = {
        "pretrained_checkpoint_sha256": file_sha256(config["pretrained_checkpoint"]),
        "feature_cache_sha256": file_sha256(config["feature_cache"]),
        "synthetic_features_sha256": (
            file_sha256(config["synthetic_features"]) if config["mode"] == "synthetic" else None
        ),
        "classifier_code_sha256": file_sha256(__file__),
    }
    seed_everything(config["seed"])
    model, pretrained = load_checkpoint(config["pretrained_checkpoint"], device)
    cache = torch.load(config["feature_cache"], map_location="cpu", weights_only=True)
    validate_real_cache(cache, pretrained, hashes["pretrained_checkpoint_sha256"])
    synthetic = (
        torch.load(config["synthetic_features"], map_location="cpu", weights_only=True)
        if config["mode"] == "synthetic" else None
    )
    features, labels = prepare_training_features(
        cache, config, hashes["feature_cache_sha256"], synthetic
    )
    if not torch.isfinite(features).all():
        raise ValueError("Features overflowed when converting to classifier dtype")
    model.backbone.requires_grad_(False)
    model.eval()
    model.fc.train()
    reference = frozen_state(model)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
        generator=generator, worker_init_fn=seed_worker,
    )
    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=config["learning_rate"],
        momentum=config["momentum"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    history, start_epoch, elapsed = [], 0, 0.0
    final_path, last_path = output / "final.pt", output / "last.pt"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Output exists: {output}; use --resume or another folder")
        saved_config = json.loads((output / "config.json").read_text())
        saved_hashes = json.loads((output / "input_hashes.json").read_text())
        if saved_config != config or saved_hashes != hashes:
            raise ValueError("Resume requires identical configuration, inputs, and classifier code")
        if final_path.exists():
            print(f"Completed classifier already exists: {final_path}", flush=True)
            return final_path
        if last_path.exists():
            previous = torch.load(last_path, map_location="cpu", weights_only=True)
            if previous["config"] != config or previous["input_hashes"] != hashes:
                raise ValueError("Resume checkpoint metadata does not match")
            model.load_state_dict(previous["model_state_dict"])
            verify_backbone(model, reference)
            optimizer.load_state_dict(previous["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            scheduler.load_state_dict(previous["scheduler_state_dict"])
            generator.set_state(previous["loader_rng_state"])
            torch.set_rng_state(previous["torch_rng_state"])
            if device.type == "cuda" and previous["cuda_rng_states"]:
                torch.cuda.set_rng_state_all(previous["cuda_rng_states"])
            start_epoch = previous["epoch"]
            history, elapsed = previous["history"], previous["training_seconds"]
            if len(history) != start_epoch or not 0 <= start_epoch <= config["epochs"]:
                raise ValueError("Invalid resume epoch/history")
            print(f"Resuming after epoch {start_epoch}", flush=True)
    else:
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
        write_json(output / "input_hashes.json", hashes)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        started = time.perf_counter()
        model.eval()
        model.fc.train()
        loss_sum, seen = 0.0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch_features, targets in loader:
            batch_features, targets = batch_features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.fc(batch_features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite classifier loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            seen += len(targets)
        if not all(torch.isfinite(p).all() for p in model.fc.parameters()):
            raise RuntimeError("Nonfinite classifier parameters")
        verify_backbone(model, reference)
        scheduler.step()
        elapsed += time.perf_counter() - started
        history.append({
            "epoch": epoch, "stage": "classifier", "loss": loss_sum / seen,
            "learning_rate": learning_rate, "examples_processed": seen,
        })
        last = {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "config": config, "class_counts": cache["class_counts"],
            "data_settings": cache["data_settings"], "input_hashes": hashes,
            "epoch": epoch, "history": history, "training_seconds": elapsed,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loader_rng_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        atomic_torch_save(last, last_path)
        write_json(output / "history.json", history)
        print(f"Classifier epoch {epoch}/{config['epochs']} loss={loss_sum / seen:.6f}", flush=True)
    verify_backbone(model, reference)
    summary = {
        "mode": config["mode"], "real_examples": len(cache["features"]),
        "additional_examples": len(features) - len(cache["features"]),
        "examples_per_epoch": len(features), "epochs": config["epochs"],
        "total_examples_processed": len(features) * config["epochs"],
        "training_seconds": elapsed, "backbone_unchanged": True,
        "combined_class_counts": torch.bincount(labels, minlength=100).tolist(),
    }
    final = {
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": config, "class_counts": cache["class_counts"],
        "data_settings": cache["data_settings"], "input_hashes": hashes,
        "training_summary": summary,
    }
    # Preserve the original REAL class counts for Head/Medium/Tail evaluation.
    write_json(output / "training_summary.json", summary)
    write_json(output / "history.json", history)
    atomic_torch_save(final, final_path)
    print(f"Saved final checkpoint to {final_path}", flush=True)
    return final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.output_dir:
            config["output_dir"] = args.output_dir
        run_training(config, args.device, args.resume)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
