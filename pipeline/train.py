"""Create shared inputs, invoke one training method, and save its final model."""

import argparse
import json
from pathlib import Path

from pipeline.data import load_training_data
from pipeline.methods import baseline
from pipeline.model import create_model
from pipeline.runtime import resolve_device, save_checkpoint, seed_everything, write_json

# A method supplies its training function and its own option resolver.
METHODS = {"baseline": (baseline.train, baseline.resolve_options)}


def resolve_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    allowed = {"method", "train_data", "seed", "batch_size", "num_workers", "method_options"}
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown configuration settings: {', '.join(sorted(unknown))}")
    resolved = {"seed": 0, "batch_size": 128, "num_workers": 4, "method_options": {}} | config
    if not isinstance(resolved.get("method"), str) or resolved["method"] not in METHODS:
        raise ValueError(f"Unknown method; available methods: {', '.join(METHODS)}")
    if not isinstance(resolved.get("train_data"), str) or not resolved["train_data"]:
        raise ValueError("train_data must be a prepared-data file path")
    for name, minimum in (("seed", 0), ("batch_size", 1), ("num_workers", 0)):
        if type(resolved[name]) is not int or resolved[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if resolved["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    if not isinstance(resolved["method_options"], dict):
        raise ValueError("method_options must be a JSON object")
    _, resolve_options = METHODS[resolved["method"]]
    resolved["method_options"] = resolve_options(resolved["method_options"])
    return resolved


def run_training(config, output_dir, device="cpu"):
    config = resolve_config(config)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output folder already exists: {output_dir}")
    device = resolve_device(device)
    train_data = load_training_data(config["train_data"])
    seed_everything(config["seed"])
    model = create_model().to(device)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "config.json", config)
    train_method, _ = METHODS[config["method"]]
    history = train_method(model, train_data, config, output_dir, device)
    write_json(output_dir / "history.json", history)
    checkpoint_path = output_dir / "final.pt"
    save_checkpoint(checkpoint_path, model, config, train_data)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        checkpoint = run_training(config, args.output_dir, args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved final checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
