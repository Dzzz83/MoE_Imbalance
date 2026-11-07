"""Measure one full-pool OT update on synthetic features, without image training."""

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from pipeline.data import NUM_CLASSES
from pipeline.methods import reweighting
from pipeline.runtime import resolve_device, seed_everything
from pipeline.train import resolve_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reweighting.json"))
    parser.add_argument("--examples", type=int, default=10_847)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.examples < NUM_CLASSES:
        parser.error("--examples must be at least 100")
    config = resolve_config(json.loads(args.config.read_text()))
    if config["method"] != "reweighting":
        parser.error("--config must select reweighting")
    options = config["method_options"]
    device = resolve_device(args.device)
    seed_everything(config["seed"])
    torch.set_num_threads(1)
    labels = torch.arange(args.examples) % NUM_CLASSES
    raw_features = torch.randn(args.examples, 64).relu()
    features, references, masses = reweighting._build_ot_references(raw_features, labels)
    cost = reweighting._build_ot_cost(
        features.to(device), references.to(device), labels.to(device),
        feature_cost_scale=options["feature_cost_scale"], label_mismatch_cost=options["label_mismatch_cost"],
    )
    scores = reweighting._initialize_example_scores(args.examples, device, dtype=cost.dtype)
    optimizer = reweighting._make_score_optimizer(scores, options["score_learning_rate"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    weights, diagnostics = reweighting._update_example_scores(
        scores, optimizer, cost, masses.to(device), score_updates=1,
        epsilon=options["epsilon"], max_iterations=options["sinkhorn_iterations"],
        marginal_tolerance=options["marginal_tolerance"],
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(json.dumps({
        "device": str(device), "examples": args.examples, "references": NUM_CLASSES,
        "update_seconds": time.perf_counter() - started,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None,
        "effective_sample_size": reweighting._effective_sample_size(weights),
        **diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
