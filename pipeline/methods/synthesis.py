"""Stage 3: guided OT synthesis from cached, frozen training features.

This is the proposed multiclass feature-space adaptation, not the original
binary OTOS evaluation pipeline. It does not pretrain or train model.fc.
Run with: python -m pipeline.methods.synthesis --config configs/synthesis.json
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


DEFAULT_OPTIONS = {
    "target_count": 500,
    "guide_steps": 500,
    "guide_learning_rate": 0.01,
    "guide_regularization": 0.001,
    "temperature": 1.0,
    "transport_weight": 1.0,
    "guidance_weight": 1.0,
    "entropy_coefficient": 0.05,
    "max_iterations": 100,
    "step_size": 0.1,
    "max_backtracks": 10,
    "projection_iterations": 200,
    "marginal_tolerance": 0.0001,
    "objective_tolerance": 0.00001,
    "acceptance_tolerance": 1e-12,
    "uniform_targets": False,
}


def resolve_options(options):
    if not isinstance(options, dict):
        raise ValueError("method_options must be an object")
    unknown = options.keys() - DEFAULT_OPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown options: {', '.join(sorted(unknown))}")
    result = DEFAULT_OPTIONS | options
    integers = {
        "target_count": 1, "guide_steps": 1, "max_iterations": 1,
        "max_backtracks": 0, "projection_iterations": 1,
    }
    for name, minimum in integers.items():
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    nonnegative = {
        "guide_regularization", "transport_weight", "guidance_weight",
        "entropy_coefficient", "acceptance_tolerance",
    }
    for name in DEFAULT_OPTIONS.keys() - integers.keys() - {"uniform_targets"}:
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (value == 0 and name not in nonnegative):
            raise ValueError(f"Invalid value for {name}")
    if type(result["uniform_targets"]) is not bool:
        raise ValueError("uniform_targets must be boolean")
    return result


def synthetic_count(real_count, target_count):
    return max(0, target_count - real_count)


def guide_loss(weight, positive, negative, regularization):
    return (
        0.5 * regularization * weight.square().sum()
        + 0.5 * torch.relu(1.0 - positive @ weight).mean()
        + 0.5 * torch.relu(1.0 + negative @ weight).mean()
    )


def fit_guide(positive, negative, options):
    if not len(positive) or not len(negative):
        raise ValueError("Guide fitting requires positive and negative examples")
    weight = torch.zeros(
        positive.shape[1], dtype=positive.dtype,
        device=positive.device, requires_grad=True,
    )
    optimizer = torch.optim.Adam([weight], lr=options["guide_learning_rate"])
    for _ in range(options["guide_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = guide_loss(weight, positive, negative, options["guide_regularization"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite guide loss")
        loss.backward()
        optimizer.step()
    weight = weight.detach()
    with torch.no_grad():
        loss = guide_loss(weight, positive, negative, options["guide_regularization"])
    if not torch.isfinite(weight).all() or not torch.isfinite(loss):
        raise RuntimeError("Nonfinite fitted guide")
    return weight, {"loss": loss.item(), "steps": options["guide_steps"],
                    "status": "fixed_steps_completed"}


def target_masses(positive, weight, temperature=1.0, uniform=False):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if uniform:
        return positive.new_full((len(positive),), 1.0 / len(positive))
    losses = torch.relu(1.0 - positive @ weight)
    masses = torch.log_softmax(losses / temperature, dim=0).exp()
    if not torch.isfinite(masses).all() or not (masses > 0).all():
        raise RuntimeError("Target masses underflowed; increase temperature")
    return masses


def sample_prior(positive, count, seed):
    # CPU RNG makes the prior independent of CUDA RNG state.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(count, positive.shape[1], generator=generator,
                        dtype=positive.dtype).to(positive.device)
    return positive.mean(0) + positive.std(0, unbiased=False) * noise


def geometry_cost(prior, positive):
    return (prior[:, None, :] - positive[None, :, :]).square().mean(dim=2)


def barycentric_features(transport, positive):
    return (transport / transport.sum(dim=1, keepdim=True)) @ positive


def negative_score_moments(negative, weight):
    scores = negative @ weight
    mean = scores.mean()
    return mean.detach(), (scores - mean).square().mean().detach()


def guidance_penalty(synthetic, weight, negative_mean, negative_variance):
    scores = synthetic @ weight
    return 0.5 * ((scores - negative_mean - 1.0).square()
                  + negative_variance).mean()


def transport_objective(transport, cost, positive, weight, moments, options):
    objective = options["transport_weight"] * (transport * cost).sum()
    if options["guidance_weight"]:
        synthetic = barycentric_features(transport, positive)
        objective = objective + options["guidance_weight"] * guidance_penalty(
            synthetic, weight, *moments
        )
    if options["entropy_coefficient"]:
        objective = objective + options["entropy_coefficient"] * (
            transport * (transport.log() - 1.0)
        ).sum()
    return objective


def marginal_errors(transport, a, b):
    return ((transport.sum(1) - a).abs().sum().item(),
            (transport.sum(0) - b).abs().sum().item())


def kl_project(log_q, a, b, max_iterations=200, tolerance=1e-4):
    """Scale exp(log_q) itself, not a kernel reconstructed from geometry cost."""
    if log_q.shape != (len(a), len(b)) or not torch.isfinite(log_q).all():
        raise ValueError("Invalid log projection matrix")
    if (not (a > 0).all() or not (b > 0).all()
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise ValueError("Projection masses must be positive and finite")
    if not torch.allclose(a.sum(), b.sum(), rtol=1e-10, atol=1e-12):
        raise ValueError("Projection masses must have equal totals")
    log_q = log_q.detach() - log_q.detach().max()
    log_a, log_b = a.log(), b.log()
    log_v = torch.zeros_like(b)
    with torch.no_grad():
        for iteration in range(1, max_iterations + 1):
            log_u = log_a - torch.logsumexp(log_q + log_v[None, :], dim=1)
            log_v = log_b - torch.logsumexp(log_q + log_u[:, None], dim=0)
            log_t = log_q + log_u[:, None] + log_v[None, :]
            transport = log_t.exp()
            row_error, column_error = marginal_errors(transport, a, b)
            if max(row_error, column_error) <= tolerance:
                if not torch.isfinite(transport).all() or not (transport > 0).all():
                    raise RuntimeError("Projection underflowed or became nonfinite")
                return transport, {
                    "iterations": iteration,
                    "row_l1_error": row_error,
                    "column_l1_error": column_error,
                }
    raise RuntimeError("KL projection did not reach both marginal tolerances")


def optimize_transport(prior, positive, negative, weight, b, options):
    # The frozen guide and data do not participate in the outer autograd graph.
    prior, positive, negative, weight, b = (
        value.detach() for value in (prior, positive, negative, weight, b)
    )
    a = positive.new_full((len(prior),), 1.0 / len(prior))
    cost = geometry_cost(prior, positive)
    moments = negative_score_moments(negative, weight)
    transport = a[:, None] * b[None, :]
    objective_history = []
    stopping_reason = "iteration_limit"
    backtracks = 0
    projection = {"iterations": 0}
    for iteration in range(1, options["max_iterations"] + 1):
        current = transport.detach().requires_grad_(True)
        objective = transport_objective(current, cost, positive, weight, moments, options)
        gradient, = torch.autograd.grad(objective, current)
        old_value = objective.item()
        if not math.isfinite(old_value) or not torch.isfinite(gradient).all():
            raise RuntimeError("Nonfinite transport objective or gradient")
        if not objective_history:
            objective_history.append(old_value)
        step = options["step_size"]
        accepted = False
        last_error = "objective increased"
        for reduction in range(options["max_backtracks"] + 1):
            try:
                candidate, candidate_projection = kl_project(
                    current.detach().log() - step * gradient.detach(), a, b,
                    options["projection_iterations"], options["marginal_tolerance"],
                )
                with torch.no_grad():
                    value = transport_objective(
                        candidate, cost, positive, weight, moments, options
                    ).item()
                allowance = options["acceptance_tolerance"] * max(1.0, abs(old_value))
                if math.isfinite(value) and value <= old_value + allowance:
                    accepted = True
                    projection = candidate_projection
                    break
                last_error = "candidate objective increased or became nonfinite"
            except RuntimeError as error:
                last_error = str(error)
            step *= 0.5
        if not accepted:
            raise RuntimeError(f"No valid projected step at iteration {iteration}: {last_error}")
        backtracks += reduction
        transport = candidate.detach()
        objective_history.append(value)
        relative_change = abs(value - old_value) / max(1.0, abs(old_value))
        if relative_change <= options["objective_tolerance"]:
            stopping_reason = "objective_tolerance"
            break
    row_error, column_error = marginal_errors(transport, a, b)
    return transport, {
        "objective": objective_history[-1],
        "objective_history": objective_history,
        "iterations": iteration,
        "stopping_reason": stopping_reason,
        "backtracks": backtracks,
        "last_step_size": step,
        "last_projection_iterations": projection["iterations"],
        "row_l1_error": row_error,
        "column_l1_error": column_error,
    }


def synthesize_class(positive, negative, options, seed):
    options = resolve_options(options)
    count = synthetic_count(len(positive), options["target_count"])
    if not count:
        return positive.new_empty((0, positive.shape[1])), {"status": "at_target"}
    if not len(positive):
        raise ValueError("Cannot synthesize a class with no real features")
    weight, guide = fit_guide(positive, negative, options)
    b = target_masses(positive, weight, options["temperature"], options["uniform_targets"])
    prior = sample_prior(positive, count, seed)
    transport, diagnostics = optimize_transport(prior, positive, negative, weight, b, options)
    synthetic = barycentric_features(transport, positive).detach()
    diagnostics.update({
        "status": "generated", "guide": guide,
        "target_mass_min": b.min().item(), "target_mass_max": b.max().item(),
        "working_mean_norm": synthetic.norm(dim=1).mean().item(),
        "working_spread": (synthetic - synthetic.mean(0)).square().sum(1).mean().item(),
        "working_mean_distance_to_class_mean": (
            synthetic - positive.mean(0)
        ).norm(dim=1).mean().item(),
    })
    return synthetic, diagnostics


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


def validate_cache(cache):
    features, labels, counts = (cache[key] for key in ("features", "labels", "class_counts"))
    if (features.ndim != 2 or features.shape[1] != 64 or not len(features)
            or not features.is_floating_point() or not torch.isfinite(features).all()):
        raise ValueError("Cache must contain finite floating features with shape (N, 64)")
    if labels.dtype != torch.int64 or labels.shape != (len(features),):
        raise ValueError("Cache labels must be int64 with shape (N,)")
    if counts.dtype != torch.int64 or counts.shape != (100,):
        raise ValueError("Cache class_counts must be int64 with shape (100,)")
    if (labels < 0).any() or (labels >= 100).any():
        raise ValueError("Cache labels must lie in [0, 99]")
    if not torch.equal(torch.bincount(labels, minlength=100), counts):
        raise ValueError("Cache class counts do not match labels")
    scale = cache["feature_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Cache feature_scale must be positive and finite")


def run_generation(config, device="cpu", classes=None, output_dir=None, resume=False):
    allowed = {"feature_cache", "output_dir", "seed", "method_options"}
    if not isinstance(config, dict) or config.keys() - allowed:
        raise ValueError("Invalid generation configuration keys")
    options = resolve_options(config.get("method_options", {}))
    seed = config.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    selected = list(range(100)) if classes is None else sorted(classes)
    if (not selected or len(set(selected)) != len(selected)
            or any(type(c) is not int or not 0 <= c < 100 for c in selected)):
        raise ValueError("classes must be unique class IDs in [0, 99]")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use a cpu or cuda device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    cache_path = Path(config["feature_cache"])
    output = Path(output_dir or config["output_dir"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    validate_cache(cache)
    manifest = {
        "format_version": 1, "feature_cache_sha256": file_sha256(cache_path),
        "seed": seed, "classes": selected, "method_options": options,
    }
    manifest_path = output / "generation_config.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Output already exists: {output}; use --resume or a new folder")
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume requires identical input cache, classes, seed, and options")
    else:
        output.mkdir(parents=True)
        write_json(manifest_path, manifest)
    per_class = output / "per_class"
    per_class.mkdir(exist_ok=True)
    raw = cache["features"]
    labels = cache["labels"].to(device)
    scale = float(cache["feature_scale"])
    # Float64 is deliberate: preserve positive masses and stable log projection.
    working = raw.to(device=device, dtype=torch.float64) / scale
    counts = cache["class_counts"]
    generated_features, generated_labels, summaries = [], [], []
    for class_id in selected:
        real_count = int(counts[class_id])
        count = synthetic_count(real_count, options["target_count"])
        class_path = per_class / f"class_{class_id:03d}.pt"
        if resume and class_path.exists():
            result = torch.load(class_path, map_location="cpu", weights_only=True)
            if (result["features"].shape != (count, 64)
                    or result["labels"].shape != (count,)
                    or not (result["labels"] == class_id).all()
                    or not torch.isfinite(result["features"]).all()):
                raise ValueError(f"Invalid resumed class file: {class_path}")
            print(f"Class {class_id}: restored {count} synthetic features", flush=True)
        else:
            positive = working[labels == class_id]
            negative = working[labels != class_id]
            print(f"Class {class_id}: real={real_count}, generating={count}", flush=True)
            synthetic, diagnostics = synthesize_class(
                positive, negative, options, seed + class_id
            )
            synthetic_raw = (synthetic * scale).to(device="cpu", dtype=raw.dtype)
            if not torch.isfinite(synthetic_raw).all():
                raise RuntimeError("Nonfinite synthetic features after coordinate conversion")
            result = {
                "features": synthetic_raw,
                "labels": torch.full((count,), class_id, dtype=torch.int64),
                "diagnostics": {
                    "class_id": class_id, "real_count": real_count,
                    "synthetic_count": count, **diagnostics,
                },
            }
            atomic_torch_save(result, class_path)
        generated_features.append(result["features"])
        generated_labels.append(result["labels"])
        summaries.append(result["diagnostics"])
    features = torch.cat(generated_features)
    synthetic_labels = torch.cat(generated_labels)
    synthetic_counts = torch.bincount(synthetic_labels, minlength=100)
    expected = torch.zeros_like(counts)
    for class_id in selected:
        expected[class_id] = synthetic_count(int(counts[class_id]), options["target_count"])
    if not torch.equal(synthetic_counts, expected):
        raise RuntimeError("Generated class counts do not match requested counts")
    result = {
        "features": features, "labels": synthetic_labels,
        "real_class_counts": counts, "synthetic_class_counts": synthetic_counts,
        "feature_scale": scale, "generation_config": manifest,
        "data_settings": cache.get("data_settings", {}),
        "checkpoint_sha256": cache.get("checkpoint_sha256", ""),
        "train_data_sha256": cache.get("train_data_sha256", ""),
    }
    atomic_torch_save(result, output / "synthetic_features.pt")
    write_json(output / "generation_summary.json", {
        "real_total": len(raw), "synthetic_total": len(features),
        "combined_total": len(raw) + len(features),
        "all_classes_processed": selected == list(range(100)),
        "classes": summaries,
    })
    print(f"Saved synthetic features {tuple(features.shape)} to {output / 'synthetic_features.pt'}",
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        run_generation(config, args.device, args.classes, args.output_dir, args.resume)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
