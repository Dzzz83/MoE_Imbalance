"""Matched classifier continuations from a saved baseline."""

import torch

from pipeline.methods import baseline, prototype_weights, reweighting, sava_weights


def prepare_sava_options(options):
    shared = {name: value for name, value in options.items() if name != "sava"}
    return _prepare_options(shared, use_ot=False) | {
        "sava": sava_weights.prepare_options(options.get("sava", {})),
    }


def train_sava(model, train_data, config, output_dir, device):
    return _train(model, train_data, config, output_dir, device,
                  use_ot=False, weight_builder=sava_weights.build_weights)


def _prepare_options(options, *, use_ot):
    allowed = {"pretrained_checkpoint", "classifier"}
    if use_ot:
        allowed |= reweighting.DEFAULT_OPTIONS.keys() - {"pretraining"}
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown fine-tuning options: {', '.join(sorted(unknown))}")
    checkpoint_path = options.get("pretrained_checkpoint")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError("pretrained_checkpoint must be a baseline checkpoint file path")
    if use_ot:
        method_options = reweighting.prepare_method_options({
            name: value for name, value in options.items() if name != "pretrained_checkpoint"
        })
        del method_options["pretraining"]
    else:
        method_options = {"classifier": reweighting.prepare_classifier_options(options.get("classifier", {}))}
    return method_options | {"pretrained_checkpoint": checkpoint_path}


def prepare_uniform_options(options):
    return _prepare_options(options, use_ot=False)


def prepare_ot_options(options):
    return _prepare_options(options, use_ot=True)


def prepare_prototype_options(options):
    shared_names = {"pretrained_checkpoint", "classifier"}
    shared_options = {name: value for name, value in options.items() if name in shared_names}
    weighting_options = {name: value for name, value in options.items() if name not in shared_names}
    return (
        _prepare_options(shared_options, use_ot=False)
        | prototype_weights.prepare_options(weighting_options)
    )


def _load_pretrained_baseline(model, train_data, path):
    """Restore all model state and check the source's method and pool metadata.

    This starts a new optimizer, not a resume. The caller must reuse the same
    prepared pool; counts and preparation settings cannot identify image content.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("pretrained_checkpoint must contain a baseline checkpoint")
    source_config = checkpoint.get("config")
    if not isinstance(source_config, dict) or source_config.get("method") != "baseline":
        raise ValueError("pretrained_checkpoint must come from method='baseline'")
    class_counts = checkpoint.get("class_counts")
    if not isinstance(class_counts, torch.Tensor) or not torch.equal(
        class_counts, torch.from_numpy(train_data.class_counts)
    ):
        raise ValueError("Pretrained checkpoint class counts do not match the retained pool")
    if checkpoint.get("data_settings") != train_data.data_settings:
        raise ValueError("Pretrained checkpoint preparation settings do not match the retained pool")
    source_options = source_config.get("method_options")
    if not isinstance(source_options, dict):
        raise ValueError("Pretrained baseline checkpoint is missing method_options")
    source_options = baseline.prepare_method_options(source_options)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Pretrained baseline checkpoint is missing model_state_dict")
    model.load_state_dict(state)
    return source_options["epochs"]


def _train(model, train_data, config, output_dir, device, *, use_ot, weight_builder=None):
    if use_ot and (train_data.class_counts == 0).any():
        raise ValueError("OT fine-tuning requires retained training examples in every class")
    checkpoint_path = config["method_options"]["pretrained_checkpoint"]
    epoch_offset = _load_pretrained_baseline(model, train_data, checkpoint_path)
    print(f"Loaded baseline after {epoch_offset} epochs from {checkpoint_path}", flush=True)
    return reweighting.train_classifier(
        model, train_data, config, output_dir, device,
        epoch_offset=epoch_offset, use_ot=use_ot, weight_builder=weight_builder,
    )


def train_uniform(model, train_data, config, output_dir, device):
    return _train(model, train_data, config, output_dir, device, use_ot=False)


def train_ot(model, train_data, config, output_dir, device):
    return _train(model, train_data, config, output_dir, device, use_ot=True)


def train_prototype(model, train_data, config, output_dir, device):
    if (train_data.class_counts == 0).any():
        raise ValueError("Prototype fine-tuning requires retained training examples in every class")
    return _train(
        model, train_data, config, output_dir, device,
        use_ot=False, weight_builder=prototype_weights.build_weights,
    )
