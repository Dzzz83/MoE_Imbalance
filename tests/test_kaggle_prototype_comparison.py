import json
from pathlib import Path

import pytest

from scripts import kaggle_prototype_comparison as kaggle


def test_completed_run_is_reused_and_configuration_changes_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "STATE_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "CODE_ROOT", tmp_path / "code")
    # Keep the temporary attempt beside the result directory, as on Kaggle.
    create_directory = kaggle.tempfile.mkdtemp
    monkeypatch.setattr(
        kaggle.tempfile, "mkdtemp",
        lambda prefix, dir: create_directory(prefix=prefix, dir=tmp_path),
    )
    training_calls = []
    pushes = []

    def train(command, **kwargs):
        training_calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "final.pt").write_bytes(b"checkpoint")

    monkeypatch.setattr(kaggle, "run", train)
    monkeypatch.setattr(kaggle, "push_state", lambda message, auth: pushes.append(message))
    config = {"method": "uniform_finetuning"}
    first = kaggle.train_run("uniform_seed0", config, {}, {})
    second = kaggle.train_run("uniform_seed0", config, {}, {})
    assert first == second
    assert len(training_calls) == 1
    assert len(pushes) == 2
    assert json.loads((kaggle.STATE_ROOT / "configs/uniform_seed0.json").read_text()) == config
    with pytest.raises(RuntimeError, match="configuration differs"):
        kaggle.train_run("uniform_seed0", {"method": "changed"}, {}, {})


def test_push_retries_even_when_commit_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "STATE_ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(kaggle, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(
        kaggle.subprocess, "run",
        lambda *args, **kwargs: kaggle.subprocess.CompletedProcess(args, 0),
    )
    kaggle.push_state("retry", {})
    assert [command[1] for command in commands] == ["add", "push"]


def test_remaining_seeds_use_matched_baselines_and_preserve_seed_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "SAVA_ONLY", False)
    monkeypatch.setattr(kaggle, "STATE_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "CODE_ROOT", tmp_path / "code")
    monkeypatch.setattr(kaggle, "BASELINE_CHECKPOINTS", {})
    configs = kaggle.CODE_ROOT / "configs"
    kaggle.write_json(configs / "baseline.json", {"seed": 0})
    for mode in kaggle.MODES:
        kaggle.write_json(configs / f"{mode}_finetuning.json", {
            "seed": 0, "method_options": {},
        })
    kaggle.write_json(kaggle.STATE_ROOT / "comparison.json", {"seed_zero": True})
    kaggle.write_json(kaggle.STATE_ROOT / "experiment.json", {"seed": 0})
    calls = []

    def train(name, config, auth_env, training_env):
        calls.append((name, config))
        output = kaggle.STATE_ROOT / "runs" / name
        output.mkdir(parents=True)
        (output / "final.pt").write_bytes(b"checkpoint")
        return output

    def evaluate(command, **kwargs):
        assert command[2] == "pipeline.evaluate"
        output = Path(command[command.index("--output") + 1])
        kaggle.write_json(output, {"accuracy": 0.4})

    monkeypatch.setattr(kaggle, "train_run", train)
    monkeypatch.setattr(kaggle, "run", evaluate)
    monkeypatch.setattr(kaggle, "push_state", lambda *args: None)
    temporary_directory = kaggle.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        kaggle.tempfile, "TemporaryDirectory",
        lambda **kwargs: temporary_directory(dir=tmp_path),
    )
    for seed in (1, 2):
        kaggle.run_comparison(seed, tmp_path / "train.npz", tmp_path / "raw", {}, {})
        seed_calls = calls[-5:]
        assert [name for name, _ in seed_calls] == [
            f"{mode}_seed{seed}" for mode in ("baseline", *kaggle.MODES)
        ]
        assert all(config["seed"] == seed for _, config in seed_calls)
        for _, config in seed_calls[1:]:
            assert config["method_options"]["pretrained_checkpoint"] == str(
                kaggle.STATE_ROOT / "runs" / f"baseline_seed{seed}" / "final.pt"
            )
        manifest = json.loads((kaggle.STATE_ROOT / f"experiment_seed{seed}.json").read_text())
        assert manifest["seed"] == seed
        assert manifest["data_seed"] == 42
        summary = json.loads((kaggle.STATE_ROOT / f"comparison_seed{seed}.json").read_text())
        assert set(summary) == {"baseline", *kaggle.MODES}
    assert json.loads((kaggle.STATE_ROOT / "comparison.json").read_text()) == {"seed_zero": True}
    assert json.loads((kaggle.STATE_ROOT / "experiment.json").read_text()) == {"seed": 0}


def test_sava_only_reuses_completed_runs_and_preserves_previous_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "STATE_ROOT", tmp_path / "results")
    monkeypatch.setattr(kaggle, "CODE_ROOT", tmp_path / "code")
    monkeypatch.setattr(kaggle, "SAVA_ONLY", True)
    monkeypatch.setattr(kaggle, "BASELINE_CHECKPOINTS", {})
    monkeypatch.setattr(kaggle, "push_state", lambda *args: None)

    def unexpected_command(*args, **kwargs):
        pytest.fail("Completed runs must not train or evaluate again")

    monkeypatch.setattr(kaggle, "run", unexpected_command)
    prepared = tmp_path / "train.npz"
    baseline = dict(train_data=str(prepared), seed=1, num_workers=2)
    sava = baseline | {"method_options": {
        "pretrained_checkpoint": str(kaggle.STATE_ROOT / "runs/baseline_seed1/final.pt"),
    }}
    kaggle.write_json(kaggle.CODE_ROOT / "configs/baseline.json", {})
    kaggle.write_json(kaggle.CODE_ROOT / "configs/sava_finetuning.json", {"method_options": {}})
    for name, config in (("baseline_seed1", baseline), ("sava_seed1", sava)):
        kaggle.write_json(kaggle.STATE_ROOT / "configs" / f"{name}.json", config)
        output = kaggle.STATE_ROOT / "runs" / name
        kaggle.write_json(output / "metrics.json", {"accuracy": 0.4})
        (output / "final.pt").write_bytes(b"checkpoint")
    previous = kaggle.STATE_ROOT / "comparison_seed1.json"
    kaggle.write_json(previous, {"original": True})
    kaggle.run_comparison(1, prepared, tmp_path / "raw", {}, {})
    assert json.loads(previous.read_text()) == {"original": True}
    summary = json.loads((kaggle.STATE_ROOT / "comparison_sava_seed1.json").read_text())
    assert set(summary) == {"baseline", "sava"}
    manifest = json.loads((kaggle.STATE_ROOT / "experiment_sava_seed1.json").read_text())
    assert manifest["modes"] == ["sava"]
