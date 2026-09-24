"""Regression tests for evaluation and protocol hardening.

All artifacts in this module are synthetic and temporary.  In particular, no
test-set dataset or cached test prediction is loaded.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.resnet32 import ResNet32
from scripts import evaluation as ev


EXPERTS = ["CE", "LAL", "BalancedSoftmax", "Mixup"]
SEEDS = [78, 88, 1034]


def _state_dict() -> dict[str, torch.Tensor]:
    """Return one compatible synthetic ResNet-32 state dict."""
    return ResNet32(num_classes=100).state_dict()


def _write_checkpoint(
    directory: str | os.PathLike[str],
    expert: str,
    seed: int,
    *,
    state: dict | None = None,
    **metadata,
) -> Path:
    path = Path(directory) / f"{expert}_seed{seed}_final.pt"
    checkpoint = {
        "epoch": 200,
        "seed": seed,
        "expert_name": expert,
        "model_state_dict": _state_dict() if state is None else state,
        "optimiser_state_dict": {},
        "is_final": True,
        "log": {},
    }
    checkpoint.update(metadata)
    torch.save(checkpoint, path)
    return path


def _write_pool(
    directory: str | os.PathLike[str],
    experts: list[str] = EXPERTS,
    seeds: list[int] = SEEDS,
    *,
    missing: set[tuple[str, int]] | None = None,
) -> None:
    missing = missing or set()
    state = _state_dict()
    for expert in experts:
        for seed in seeds:
            if (expert, seed) not in missing:
                _write_checkpoint(directory, expert, seed, state=state)


def _write_invalid_metadata_checkpoint(
    directory: str | os.PathLike[str],
    *,
    remove: set[str] | None = None,
    stored_seed: int | None = None,
    **metadata,
) -> None:
    """Write a two-expert pool with altered metadata on CE."""
    _write_checkpoint(directory, "LAL", 78)
    if stored_seed is not None:
        metadata["seed"] = stored_seed
    checkpoint_seed = metadata.pop("seed", 78)
    path = _write_checkpoint(directory, "CE", 78, **metadata)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["seed"] = checkpoint_seed
    for key in remove or set():
        state.pop(key, None)
    torch.save(state, path)


def _assert_raises(exc_type, function, *args, **kwargs) -> str:
    try:
        function(*args, **kwargs)
    except exc_type as exc:
        return str(exc)
    raise AssertionError(f"expected {exc_type.__name__} from {function.__name__}")


def test_complete_expert_seed_matrix_passes_and_preserves_order():
    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_pool(directory)
        requested = ["Mixup", "CE", "LAL", "BalancedSoftmax"]
        pool = ev.ExpertPool(requested, SEEDS, checkpoint_dir=directory)
        pool.validate_complete()
        pool.load(seed=88)
        assert pool.expert_names == requested
        assert pool.loaded == requested


def test_missing_expert_fails_with_the_missing_matrix_entry():
    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_pool(directory, missing={("Mixup", 78), ("Mixup", 88), ("Mixup", 1034)})
        pool = ev.ExpertPool(EXPERTS, SEEDS, checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.validate_complete)
        assert "Mixup" in message and "seed=78" in message


def test_missing_seed_fails_with_the_missing_matrix_entry():
    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_pool(directory, missing={("CE", 1034)})
        pool = ev.ExpertPool(EXPERTS, SEEDS, checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.validate_complete)
        assert "CE" in message and "seed=1034" in message


def test_multiple_missing_matrix_entries_are_all_reported():
    missing = {("CE", 78), ("LAL", 1034), ("Mixup", 88)}
    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_pool(directory, missing=missing)
        pool = ev.ExpertPool(EXPERTS, SEEDS, checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.validate_complete)
        for expert, seed in sorted(missing):
            assert f"{expert} seed={seed}" in message, message


def test_evaluation_rejects_incomplete_pool_before_loading_the_dataset():
    """The evaluation CLI must fail before its loader is constructed."""
    import scripts.evaluate_experts as evaluate_experts

    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_pool(directory, experts=["CE", "LAL", "Mixup"], seeds=[78],
                    missing={("Mixup", 78)})
        called = []

        def unexpected_loader(*args, **kwargs):
            called.append(True)
            raise AssertionError("test/evaluation loader was accessed")

        original_loader = evaluate_experts.build_test_loader
        evaluate_experts.build_test_loader = unexpected_loader
        try:
            message = _assert_raises(
                ev.EvaluationError,
                evaluate_experts.main,
                [
                    "--checkpoint-dir", directory,
                    "--experts", "CE", "LAL", "Mixup",
                    "--seeds", "78",
                    "--device", "cpu",
                    "--access-log", str(Path(directory) / "access.md"),
                    "--output", str(Path(directory) / "results.json"),
                ],
            )
        finally:
            evaluate_experts.build_test_loader = original_loader
        assert "Mixup" in message and "seed=78" in message
        assert called == []


def test_subset_analysis_rejects_incomplete_pool_before_loading_the_dataset():
    import scripts.analyze_subsets as analyze_subsets

    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_pool(directory, experts=["CE", "LAL", "Mixup"], seeds=[78],
                    missing={("Mixup", 78)})
        called = []

        def unexpected_loader(*args, **kwargs):
            called.append(True)
            raise AssertionError("test/evaluation loader was accessed")

        original_loader = analyze_subsets.build_test_loader
        analyze_subsets.build_test_loader = unexpected_loader
        try:
            message = _assert_raises(
                ev.EvaluationError,
                analyze_subsets.main,
                [
                    "--checkpoint-dir", directory,
                    "--experts", "CE", "LAL", "Mixup",
                    "--seeds", "78",
                    "--device", "cpu",
                    "--access-log", str(Path(directory) / "access.md"),
                    "--output", str(Path(directory) / "results.json"),
                ],
            )
        finally:
            analyze_subsets.build_test_loader = original_loader
        assert "Mixup" in message and "seed=78" in message
        assert called == []


def test_invalid_checkpoint_rejects_evaluation_before_loading_the_dataset():
    """Checkpoint contents must be validated before the test loader is built."""
    import scripts.evaluate_experts as evaluate_experts

    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_invalid_metadata_checkpoint(directory, expert_name="LAL")
        called = []

        def unexpected_loader(*args, **kwargs):
            called.append(True)
            raise AssertionError("test/evaluation loader was accessed")

        original_loader = evaluate_experts.build_test_loader
        evaluate_experts.build_test_loader = unexpected_loader
        try:
            message = _assert_raises(
                ev.EvaluationError,
                evaluate_experts.main,
                [
                    "--checkpoint-dir", directory,
                    "--experts", "CE", "LAL",
                    "--seeds", "78",
                    "--device", "cpu",
                    "--access-log", str(Path(directory) / "access.md"),
                    "--output", str(Path(directory) / "results.json"),
                ],
            )
        finally:
            evaluate_experts.build_test_loader = original_loader
        assert "CE" in message and "seed=78" in message
        assert called == []


def test_invalid_checkpoint_rejects_subset_analysis_before_loading_the_dataset():
    """Subset analysis must use the same checkpoint preflight as evaluation."""
    import scripts.analyze_subsets as analyze_subsets

    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_invalid_metadata_checkpoint(directory, epoch=199)
        called = []

        def unexpected_loader(*args, **kwargs):
            called.append(True)
            raise AssertionError("test/evaluation loader was accessed")

        original_loader = analyze_subsets.build_test_loader
        analyze_subsets.build_test_loader = unexpected_loader
        try:
            message = _assert_raises(
                ev.EvaluationError,
                analyze_subsets.main,
                [
                    "--checkpoint-dir", directory,
                    "--experts", "CE", "LAL",
                    "--seeds", "78",
                    "--device", "cpu",
                    "--access-log", str(Path(directory) / "access.md"),
                    "--output", str(Path(directory) / "results.json"),
                ],
            )
        finally:
            analyze_subsets.build_test_loader = original_loader
        assert "epoch" in message and "CE" in message
        assert called == []


def test_seeds_present_rejects_a_partial_requested_matrix():
    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_pool(directory, experts=["CE", "LAL"], seeds=[78],
                    missing={("LAL", 78)})
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        _assert_raises(ev.EvaluationError, pool.seeds_present)


def test_legacy_load_all_experts_does_not_skip_a_missing_expert():
    from scripts.utils.data import load_all_experts

    with tempfile.TemporaryDirectory(prefix="evaluation_pool_") as directory:
        _write_checkpoint(directory, "CE", 78)
        message = _assert_raises(
            FileNotFoundError,
            load_all_experts,
            ["CE", "LAL"],
            checkpoint_dir=directory,
            device="cpu",
            seed=78,
        )
        assert "LAL" in message


def test_valid_checkpoint_identity_and_final_metadata_loads():
    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        _write_pool(directory, experts=["CE", "LAL"], seeds=[78])
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        pool.load(seed=78)
        assert pool.loaded == ["CE", "LAL"]


def test_checkpoint_with_mismatched_expert_identity_is_rejected():
    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        _write_invalid_metadata_checkpoint(directory, expert_name="LAL")
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.load, seed=78)
        assert "expert_name" in message and "CE" in message


def test_checkpoint_with_mismatched_seed_is_rejected():
    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        _write_invalid_metadata_checkpoint(directory, stored_seed=88)
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.load, seed=78)
        assert "seed" in message and "78" in message


def test_nonfinal_checkpoint_is_rejected_for_canonical_evaluation():
    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        _write_invalid_metadata_checkpoint(directory, is_final=False)
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.load, seed=78)
        assert "is_final" in message


def test_noncanonical_epoch_is_rejected_for_canonical_evaluation():
    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        _write_invalid_metadata_checkpoint(directory, epoch=199)
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.load, seed=78)
        assert "epoch" in message and "200" in message


def test_missing_checkpoint_identity_metadata_is_rejected():
    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        _write_invalid_metadata_checkpoint(directory, remove={"expert_name"})
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.load, seed=78)
        assert "expert_name" in message and "missing" in message


def test_incompatible_checkpoint_model_state_is_rejected():
    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        incompatible = torch.nn.Linear(4, 4).state_dict()
        _write_checkpoint(directory, "CE", 78, state=incompatible)
        _write_checkpoint(directory, "LAL", 78)
        pool = ev.ExpertPool(["CE", "LAL"], [78], checkpoint_dir=directory)
        message = _assert_raises(ev.EvaluationError, pool.load, seed=78)
        assert "incompatible" in message.lower() or "state" in message.lower()


def test_base_trainer_checkpoint_loader_validates_identity_metadata():
    from scripts.base_trainer import BaseTrainer, CheckpointValidationError

    with tempfile.TemporaryDirectory(prefix="evaluation_checkpoint_") as directory:
        trainer = BaseTrainer(
            torch.nn.Linear(2, 2), "CE", device="cpu", seed=78,
            checkpoint_dir=directory,
        )
        path = Path(directory) / "synthetic.pt"
        state = {
            "epoch": 200,
            "seed": 78,
            "expert_name": "CE",
            "model_state_dict": trainer.model.state_dict(),
            "optimiser_state_dict": trainer.optimiser.state_dict(),
            "is_final": True,
        }
        torch.save(state, path)
        trainer.load_checkpoint(str(path))
        state["expert_name"] = "LAL"
        torch.save(state, path)
        _assert_raises(CheckpointValidationError, trainer.load_checkpoint, str(path))


def _healthy_history(epochs: int = 200) -> list[dict]:
    return [
        {
            "epoch": epoch,
            "lr": 0.1 if epoch <= 160 else (0.001 if epoch <= 180 else 1e-5),
            "train_loss": 4.5 / epoch ** 0.5,
            "train_acc": None,
            "grad_norm": 1.0,
            "time_s": 1.0,
        }
        for epoch in range(1, epochs + 1)
    ]


def _write_health_run(
    directory: str | os.PathLike[str],
    expert: str,
    seed: int,
    *,
    history: list[dict] | None = None,
    checkpoint: bool = True,
) -> None:
    path = Path(directory)
    (path / f"{expert}_seed{seed}_history.json").write_text(
        json.dumps(_healthy_history() if history is None else history)
    )
    if checkpoint:
        (path / f"{expert}_seed{seed}_final.pt").write_bytes(b"synthetic checkpoint")


def _run_health_check(directory: str, experts: list[str], seeds: list[int]) -> tuple[int, str]:
    import scripts.check_runs as check_runs

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = check_runs.main([
            "--checkpoint-dir", directory,
            "--experts", *experts,
            "--seeds", *(str(seed) for seed in seeds),
        ])
    return code, output.getvalue()


def test_complete_synthetic_experiment_passes_run_health():
    with tempfile.TemporaryDirectory(prefix="evaluation_health_") as directory:
        for expert in ["CE", "LAL"]:
            for seed in [78, 88]:
                _write_health_run(directory, expert, seed)
        code, output = _run_health_check(directory, ["CE", "LAL"], [78, 88])
        assert code == 0, output
        assert "4 run(s) checked, 0 with problems, 0 not found" in output


def test_missing_history_makes_run_health_fail():
    with tempfile.TemporaryDirectory(prefix="evaluation_health_") as directory:
        _write_health_run(directory, "CE", 78)
        (Path(directory) / "CE_seed78_history.json").unlink()
        code, output = _run_health_check(directory, ["CE"], [78])
        assert code != 0, output
        assert "no history file" in output.lower()


def test_missing_checkpoint_makes_run_health_fail():
    with tempfile.TemporaryDirectory(prefix="evaluation_health_") as directory:
        _write_health_run(directory, "CE", 78, checkpoint=False)
        code, output = _run_health_check(directory, ["CE"], [78])
        assert code != 0, output
        assert "missing final checkpoint" in output.lower()


def test_truncated_history_makes_run_health_fail():
    with tempfile.TemporaryDirectory(prefix="evaluation_health_") as directory:
        _write_health_run(directory, "CE", 78, history=_healthy_history(150))
        code, output = _run_health_check(directory, ["CE"], [78])
        assert code != 0, output
        assert "expected 200" in output


def test_invalid_learning_rate_schedule_makes_run_health_fail():
    history = _healthy_history()
    history[160]["lr"] = 0.1
    with tempfile.TemporaryDirectory(prefix="evaluation_health_") as directory:
        _write_health_run(directory, "CE", 78, history=history)
        code, output = _run_health_check(directory, ["CE"], [78])
        assert code != 0, output
        assert "lr at epoch 161" in output


def test_multiple_run_health_failures_are_all_reported():
    with tempfile.TemporaryDirectory(prefix="evaluation_health_") as directory:
        _write_health_run(directory, "CE", 78)
        (Path(directory) / "CE_seed78_history.json").unlink()
        _write_health_run(directory, "CE", 88, checkpoint=False)
        bad_history = _healthy_history()
        bad_history[160]["lr"] = 0.1
        _write_health_run(directory, "LAL", 78, history=bad_history)
        _write_health_run(directory, "LAL", 88)
        code, output = _run_health_check(directory, ["CE", "LAL"], [78, 88])
        assert code != 0
        assert "CE seed=78" in output
        assert "CE seed=88" in output
        assert "LAL seed=78" in output


def _metric(ba: float, tail: float) -> dict[str, float]:
    return {"ba": ba, "tail": tail}


def _success(candidate: dict[int, dict], uniform: dict[int, dict]) -> bool:
    return ev.passes_success_criterion(candidate, uniform)


def test_success_criterion_passes_only_with_both_metrics_better_per_seed():
    uniform = {seed: _metric(0.50, 0.20) for seed in SEEDS}
    candidate = {seed: _metric(0.60, 0.30) for seed in SEEDS}
    assert _success(candidate, uniform)


def test_positive_mean_ba_does_not_hide_one_worse_seed():
    uniform = {seed: _metric(0.50, 0.20) for seed in SEEDS}
    candidate = {
        78: _metric(0.70, 0.30),
        88: _metric(0.70, 0.30),
        1034: _metric(0.40, 0.30),
    }
    assert not _success(candidate, uniform)


def test_positive_mean_tail_does_not_hide_one_worse_seed():
    uniform = {seed: _metric(0.60, 0.20) for seed in SEEDS}
    candidate = {
        78: _metric(0.70, 0.30),
        88: _metric(0.70, 0.30),
        1034: _metric(0.70, 0.10),
    }
    assert not _success(candidate, uniform)


def test_one_metric_worsening_rejects_the_candidate():
    uniform = {seed: _metric(0.50, 0.20) for seed in SEEDS}
    candidate = {seed: _metric(0.60, 0.10) for seed in SEEDS}
    assert not _success(candidate, uniform)


def test_equal_performance_is_not_an_improvement():
    uniform = {seed: _metric(0.50, 0.20) for seed in SEEDS}
    candidate = {seed: _metric(0.50, 0.20) for seed in SEEDS}
    assert not _success(candidate, uniform)


def test_success_criterion_rejects_mismatched_seed_sets():
    uniform = {seed: _metric(0.50, 0.20) for seed in SEEDS}
    candidate = {seed: _metric(0.60, 0.30) for seed in SEEDS[:2]}
    message = _assert_raises(ev.EvaluationError, _success, candidate, uniform)
    assert "seed" in message.lower()


def test_aggregation_rejects_missing_metric_keys():
    message = _assert_raises(
        ev.EvaluationError,
        ev.aggregate_across_seeds,
        [{"ba": 0.4, "tail": 0.1}, {"ba": 0.5}],
    )
    assert "metric" in message.lower() or "key" in message.lower()


def test_aggregation_rejects_nonfinite_metric_values():
    message = _assert_raises(
        ev.EvaluationError,
        ev.aggregate_across_seeds,
        [{"ba": 0.4, "tail": 0.1}, {"ba": float("nan"), "tail": 0.2}],
    )
    assert "finite" in message.lower()


def test_named_aggregation_rejects_inconsistent_expert_membership():
    per_seed = {
        78: {"CE": _metric(0.4, 0.1), "LAL": _metric(0.5, 0.2)},
        88: {"CE": _metric(0.4, 0.1)},
    }
    message = _assert_raises(ev.EvaluationError, ev.aggregate_named_metrics, per_seed)
    assert "membership" in message.lower() or "expert" in message.lower()


def test_successful_test_access_authorization_records_one_entry():
    from scripts.utils.test_access import TestAccessLog

    with tempfile.TemporaryDirectory(prefix="evaluation_access_") as directory:
        log = TestAccessLog(Path(directory) / "access.md")
        log.authorize("synthetic evaluation", note="synthetic")
        assert len(log.entries()) == 1


def test_failed_test_access_authorization_is_a_clear_error():
    from scripts.utils.test_access import TestAccessLog

    message = _assert_raises(
        RuntimeError,
        TestAccessLog("/proc/nonexistent_dir/access.md").authorize,
        "synthetic evaluation",
    )
    assert "test-set" in message.lower() and "log" in message.lower()


def test_protected_test_loader_requires_explicit_authorization():
    import scripts.evaluate_experts as evaluate_experts
    from scripts.utils.test_access import TestAccessError

    class ExplodingDataset:
        def __init__(self, *args, **kwargs):
            raise AssertionError("test dataset was constructed")

    original_dataset = evaluate_experts.LongTailCIFAR100
    evaluate_experts.LongTailCIFAR100 = ExplodingDataset
    try:
        message = _assert_raises(
            TestAccessError,
            evaluate_experts.build_test_loader,
            tempfile.mkdtemp(prefix="evaluation_access_loader_"),
            authorization=None,
        )
    finally:
        evaluate_experts.LongTailCIFAR100 = original_dataset
    assert "authoriz" in message.lower()


def test_authorized_test_loader_uses_one_scoped_grant():
    import scripts.evaluate_experts as evaluate_experts
    from scripts.utils.test_access import TestAccessLog

    class SyntheticTestDataset:
        def __init__(self, *args, **kwargs):
            self.targets = [0, 1]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), self.targets[index]

    with tempfile.TemporaryDirectory(prefix="evaluation_access_loader_") as directory:
        log = TestAccessLog(Path(directory) / "access.md")
        grant = log.authorize("synthetic evaluation", note="loader")
        original_dataset = evaluate_experts.LongTailCIFAR100
        evaluate_experts.LongTailCIFAR100 = SyntheticTestDataset
        try:
            loader = evaluate_experts.build_test_loader(
                directory, batch_size=2, authorization=grant,
            )
        finally:
            evaluate_experts.LongTailCIFAR100 = original_dataset
        assert len(loader.dataset) == 2
        assert len(log.entries()) == 1


def test_test_loader_grant_cannot_be_reused_or_duplicate_the_log():
    import scripts.evaluate_experts as evaluate_experts
    from scripts.utils.test_access import TestAccessError, TestAccessLog

    class SyntheticTestDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), 0

    with tempfile.TemporaryDirectory(prefix="evaluation_access_loader_") as directory:
        log = TestAccessLog(Path(directory) / "access.md")
        grant = log.authorize("synthetic evaluation", note="loader")
        original_dataset = evaluate_experts.LongTailCIFAR100
        evaluate_experts.LongTailCIFAR100 = SyntheticTestDataset
        try:
            evaluate_experts.build_test_loader(
                directory, batch_size=1, authorization=grant,
            )
            message = _assert_raises(
                TestAccessError,
                evaluate_experts.build_test_loader,
                directory,
                batch_size=1,
                authorization=grant,
            )
        finally:
            evaluate_experts.LongTailCIFAR100 = original_dataset
        assert "grant" in message.lower() or "authoriz" in message.lower()
        assert len(log.entries()) == 1


def test_failed_evaluation_logging_prevents_dataset_access():
    import scripts.evaluate_experts as evaluate_experts

    with tempfile.TemporaryDirectory(prefix="evaluation_access_") as directory:
        _write_pool(directory, experts=["CE", "LAL"], seeds=[78])
        called = []

        def unexpected_loader(*args, **kwargs):
            called.append(True)
            raise AssertionError("dataset access was not prevented")

        original_loader = evaluate_experts.build_test_loader
        evaluate_experts.build_test_loader = unexpected_loader
        try:
            message = _assert_raises(
                RuntimeError,
                evaluate_experts.main,
                [
                    "--checkpoint-dir", directory,
                    "--experts", "CE", "LAL",
                    "--seeds", "78",
                    "--device", "cpu",
                    "--access-log", "/proc/nonexistent_dir/access.md",
                    "--output", str(Path(directory) / "results.json"),
                ],
            )
        finally:
            evaluate_experts.build_test_loader = original_loader
        assert "test-set" in message.lower()
        assert called == []


def test_legacy_loader_cannot_read_test_data_when_logging_fails():
    import scripts.utils.data as data_utils

    class ExplodingDataset:
        def __init__(self, *args, **kwargs):
            raise AssertionError("test dataset was constructed")

    original_dataset = data_utils.LongTailCIFAR100
    data_utils.LongTailCIFAR100 = ExplodingDataset
    try:
        message = _assert_raises(
            RuntimeError,
            data_utils.create_cifar_loader,
            "test",
            data_root=tempfile.mkdtemp(prefix="evaluation_access_data_"),
            num_workers=0,
            pin_memory=False,
            test_access_log="/proc/nonexistent_dir/access.md",
        )
    finally:
        data_utils.LongTailCIFAR100 = original_dataset
    assert "test-set" in message.lower()


def test_legacy_test_loader_consumes_its_access_grant():
    """The legacy loader must use the scoped authorization it requests."""
    import scripts.utils.data as data_utils

    class TrackingGrant:
        def __init__(self):
            self.consumed = False

        def consume(self):
            assert not self.consumed
            self.consumed = True

    class TrackingLog:
        grant = TrackingGrant()

        def __init__(self, _path):
            pass

        def authorize(self, *_args, **_kwargs):
            return self.grant

    class SyntheticTestDataset:
        def __init__(self, *args, **kwargs):
            self.targets = [0, 1]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), self.targets[index]

        def get_class_counts(self):
            return np.array([1, 1], dtype=np.int64)

    original_log = data_utils.TestAccessLog
    original_dataset = data_utils.LongTailCIFAR100
    data_utils.TestAccessLog = TrackingLog
    data_utils.LongTailCIFAR100 = SyntheticTestDataset
    try:
        data_utils.create_cifar_loader(
            "test", data_root=tempfile.mkdtemp(prefix="evaluation_access_data_"),
            num_workers=0, pin_memory=False,
        )
    finally:
        data_utils.TestAccessLog = original_log
        data_utils.LongTailCIFAR100 = original_dataset
    assert TrackingLog.grant.consumed


def test_training_loader_does_not_trigger_test_access_logging():
    import scripts.utils.data as data_utils

    class SyntheticTrainDataset:
        def __init__(self, *args, **kwargs):
            self.transform = None

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), index % 2

        def get_class_counts(self):
            return np.array([1, 1], dtype=np.int64)

    with tempfile.TemporaryDirectory(prefix="evaluation_access_") as directory:
        log_path = Path(directory) / "access.md"
        original_dataset = data_utils.LongTailCIFAR100
        original_indices = data_utils.load_lt_train_indices
        data_utils.LongTailCIFAR100 = SyntheticTrainDataset
        data_utils.load_lt_train_indices = lambda root: np.array([0, 1])
        try:
            data_utils.create_cifar_loader(
                "train", data_root=directory, num_workers=0,
                pin_memory=False, test_access_log=log_path,
            )
        finally:
            data_utils.LongTailCIFAR100 = original_dataset
            data_utils.load_lt_train_indices = original_indices
        assert not log_path.exists()


def test_legacy_dace_reader_declares_the_central_guard():
    source = (Path(_PROJECT_ROOT) / "scripts" / "evaluate_dace.py").read_text()
    assert "TestAccessLog" in source and ".authorize(" in source


TESTS = [
    ("complete pool preserves order", test_complete_expert_seed_matrix_passes_and_preserves_order),
    ("missing expert", test_missing_expert_fails_with_the_missing_matrix_entry),
    ("missing seed", test_missing_seed_fails_with_the_missing_matrix_entry),
    ("multiple missing combinations", test_multiple_missing_matrix_entries_are_all_reported),
    ("pool validation precedes dataset", test_evaluation_rejects_incomplete_pool_before_loading_the_dataset),
    ("subset validation precedes dataset", test_subset_analysis_rejects_incomplete_pool_before_loading_the_dataset),
    ("invalid checkpoint precedes evaluation dataset", test_invalid_checkpoint_rejects_evaluation_before_loading_the_dataset),
    ("invalid checkpoint precedes subset dataset", test_invalid_checkpoint_rejects_subset_analysis_before_loading_the_dataset),
    ("partial seeds rejected", test_seeds_present_rejects_a_partial_requested_matrix),
    ("legacy pool missing expert", test_legacy_load_all_experts_does_not_skip_a_missing_expert),
    ("valid checkpoint metadata", test_valid_checkpoint_identity_and_final_metadata_loads),
    ("mismatched checkpoint expert", test_checkpoint_with_mismatched_expert_identity_is_rejected),
    ("mismatched checkpoint seed", test_checkpoint_with_mismatched_seed_is_rejected),
    ("nonfinal checkpoint", test_nonfinal_checkpoint_is_rejected_for_canonical_evaluation),
    ("incorrect checkpoint epoch", test_noncanonical_epoch_is_rejected_for_canonical_evaluation),
    ("missing checkpoint metadata", test_missing_checkpoint_identity_metadata_is_rejected),
    ("incompatible model state", test_incompatible_checkpoint_model_state_is_rejected),
    ("base trainer checkpoint metadata", test_base_trainer_checkpoint_loader_validates_identity_metadata),
    ("complete run health", test_complete_synthetic_experiment_passes_run_health),
    ("missing history health failure", test_missing_history_makes_run_health_fail),
    ("missing checkpoint health failure", test_missing_checkpoint_makes_run_health_fail),
    ("truncated history health failure", test_truncated_history_makes_run_health_fail),
    ("invalid LR health failure", test_invalid_learning_rate_schedule_makes_run_health_fail),
    ("multiple health failures", test_multiple_run_health_failures_are_all_reported),
    ("success criterion pass", test_success_criterion_passes_only_with_both_metrics_better_per_seed),
    ("success criterion BA consistency", test_positive_mean_ba_does_not_hide_one_worse_seed),
    ("success criterion Tail consistency", test_positive_mean_tail_does_not_hide_one_worse_seed),
    ("success criterion both metrics", test_one_metric_worsening_rejects_the_candidate),
    ("success criterion strict equality", test_equal_performance_is_not_an_improvement),
    ("success criterion seed pairing", test_success_criterion_rejects_mismatched_seed_sets),
    ("aggregation metric keys", test_aggregation_rejects_missing_metric_keys),
    ("aggregation finite values", test_aggregation_rejects_nonfinite_metric_values),
    ("aggregation membership", test_named_aggregation_rejects_inconsistent_expert_membership),
    ("successful access authorization", test_successful_test_access_authorization_records_one_entry),
    ("failed access authorization", test_failed_test_access_authorization_is_a_clear_error),
    ("loader requires authorization", test_protected_test_loader_requires_explicit_authorization),
    ("authorized loader grant", test_authorized_test_loader_uses_one_scoped_grant),
    ("loader grant is single-use", test_test_loader_grant_cannot_be_reused_or_duplicate_the_log),
    ("failed logging blocks evaluation", test_failed_evaluation_logging_prevents_dataset_access),
    ("legacy loader guard", test_legacy_loader_cannot_read_test_data_when_logging_fails),
    ("training loader avoids test guard", test_training_loader_does_not_trigger_test_access_logging),
    ("legacy loader consumes access grant", test_legacy_test_loader_consumes_its_access_grant),
    ("legacy DACE guard", test_legacy_dace_reader_declares_the_central_guard),
]


def main() -> int:
    passed = failed = 0
    for name, test in TESTS:
        try:
            test()
            passed += 1
            print(f"  ✅ {name}")
        except Exception as exc:  # noqa: BLE001 - standalone test harness
            failed += 1
            print(f"  ❌ {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS)} tests: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
