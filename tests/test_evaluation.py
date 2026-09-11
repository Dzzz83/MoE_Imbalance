"""
Tests for the evaluation harness: metrics, expert-pool loading, routing
headroom, run-health checking, and the test-set access log.

These are the tools used to verify the Kaggle runs and to measure how much
headroom routing has, **before** any routing rule is compared on the test set.
"""

import json
import os
import sys
import tempfile

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch

import importlib

# Guarded so a not-yet-written module fails one test at a time, not the file.
def _safe(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


ev = _safe('scripts.evaluation')
_ta = _safe('scripts.utils.test_access')
TestAccessLog = getattr(_ta, 'TestAccessLog', None)

EXPERT_LABELS = ['CE', 'LAL', 'BalancedSoftmax', 'Mixup']


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_balanced_accuracy_is_mean_per_class_recall():
    targets = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 0, 1, 0, 2, 2])   # class 1 recall 0.5, others 1.0
    ba = ev.balanced_accuracy(targets, preds)
    assert abs(ba - (1.0 + 0.5 + 1.0) / 3) < 1e-9, ba
    print(f"  ✅ balanced accuracy = {ba:.4f}")


def test_group_accuracies_splits_head_med_tail():
    from scripts.base_trainer import compute_class_groups
    counts = np.array([500] * 30 + [50] * 36 + [5] * 34, dtype=np.int64)
    groups = compute_class_groups(counts)
    assert len(groups['head']) == 30 and len(groups['medium']) == 36
    assert len(groups['tail']) == 34, len(groups['tail'])
    targets = np.array([0, 30, 99])
    preds = np.array([0, 30, 0])          # tail class wrong
    acc = ev.group_accuracies(targets, preds, groups)
    assert acc['head'] == 1.0 and acc['tail'] == 0.0, acc
    print(f"  ✅ group accuracies: {acc}")


def test_ece_is_zero_for_perfectly_calibrated_binary_confidences():
    # 100 samples: 70 correct at confidence 0.7, 30 wrong at confidence 0.3
    targets = np.array([1] * 70 + [0] * 30)
    probs = np.stack([np.full(100, 0.3), np.full(100, 0.7)], axis=1)
    ece = ev.expected_calibration_error(probs, targets, n_bins=10)
    assert ece < 1e-6, f"ECE {ece} should be ~0 for a calibrated predictor"
    print(f"  ✅ ECE of calibrated predictor = {ece:.6f}")


def test_ece_is_high_for_overconfident_predictor():
    targets = np.array([1] * 50 + [0] * 50)          # 50% accuracy
    probs = np.stack([np.full(100, 0.01), np.full(100, 0.99)], axis=1)
    ece = ev.expected_calibration_error(probs, targets, n_bins=10)
    assert ece > 0.4, f"ECE {ece} should be high for an overconfident predictor"
    print(f"  ✅ ECE of overconfident predictor = {ece:.4f}")


def test_evaluate_predictions_reports_all_groups():
    counts = np.array([500] * 30 + [50] * 36 + [5] * 34, dtype=np.int64)
    rng = np.random.default_rng(0)
    n, c = 200, 100
    targets = rng.integers(0, c, n)
    probs = rng.dirichlet(np.ones(c), size=n)
    preds = probs.argmax(axis=1)
    out = ev.evaluate_predictions(targets, preds, probs, counts)
    for key in ('ba', 'head', 'medium', 'tail', 'ece', 'n'):
        assert key in out, f"missing metric '{key}': {sorted(out)}"
    print(f"  ✅ evaluate_predictions keys: {sorted(out)}")


# ---------------------------------------------------------------------------
# Expert pool loading
# ---------------------------------------------------------------------------

def _fake_checkpoint(directory, label, seed, epoch=200):
    from models.resnet32 import ResNet32
    path = os.path.join(directory, f'{label}_seed{seed}_final.pt')
    torch.save({
        'epoch': epoch,
        'seed': seed,
        'expert_name': label,
        'model_state_dict': ResNet32(num_classes=100).state_dict(),
        'optimiser_state_dict': {},
        'is_final': True,
        'log': {},
    }, path)
    return path


def test_expert_pool_finds_checkpoints_by_naming_convention():
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    for label in EXPERT_LABELS:
        _fake_checkpoint(d, label, 78)
    pool = ev.ExpertPool(EXPERT_LABELS, seeds=[78], checkpoint_dir=d)
    found = pool.available()
    assert set(found) == set(EXPERT_LABELS), f"found {sorted(found)}"
    print(f"  ✅ ExpertPool located {len(found)} checkpoints")


def test_expert_pool_reports_missing_checkpoints():
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    _fake_checkpoint(d, 'CE', 78)
    pool = ev.ExpertPool(['CE', 'LAL'], seeds=[78], checkpoint_dir=d)
    found = pool.available()
    assert 'CE' in found and 'LAL' not in found, found
    print("  ✅ ExpertPool reports which checkpoints are missing")


def test_expert_pool_rejects_duplicate_seed_only_pool():
    """A pool with fewer than two distinct experts cannot be routed over."""
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    _fake_checkpoint(d, 'CE', 78)
    pool = ev.ExpertPool(['CE'], seeds=[78], checkpoint_dir=d)
    try:
        pool.load()
    except ev.EvaluationError as e:
        print(f"  ✅ single-expert pool rejected: {str(e)[:60]}")
        return
    raise AssertionError("a 1-expert pool was accepted for routing")


def test_expert_pool_logits_shape():
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    for label in ['CE', 'LAL']:
        _fake_checkpoint(d, label, 78)
    pool = ev.ExpertPool(['CE', 'LAL'], seeds=[78], checkpoint_dir=d, device='cpu')
    pool.load()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = pool.logits_from_batch(x)
    # (N, E, C) — the same convention the routers use
    assert out.shape == (4, 2, 100), f"expected (4,2,100), got {out.shape}"
    print(f"  ✅ stacked logits shape {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# Routing headroom
# ---------------------------------------------------------------------------

def test_headroom_all_wrong_and_oracle():
    # 3 experts, 4 samples:
    #  s0: e0 correct, e1 wrong, e2 wrong  -> savable
    #  s1: all correct
    #  s2: all wrong                       -> unrecoverable
    #  s3: e2 correct only                 -> savable
    targets = np.array([0, 1, 2, 3])
    logits = np.zeros((4, 3, 5), dtype=np.float32)
    # experts that are correct on each sample
    correct_experts = [{0}, {0, 1, 2}, set(), {2}]
    for i, correct in enumerate(correct_experts):
        for e in range(3):
            cls = targets[i] if e in correct else (targets[i] + 1) % 5
            logits[i, e, cls] = 10.0

    h = ev.HeadroomAnalyzer(logits, targets)
    assert abs(h.all_wrong_fraction() - 0.25) < 1e-9, h.all_wrong_fraction()
    assert abs(h.oracle_accuracy() - 0.75) < 1e-9, h.oracle_accuracy()
    counts = h.correctness_counts()
    assert counts[0] == 1 and counts[1] == 2 and counts[3] == 1, counts
    print(f"  ✅ all-wrong {h.all_wrong_fraction():.2f}, oracle {h.oracle_accuracy():.2f}")


def test_headroom_kappa_is_one_for_identical_experts():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 3, 10)).astype(np.float32)
    logits[:, 1] = logits[:, 0]      # experts 0 and 1 are identical
    logits[:, 2] = rng.normal(size=(50, 10))
    targets = rng.integers(0, 10, 50)
    h = ev.HeadroomAnalyzer(logits, targets)
    k = h.pairwise_kappa()
    assert abs(k[('E0', 'E1')] - 1.0) < 1e-9, k
    assert abs(k[('E0', 'E2')]) < 0.9, k
    print(f"  ✅ identical experts give kappa 1.0: {k}")


def test_headroom_names_experts():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(10, 2, 5)).astype(np.float32)
    targets = rng.integers(0, 5, 10)
    h = ev.HeadroomAnalyzer(logits, targets, expert_names=['CE', 'Mixup'])
    assert h.expert_names == ['CE', 'Mixup']
    print("  ✅ expert names carried through")


# ---------------------------------------------------------------------------
# Run health
# ---------------------------------------------------------------------------

def _healthy_history(epochs=200):
    return [
        {'epoch': e, 'lr': (0.1 if e <= 160 else (0.001 if e <= 180 else 1e-5)),
         'train_loss': 4.5 / e ** 0.5, 'train_acc': None, 'grad_norm': 1.0,
         'time_s': 1.0}
        for e in range(1, epochs + 1)
    ]


def test_health_checker_accepts_a_healthy_run():
    rep = ev.RunHealthChecker('CE', seed=78).check(_healthy_history())
    assert rep.ok, rep.problems
    assert rep.epochs == 200
    print(f"  ✅ healthy run accepted ({rep.epochs} epochs)")


def test_health_checker_flags_wrong_lr_schedule():
    h = _healthy_history()
    for row in h:
        if row['epoch'] == 161:
            row['lr'] = 0.1                       # decay never applied
    rep = ev.RunHealthChecker('CE', seed=78).check(h)
    assert not rep.ok, "a broken LR schedule was accepted"
    assert any('lr' in p.lower() for p in rep.problems), rep.problems
    print(f"  ✅ LR schedule violation caught: {rep.problems[0][:60]}")


def test_health_checker_flags_nonfinite_loss():
    h = _healthy_history()
    h[50]['train_loss'] = float('nan')
    rep = ev.RunHealthChecker('CE', seed=78).check(h)
    assert not rep.ok and any('loss' in p.lower() for p in rep.problems), rep.problems
    print("  ✅ non-finite loss caught")


def test_health_checker_flags_rising_loss():
    h = _healthy_history()
    for row in h:
        row['train_loss'] = float(row['epoch'])   # loss increases
    rep = ev.RunHealthChecker('CE', seed=78).check(h)
    assert not rep.ok and any('descend' in p.lower() or 'increase' in p.lower()
                              for p in rep.problems), rep.problems
    print("  ✅ non-descending loss caught")


def test_health_checker_flags_short_run():
    rep = ev.RunHealthChecker('CE', seed=78).check(_healthy_history(epochs=150))
    assert not rep.ok and any('epoch' in p.lower() for p in rep.problems), rep.problems
    print("  ✅ truncated run caught")


def test_health_checker_flags_missing_final_checkpoint():
    d = tempfile.mkdtemp(prefix='dsh_health_')
    rep = ev.RunHealthChecker('CE', seed=78, checkpoint_dir=d).check(_healthy_history())
    assert not rep.ok and any('checkpoint' in p.lower() for p in rep.problems), rep.problems
    print("  ✅ missing final checkpoint caught")


def test_health_checker_accepts_present_final_checkpoint():
    d = tempfile.mkdtemp(prefix='dsh_health_')
    _fake_checkpoint(d, 'CE', 78)
    rep = ev.RunHealthChecker('CE', seed=78, checkpoint_dir=d).check(_healthy_history())
    assert rep.ok, rep.problems
    print("  ✅ present final checkpoint accepted")


# ---------------------------------------------------------------------------
# Test-set access log
# ---------------------------------------------------------------------------

def test_access_log_appends_and_never_truncates():
    path = os.path.join(tempfile.mkdtemp(prefix='dsh_log_'), 'access.md')
    log = TestAccessLog(path)
    log.record('python scripts/evaluate_experts.py', note='first')
    log.record('python scripts/evaluate_experts.py', note='second')
    entries = log.entries()
    assert len(entries) == 2, f"expected 2 entries, got {len(entries)}"
    print(f"  ✅ access log appended {len(entries)} entries")


def test_access_log_records_command_and_timestamp():
    path = os.path.join(tempfile.mkdtemp(prefix='dsh_log_'), 'access.md')
    log = TestAccessLog(path)
    log.record('python scripts/evaluate_experts.py --seeds 78', note='dry check')
    text = open(path).read()
    for token in ('scripts/evaluate_experts.py', 'dry check'):
        assert token in text, f"log missing '{token}'"
    print("  ✅ access log records the command and note")


def test_access_log_does_not_fail_the_run_when_unwritable():
    log = TestAccessLog('/proc/nonexistent_dir/access.md')
    log.record('anything')          # must not raise
    print("  ✅ unwritable log path does not break the run")


def test_access_log_detects_no_prior_access():
    path = os.path.join(tempfile.mkdtemp(prefix='dsh_log_'), 'access.md')
    log = TestAccessLog(path)
    assert log.entries() == [], "a fresh log should have no entries"
    print("  ✅ fresh log has no entries")


TESTS = [
    ("Balanced accuracy", test_balanced_accuracy_is_mean_per_class_recall),
    ("Group accuracies", test_group_accuracies_splits_head_med_tail),
    ("ECE calibrated", test_ece_is_zero_for_perfectly_calibrated_binary_confidences),
    ("ECE overconfident", test_ece_is_high_for_overconfident_predictor),
    ("evaluate_predictions keys", test_evaluate_predictions_reports_all_groups),
    ("Pool finds checkpoints", test_expert_pool_finds_checkpoints_by_naming_convention),
    ("Pool reports missing", test_expert_pool_reports_missing_checkpoints),
    ("Pool rejects single expert", test_expert_pool_rejects_duplicate_seed_only_pool),
    ("Pool stacks logits", test_expert_pool_logits_shape),
    ("Headroom all-wrong/oracle", test_headroom_all_wrong_and_oracle),
    ("Headroom kappa", test_headroom_kappa_is_one_for_identical_experts),
    ("Headroom carries names", test_headroom_names_experts),
    ("Health accepts healthy", test_health_checker_accepts_a_healthy_run),
    ("Health flags LR schedule", test_health_checker_flags_wrong_lr_schedule),
    ("Health flags NaN loss", test_health_checker_flags_nonfinite_loss),
    ("Health flags rising loss", test_health_checker_flags_rising_loss),
    ("Health flags short run", test_health_checker_flags_short_run),
    ("Health flags missing ckpt", test_health_checker_flags_missing_final_checkpoint),
    ("Health accepts present ckpt", test_health_checker_accepts_present_final_checkpoint),
    ("Access log appends", test_access_log_appends_and_never_truncates),
    ("Access log content", test_access_log_records_command_and_timestamp),
    ("Access log unwritable safe", test_access_log_does_not_fail_the_run_when_unwritable),
    ("Access log starts empty", test_access_log_detects_no_prior_access),
]


def main() -> int:
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001 - surface the real cause
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 62}")
    print(f"  {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
