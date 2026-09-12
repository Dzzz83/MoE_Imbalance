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


def test_group_definitions_agree_across_modules():
    """Every Head/Med/Tail splitter must implement the one frozen definition.

    AGENTs.md section 6 fixes the split immutably: Head >= 100 training samples,
    Medium 20-100, Tail < 20. A class with exactly 20 samples is therefore
    Medium. Three modules historically split classes, and one of them used
    ``> 20`` for Medium, which moved that boundary class into Tail.
    """
    from scripts.base_trainer import compute_class_groups

    utils_data = _safe('scripts.utils.data')
    dace = _safe('scripts.evaluate_dace')
    if utils_data is None or dace is None:
        raise AssertionError("could not import the class-group helpers")

    # class 99 holds exactly 20 samples: the boundary case, by construction
    counts = np.array([500] * 30 + [50] * 36 + [19] * 33 + [20], dtype=np.int64)
    canonical = compute_class_groups(counts)
    variants = {
        'scripts.utils.data': {
            'head': utils_data.get_class_groups(counts)['Head'],
            'medium': utils_data.get_class_groups(counts)['Med'],
            'tail': utils_data.get_class_groups(counts)['Tail'],
        },
        'scripts.evaluate_dace': {
            'head': dace.get_class_groups(counts)['Head'],
            'medium': dace.get_class_groups(counts)['Med'],
            'tail': dace.get_class_groups(counts)['Tail'],
        },
    }
    for name, groups in variants.items():
        for key in ('head', 'medium', 'tail'):
            assert set(groups[key].tolist()) == set(canonical[key].tolist()), (
                f"{name}.{key} disagrees with compute_class_groups: "
                f"{sorted(groups[key].tolist())} != {sorted(canonical[key].tolist())}"
            )
    assert 99 in canonical['medium'], "a class with exactly 20 samples must be Medium"
    print(f"  ✅ one definition: head={len(canonical['head'])} "
          f"medium={len(canonical['medium'])} tail={len(canonical['tail'])}")


def test_ece_includes_samples_at_full_confidence():
    """A prediction with confidence exactly 1.0 must land in the last bin.

    The live evaluator bins with ``(lo, hi]``; ``scripts/utils/metrics.ece``
    binned with ``[lo, hi)``, which dropped every confidence equal to 1.0 from
    all bins while still dividing by the full sample count — an understated ECE.
    """
    from scripts.utils.metrics import ece as routing_ece

    confidences = np.concatenate([np.ones(50), np.full(50, 0.5)])
    correct = np.zeros(100, dtype=bool)                 # every prediction wrong
    probs = np.stack([confidences, 1.0 - confidences], axis=1)
    targets = np.ones(100, dtype=np.int64)              # argmax is class 0 -> wrong

    expected = 0.5 * abs(0.0 - 0.5) + 0.5 * abs(0.0 - 1.0)
    live = ev.expected_calibration_error(probs, targets)
    legacy = routing_ece(confidences, correct)
    assert abs(live - expected) < 1e-9, f"live ECE {live} != {expected}"
    assert abs(legacy - expected) < 1e-9, (
        f"scripts.utils.metrics.ece = {legacy}, expected {expected}: the "
        f"confidence==1.0 samples were dropped from every bin"
    )
    print(f"  ✅ ECE counts full-confidence samples ({live:.4f} == {legacy:.4f})")


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


def test_legacy_checkpoint_loader_resolves_the_run_naming_convention():
    """Checkpoints are ``{expert}_seed{N}_final.pt``; the loader must resolve them.

    ``scripts/utils/data.load_expert_checkpoint`` looked for ``{expert}_best.pt``,
    a filename the trainer has never written, so every legacy caller died on a
    path that could not exist. A multi-seed directory must be refused rather than
    guessed, matching ExpertPool's rule.
    """
    utils_data = _safe('scripts.utils.data')
    if utils_data is None:
        raise AssertionError("scripts.utils.data could not be imported")

    d = tempfile.mkdtemp(prefix='dsh_legacy_ckpt_')
    _fake_checkpoint(d, 'CE', 78)
    try:
        model = utils_data.load_expert_checkpoint(
            'CE', checkpoint_dir=d, seed=78, device='cpu')
    except TypeError as exc:
        raise AssertionError(
            f"load_expert_checkpoint does not accept checkpoint_dir/seed: {exc}"
        ) from exc
    assert model is not None and not model.training

    _fake_checkpoint(d, 'CE', 88)
    try:
        utils_data.load_expert_checkpoint('CE', checkpoint_dir=d, device='cpu')
    except FileNotFoundError as exc:
        assert 'seed' in str(exc), f"ambiguous-seed error must name the fix: {exc}"
        print("  ✅ legacy checkpoint loader resolves _final.pt and refuses ambiguity")
        return
    raise AssertionError("an ambiguous multi-seed checkpoint dir must be refused")


def test_expert_pool_finds_checkpoints_by_naming_convention():
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    for label in EXPERT_LABELS:
        _fake_checkpoint(d, label, 78)
    pool = ev.ExpertPool(EXPERT_LABELS, seeds=[78], checkpoint_dir=d)
    found = pool.available()
    labels = {name for (name, _seed) in found}
    assert labels == set(EXPERT_LABELS), f"found {sorted(found)}"
    print(f"  ✅ ExpertPool located {len(found)} checkpoints")


def test_expert_pool_reports_missing_checkpoints():
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    _fake_checkpoint(d, 'CE', 78)
    pool = ev.ExpertPool(['CE', 'LAL'], seeds=[78], checkpoint_dir=d)
    assert pool.seeds_for('CE') == [78], pool.seeds_for('CE')
    assert pool.seeds_for('LAL') == [], pool.seeds_for('LAL')
    assert pool.missing() == ['LAL'], pool.missing()
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


def test_pool_rejects_cuda_when_the_gpu_is_unavailable():
    """A cuda request with no working GPU must fail clearly, not deep in torch.

    This happened for real: the NVIDIA driver wedged around a reboot,
    ``torch.cuda.is_available()`` went False, and ``ExpertPool.load()`` died
    inside ``torch.load`` with a 30-line traceback naming neither the cause nor
    the remedy.
    """
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    for label in ['CE', 'LAL']:
        _fake_checkpoint(d, label, 78)

    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        pool = ev.ExpertPool(['CE', 'LAL'], seeds=[78], checkpoint_dir=d, device='cuda')
        try:
            pool.load()
        except ev.EvaluationError as e:
            msg = str(e).lower()
            assert 'cuda' in msg, f"message does not mention cuda: {e}"
            assert 'cpu' in msg or 'nvidia-smi' in msg, f"message offers no remedy: {e}"
            print(f"  ✅ clear error instead of a torch traceback: {str(e)[:65]}")
            return
        raise AssertionError("a cuda request without a working GPU was accepted")
    finally:
        torch.cuda.is_available = real


def test_pool_does_not_silently_drop_seeds():
    """With three seeds on disk, available() must expose all six runs.

    Regression test: `found[name] = path` was assigned inside the seed loop, so
    every seed overwrote the previous one and a 3-seed evaluation silently
    reported a single seed's numbers.
    """
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    for label in ['CE', 'LAL']:
        for seed in (78, 88, 1034):
            _fake_checkpoint(d, label, seed)

    pool = ev.ExpertPool(['CE', 'LAL'], seeds=[78, 88, 1034], checkpoint_dir=d)
    found = pool.available()
    assert len(found) == 6, (
        f"expected 6 (expert, seed) entries, got {len(found)}: {found} - "
        f"seeds are being silently dropped"
    )
    assert ('CE', 1034) in found and ('CE', 78) in found, sorted(found)
    assert ('LAL', 88) in found, sorted(found)
    print(f"  \u2705 all {len(found)} expert/seed checkpoints visible")


def test_pool_load_refuses_an_ambiguous_multi_seed_pool():
    """Loading must not guess which seed to use."""
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    for label in ['CE', 'LAL']:
        for seed in (78, 88):
            _fake_checkpoint(d, label, seed)

    pool = ev.ExpertPool(['CE', 'LAL'], seeds=[78, 88], checkpoint_dir=d)
    try:
        pool.load()
    except ev.EvaluationError as e:
        assert 'seed' in str(e).lower(), f"message does not mention seed: {e}"
        print(f"  \u2705 ambiguous multi-seed load refused: {str(e)[:60]}")
        return
    raise AssertionError("an ambiguous multi-seed pool was loaded without a seed")


def test_pool_load_with_explicit_seed_works():
    """Passing a seed loads exactly that seed's experts."""
    d = tempfile.mkdtemp(prefix='dsh_pool_')
    for label in ['CE', 'LAL']:
        for seed in (78, 88):
            _fake_checkpoint(d, label, seed)

    pool = ev.ExpertPool(['CE', 'LAL'], seeds=[78, 88], checkpoint_dir=d)
    pool.load(seed=88)
    assert len(pool.loaded) == 2, pool.loaded
    print(f"  \u2705 explicit seed loads {pool.loaded}")


def test_aggregate_across_seeds_reports_mean_and_std():
    """3-seed reporting needs mean and spread, per AGENTs.md section 6."""
    per_seed = [
        {'ba': 0.40, 'tail': 0.10},
        {'ba': 0.42, 'tail': 0.14},
        {'ba': 0.44, 'tail': 0.12},
    ]
    agg = ev.aggregate_across_seeds(per_seed)
    assert abs(agg['ba']['mean'] - 0.42) < 1e-9, agg
    assert abs(agg['tail']['mean'] - 0.12) < 1e-9, agg
    assert abs(agg['ba']['std'] - 0.02) < 1e-9, agg
    assert agg['ba']['n'] == 3, agg
    print(f"  \u2705 aggregation: ba={agg['ba']['mean']:.4f}+-{agg['ba']['std']:.4f}")


def test_aggregate_across_seeds_handles_a_single_seed():
    agg = ev.aggregate_across_seeds([{'ba': 0.4}])
    assert abs(agg['ba']['mean'] - 0.4) < 1e-9
    assert agg['ba']['std'] == 0.0, agg
    print("  \u2705 single-seed aggregation gives std 0")


def test_cli_evaluates_every_seed_and_aggregates():
    """The CLI must evaluate each seed and report mean +- std, not one seed.

    Uses a faked test loader and a temporary access log, so this exercises the
    multi-seed path without reading the real test set.
    """
    import importlib
    from torch.utils.data import DataLoader, TensorDataset
    evr = importlib.import_module('scripts.evaluate_experts')

    d = tempfile.mkdtemp(prefix='dsh_cli_')
    for label in ['CE', 'LAL']:
        for seed in (78, 88):
            _fake_checkpoint(d, label, seed)

    g = torch.Generator().manual_seed(0)
    n = 64
    fake_loader = DataLoader(
        TensorDataset(torch.randn(n, 3, 32, 32, generator=g),
                      torch.randint(0, 100, (n,), generator=g)),
        batch_size=32,
    )
    original = evr.build_test_loader
    evr.build_test_loader = lambda *a, **k: fake_loader
    log_path = os.path.join(tempfile.mkdtemp(prefix='dsh_log_'), 'access.md')
    out = os.path.join(d, 'results.json')
    try:
        rc = evr.main([
            '--checkpoint-dir', d, '--experts', 'CE', 'LAL',
            '--seeds', '78', '88', '--device', 'cpu',
            '--tta-augs', '2', '--access-log', log_path, '--output', out,
        ])
    finally:
        evr.build_test_loader = original

    assert rc == 0, f"CLI returned {rc}"
    payload = json.load(open(out))
    assert payload['seeds'] == [78, 88], payload['seeds']
    assert set(payload['per_seed']) == {'78', '88'}, sorted(payload['per_seed'])
    # aggregated metrics must carry a mean and an std over the two seeds
    ba = payload['experts_aggregate']['CE']['ba']
    assert ba['n'] == 2, ba
    assert 'mean' in ba and 'std' in ba, ba
    print(f"  \u2705 CLI evaluated {payload['seeds']} and aggregated "
          f"(CE BA {ba['mean']:.4f}+-{ba['std']:.4f})")


def test_cli_access_log_is_written_once_per_run():
    """One evaluation = one access-log entry, regardless of seed count."""
    import importlib
    from torch.utils.data import DataLoader, TensorDataset
    evr = importlib.import_module('scripts.evaluate_experts')

    d = tempfile.mkdtemp(prefix='dsh_cli_')
    for label in ['CE', 'LAL']:
        for seed in (78, 88):
            _fake_checkpoint(d, label, seed)

    g = torch.Generator().manual_seed(0)
    fake_loader = DataLoader(
        TensorDataset(torch.randn(32, 3, 32, 32, generator=g),
                      torch.randint(0, 100, (32,), generator=g)),
        batch_size=32,
    )
    original = evr.build_test_loader
    evr.build_test_loader = lambda *a, **k: fake_loader
    log_path = os.path.join(tempfile.mkdtemp(prefix='dsh_log_'), 'access.md')
    try:
        evr.main(['--checkpoint-dir', d, '--experts', 'CE', 'LAL',
                  '--seeds', '78', '88', '--device', 'cpu', '--tta-augs', '1',
                  '--access-log', log_path,
                  '--output', os.path.join(d, 'r.json')])
    finally:
        evr.build_test_loader = original

    entries = TestAccessLog(log_path).entries()
    assert len(entries) == 1, f"expected 1 access entry, got {len(entries)}"
    print("  \u2705 one access-log entry for the whole multi-seed run")


TESTS = [
    ("Balanced accuracy", test_balanced_accuracy_is_mean_per_class_recall),
    ("Group accuracies", test_group_accuracies_splits_head_med_tail),
    ("ECE calibrated", test_ece_is_zero_for_perfectly_calibrated_binary_confidences),
    ("ECE overconfident", test_ece_is_high_for_overconfident_predictor),
    ("Group definitions agree", test_group_definitions_agree_across_modules),
    ("ECE counts confidence 1.0", test_ece_includes_samples_at_full_confidence),
    ("evaluate_predictions keys", test_evaluate_predictions_reports_all_groups),
    ("Pool finds checkpoints", test_expert_pool_finds_checkpoints_by_naming_convention),
    ("Legacy checkpoint loader naming", test_legacy_checkpoint_loader_resolves_the_run_naming_convention),
    ("Pool keeps all seeds", test_pool_does_not_silently_drop_seeds),
    ("Pool refuses ambiguous seed", test_pool_load_refuses_an_ambiguous_multi_seed_pool),
    ("Pool loads explicit seed", test_pool_load_with_explicit_seed_works),
    ("Aggregate across seeds", test_aggregate_across_seeds_reports_mean_and_std),
    ("Aggregate single seed", test_aggregate_across_seeds_handles_a_single_seed),
    ("CLI evaluates every seed", test_cli_evaluates_every_seed_and_aggregates),
    ("CLI logs access once", test_cli_access_log_is_written_once_per_run),
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
    ("Pool rejects cuda without GPU", test_pool_rejects_cuda_when_the_gpu_is_unavailable),
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
