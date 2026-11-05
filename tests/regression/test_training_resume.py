"""Checkpoint continuation must preserve the epoch budget and training state."""

import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.base_trainer import BaseTrainer, CheckpointValidationError, set_seed


class TinyTrainer(BaseTrainer):
    def __init__(self, checkpoint_dir, **kwargs):
        set_seed(19)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout(0.25))
        super().__init__(
            model, 'Tiny', device='cpu', seed=19,
            checkpoint_dir=str(checkpoint_dir), **kwargs,
        )

    def _compute_loss(self, images, targets, weights=None):
        # Exercise each RNG used by augmentation, mixup, shuffling and dropout.
        images = images * (0.8 + 0.1 * random.random() + 0.1 * np.random.random())
        logits = self.model(images)
        return torch.nn.functional.cross_entropy(logits, targets), logits, {}


def loader(explicit_generator=False):
    dataset = TensorDataset(
        torch.arange(48, dtype=torch.float32).reshape(12, 4) / 48,
        torch.arange(12) % 3,
    )
    generator = torch.Generator().manual_seed(41) if explicit_generator else None
    return DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)


def test_resume_continues_after_saved_epoch_at_decay_boundary(tmp_path):
    original = TinyTrainer(tmp_path, epochs=162)
    original.epoch = 160
    checkpoint = original._save_checkpoint({}, is_final=False)
    resumed = TinyTrainer(tmp_path, epochs=162)
    resumed.load_checkpoint(str(checkpoint))
    observed = []

    def step(_loader):
        observed.append((resumed.epoch, resumed.optimiser.param_groups[0]['lr']))
        return {'loss': 1.0, 'grad_norm': 0.0}

    resumed._train_one_epoch = step
    resumed.train(loader())
    assert observed == [(161, 0.001), (162, 0.001)]


@pytest.mark.parametrize('explicit_generator', [False, True])
def test_resume_matches_uninterrupted_training(tmp_path, explicit_generator):
    uninterrupted = TinyTrainer(tmp_path / 'full', epochs=3)
    uninterrupted.train(loader(explicit_generator))
    expected_draws = (random.random(), float(np.random.random()), torch.rand(3))

    interrupted = TinyTrainer(tmp_path / 'interrupted', epochs=3)
    checkpoint = None

    class Interrupted(Exception):
        pass

    def stop_after_first_epoch(log):
        nonlocal checkpoint
        checkpoint = interrupted._save_checkpoint(log, is_final=False)
        raise Interrupted

    interrupted.metrics_sink = stop_after_first_epoch
    with pytest.raises(Interrupted):
        interrupted.train(loader(explicit_generator))

    # Checkpoints remain compatible with the restricted torch.load default.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert [row['epoch'] for row in saved['history']] == [1]
    resumed = TinyTrainer(tmp_path / 'resumed', epochs=3)
    set_seed(999)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    resumed.load_checkpoint(str(checkpoint))
    # Inference loads weights without changing process randomness.
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    resumed.train(loader(explicit_generator))

    assert [row['epoch'] for row in resumed.history] == [1, 2, 3]
    for expected, actual in zip(uninterrupted.history, resumed.history):
        assert {k: v for k, v in actual.items() if k != 'time_s'} == {
            k: v for k, v in expected.items() if k != 'time_s'
        }
    for key, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[key], expected)
    expected_state = uninterrupted.optimiser.state_dict()
    actual_state = resumed.optimiser.state_dict()
    assert actual_state['param_groups'] == expected_state['param_groups']
    for key, state in expected_state['state'].items():
        assert torch.equal(actual_state['state'][key]['momentum_buffer'], state['momentum_buffer'])
    assert random.random() == expected_draws[0]
    assert float(np.random.random()) == expected_draws[1]
    assert torch.equal(torch.rand(3), expected_draws[2])


def test_completed_checkpoint_is_a_noop_without_reseeding_or_writing(tmp_path):
    original = TinyTrainer(tmp_path, epochs=1)
    original.train(loader())
    checkpoint = tmp_path / 'Tiny_seed19_final.pt'
    before = checkpoint.read_bytes()
    resumed = TinyTrainer(tmp_path, epochs=1)
    resumed.load_checkpoint(str(checkpoint))
    resumed._train_one_epoch = Mock(side_effect=AssertionError('already completed'))
    set_seed(991)
    rng_before = torch.get_rng_state().clone()
    history = resumed.train(loader())
    assert [row['epoch'] for row in history] == [1]
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert checkpoint.read_bytes() == before
    resumed._train_one_epoch.assert_not_called()


def test_checkpoint_beyond_budget_can_load_for_inference_but_cannot_train(tmp_path):
    original = TinyTrainer(tmp_path, epochs=3)
    original.epoch = 3
    checkpoint = original._save_checkpoint({})
    resumed = TinyTrainer(tmp_path, epochs=2)
    resumed.load_checkpoint(str(checkpoint))
    resumed._train_one_epoch = Mock(side_effect=AssertionError('past budget'))
    with pytest.raises(ValueError, match='epoch.*budget'):
        resumed.train(loader())
    resumed._train_one_epoch.assert_not_called()


def test_legacy_checkpoint_continues_with_explicit_reproducibility_warning(tmp_path):
    original = TinyTrainer(tmp_path, epochs=2)
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({
        'epoch': 1, 'seed': 19, 'expert_name': 'Tiny', 'is_final': False,
        'model_state_dict': original.model.state_dict(),
        'optimiser_state_dict': original.optimiser.state_dict(),
    }, checkpoint)
    resumed = TinyTrainer(tmp_path, epochs=2)
    resumed.load_checkpoint(str(checkpoint))
    with pytest.warns(RuntimeWarning, match='RNG'):
        resumed.train(loader())
    assert [row['epoch'] for row in resumed.history] == [2]


def test_negative_checkpoint_epoch_is_rejected(tmp_path):
    original = TinyTrainer(tmp_path, epochs=2)
    original.epoch = -1
    checkpoint = original._save_checkpoint({}, is_final=False)
    with pytest.raises(CheckpointValidationError, match='epoch'):
        TinyTrainer(tmp_path, epochs=2).load_checkpoint(str(checkpoint))


@pytest.mark.parametrize('explicit_generator', [False, True])
def test_resume_rejects_switching_loader_generator_mode(tmp_path, explicit_generator):
    original = TinyTrainer(tmp_path, epochs=1)
    original.train(loader(explicit_generator))
    resumed = TinyTrainer(tmp_path, epochs=2)
    resumed.load_checkpoint(str(tmp_path / 'Tiny_seed19_final.pt'))
    resumed._train_one_epoch = Mock(side_effect=AssertionError('incompatible loader'))
    with pytest.raises(ValueError, match='loader generator differs'):
        resumed.train(loader(not explicit_generator))
    resumed._train_one_epoch.assert_not_called()


def test_resuming_smoke_checkpoint_cannot_create_reportable_model(tmp_path):
    original = TinyTrainer(tmp_path, epochs=1)
    original.max_batches = 1
    original.train(loader())
    checkpoint = tmp_path / 'Tiny_seed19_final.pt'
    resumed = TinyTrainer(tmp_path, epochs=2)
    resumed.load_checkpoint(str(checkpoint))
    resumed.train(loader())
    state = torch.load(checkpoint, weights_only=True)
    assert state['epoch'] == 2
    assert state['max_batches'] == 1
    assert state['is_dry_run'] is True
    assert state['is_final'] is False
