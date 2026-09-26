# Tests

Tests are grouped by the scope they exercise:

- `unit/` covers isolated model, data, routing, metric, and training utilities.
- `integration/` covers multi-component workflows, configuration/CLI behavior,
  OOF execution boundaries, and the Ridge/Sinkhorn study pipeline.
- `regression/` protects corrected protocol, artifact, and evaluation behavior.

From the repository root, run the CPU-compatible suite with:

```bash
python -m pytest --ignore=tests/integration/test_gpu.py
```

The GPU checks use a custom harness (including explicit skip reporting), so run
them directly on a CUDA machine with:

```bash
python tests/integration/test_gpu.py
```

The shared `tests/repo_root.py` helper keeps repository lookup stable as tests
move between these directories. `integration/promotion_worker.py` is a
spawn-safe helper used by the atomic-promotion concurrency test.
