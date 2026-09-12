# Test-Set Access Log

> Append-only. Every entry is written automatically by the evaluation
> entry points the moment they load the test set. The candidate routing
> rules were frozen in `docs/routing-preregistration.md`; this log exists
> so that any access *after* that freeze is visible.

| timestamp (UTC) | git | command | note |
|:--|:--|:--|:--|
| 2026-09-12 01:15:31 | e2ed779 | `scripts/evaluate_experts.py --seeds 78 --device cuda` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78 |
| 2026-09-12 01:15:46 | e2ed779 | `scripts/evaluate_experts.py --seeds 78` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78 |
| 2026-09-12 03:02:29 | 1a2cc76 | `scripts/evaluate_experts.py --seeds 78 --device cuda --tta-augs 10` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78 |
