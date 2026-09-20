# Test-Set Access Log

> Append-only. Every entry is written automatically by the evaluation
> entry points the moment they load the test set. The candidate routing
> rules were frozen in `../records/routing-preregistration.md`; this log exists
> so that any access *after* that freeze is visible.

| timestamp (UTC) | git | command | note |
|:--|:--|:--|:--|
| 2026-09-12 01:15:31 | e2ed779 | `scripts/evaluate_experts.py --seeds 78 --device cuda` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78 |
| 2026-09-12 01:15:46 | e2ed779 | `scripts/evaluate_experts.py --seeds 78` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78 |
| 2026-09-12 03:02:29 | 1a2cc76 | `scripts/evaluate_experts.py --seeds 78 --device cuda --tta-augs 10` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78 |
| 2026-09-12 08:00:35 | 8cde452 | `scripts/evaluate_experts.py --seeds 78 88 1034 --device cuda --tta-augs 10` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78,88,1034 |
| 2026-09-12 09:26:23 | 8cde452 | `scripts/evaluate_experts.py --seeds 78 88 1034 --device cuda --tta-augs 10` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78,88,1034 |
| 2026-09-12 09:38:52 | 8cde452 | `scripts/analyze_subsets.py --seeds 78 88 1034 --device cuda` | subset analysis experts=CE,LAL,BalancedSoftmax,Mixup seeds=78,88,1034 |
| 2026-09-12 13:14:09 | 8eeef41 | `scripts/evaluate_experts.py --seeds 78 88 1034 --device cuda --tta-augs 10` | experts=CE,LAL,BalancedSoftmax,Mixup seeds=78,88,1034 |
| 2026-09-12 13:19:37 | 8eeef41 | `scripts/analyze_subsets.py --seeds 78 88 1034 --device cuda` | subset analysis experts=CE,LAL,BalancedSoftmax,Mixup seeds=78,88,1034 |
