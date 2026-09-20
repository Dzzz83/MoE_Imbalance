# Historical Results — Superseded Protocols

> Archive only. These results are retained for traceability and are **not comparable** to the current canonical protocol. Current authoritative results are in [`../results.md`](../results.md).

Kept so earlier claims stay traceable. **Three** protocols exist in this
project's history and their numbers are not interchangeable:

| Protocol | Training data | Validation split | Notes |
|:--|:--|:--|:--|
| **Current** (§2–§5) | full LT set, 10,847 | none | 4 experts, 3 seeds, seed-controlled init |
| **B — proper 80/20** | 8,677 (80% of 10,847) | 2,170 LT samples | non-standard: split off 20% of an already small set; left 4 tail classes with zero val samples |
| **A — flawed** | 8,677 from a 45,000 pool | balanced 5,000 held out *before* LT subsampling | the original, non-standard pipeline |

**Protocol B — proper 80/20 split** (3 experts: LAL + PaCo + Mixup):

| Metric | Value |
|:--|:--:|
| 3-expert uniform averaging | 45.68 |
| Optimal fixed weights | 44.89 |
| Oracle | 55.79 |
| All-wrong floor | 44.2 |

Per-expert BA on protocol B: PaCo 41.17, BalancedSoftmax 39.99, LAL 39.96,
Mixup 37.52, CE 36.64.

**Protocol A — flawed split** (same 3 experts):

| Metric | Value |
|:--|:--:|
| Best single expert (PaCo) | 49.28 |
| 3-expert uniform averaging | **51.12** |
| Optimal fixed weights | 52.58 |
| Oracle | ≈62.9 |
| All-wrong floor | ≈37.1 |

*Protocol A looks far better on every metric, which is the point: holding out a
balanced validation set before subsampling made the task easier and the numbers
incomparable. Note that `records/routing_mechanism.md` §3 quotes **51.12** as its
baseline — that is protocol A, not protocol B or the current one.*

**Current vs protocol B.** Moving from an 8,677-sample training set to the full
10,847 improved every expert:

| Expert | B (8,677) | Current (10,847) |
|:--|:--:|:--:|
| LAL | 39.96 | 42.43 |
| BalancedSoftmax | 39.99 | 41.34 |
| Mixup | 37.52 | 38.69 |
| CE | 36.64 | 37.69 |
| PaCo | 41.17 | not trained |

Protocols A and B also predate the seeding fix: their experts were initialised
from an uncontrolled RNG state, so their run-to-run variance is not the seed
variance reported in §2–§5.
