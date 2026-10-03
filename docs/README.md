# Documentation index

Start with the [project overview](../README.md) for the current result,
metrics, quick start, and selected references. Use [protocol.md](protocol.md)
for the authoritative data roles, metric definitions, and safeguards.

## Current guidance

| Document | Owns |
|:--|:--|
| [Protocol](protocol.md) | Data roles, class groups, metrics, leakage safeguards, and artifact rules |
| [Frozen plan](PLAN.md) | Hash-bound v1 protocol, candidate grid, and freeze rules |
| [Experiment workflow](experiment-workflow.md) | New runs, freeze and lock stages, sessions, and recovery |
| [Reproduction](reproduction.md) | Saved-study commands, evaluator provenance contract, and evidence locations |
| [Full-matrix report](rs3-final-report.md) | Current three-seed OOF outcomes, uncertainty, and audit summary |
| [Diagnostics](diagnostics.md) | Read-only inner-study diagnostics and pinned inputs |
| [Research notes](research.md) | Measured constraints, remaining questions, and next-study safeguards |

## Historical evidence and records

- [Full-data results](results.md) is the original balanced-test benchmark. That
  test set has informed earlier research decisions.
- [OOF results](oof-results.md) summarizes Tasks 3C–3F and the consumed
  seed-78 / outer-fold-0 evaluation, with links to immutable evidence.
- [Archived Task 3C–3F history](archive/oof-task-history.md) preserves the
  detailed task chronology and diagnostic findings. The OOF page is the
  navigable result record; the archive retains historical detail.
- [Archive](archive/) contains superseded plans, audits, and reports.
- [Routing mechanism record](../records/routing_mechanism.md) surveys tested
  and proposed approaches. The
  [routing preregistration](../records/routing-preregistration.md) is frozen.
- [Test-access log](test-access-log.md) is append-only.

`results.md` and `oof-results.md` remain at their original paths because the
frozen [PLAN.md](PLAN.md) links to `oof-results.md`, names both pages in its
prose, and other records use the Task 3E heading anchor. Moving them would
require compatibility stubs without reducing the file count. The docs
directory therefore keeps 11 top-level Markdown pages; this index clarifies
their ownership.

## Workspace notes

| Path | Role |
|:--|:--|
| `artifacts/oof/` | Immutable OOF predictions, reports, and analysis evidence |
| `checkpoints/` | Full-data training checkpoints |
| `.rs3-server/` | Separate runtime, environment, logs, and completed matrix outputs |
| `.rs3-server/code/` | Frozen training-source worktree at commit `e7357a7` |
| `../MoE_Imbalance-evaluation-provenance/` | Evaluator worktree at commit `64401e8` |
| `/tmp/rs3-analysis-diagnostics-85fe4e2/` | Inner-diagnostics analysis worktree at commit `85fe4e2` |

These worktrees hold distinct source identities and roles. Preserve them with
the evidence they generated. See [reproduction](reproduction.md) for the
formal evaluator identity and [diagnostics](diagnostics.md) for the pinned
inner-analysis snapshot.

## Document ownership

Definitions and safeguards live in [protocol.md](protocol.md). Commands for
new work belong in [experiment-workflow.md](experiment-workflow.md); saved
study reproduction belongs in [reproduction.md](reproduction.md). The
[full-matrix report](rs3-final-report.md) owns the current outcome, while the
full-data and OOF result pages retain their separate historical populations.
