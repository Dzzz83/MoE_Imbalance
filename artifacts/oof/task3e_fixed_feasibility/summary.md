# Task 3E-A — Fixed-Weight Ensemble Feasibility Study

Exploratory analysis of the predefined 35 global fixed-weight logit ensembles.
Every result below uses only the outer-fold-0 router-fit partition (inner folds 1–3); it is not an independently validated generalization result.

## Provenance and protocol

- Source commit: `5b28fecc40d06211c0a0fd90da278dfd6a800c06`
- Dataset: CIFAR-100-LT, IR=100; canonical population=10847
- Analyzed population: 6507 samples; inner folds [1, 2, 3]; outer fold 0
- Expert order: `CE, LAL, BalancedSoftmax, Mixup`
- Logits: original aligned OOF logits, without normalization or calibration

## Uniform baseline reproduction

Stored Task 3C primary-partition values were compared with tolerance `1e-12`. Match: **True**.

| Metric | Stored Task 3C | Task 3E-A | Absolute difference |
|---|---:|---:|---:|
| Ordinary accuracy | 59.4283% | 59.4283% | 0.000e+00 |
| Balanced Accuracy | 35.9328% | 35.9328% | 0.000e+00 |
| Head accuracy | 61.9661% | 61.9661% | 0.000e+00 |
| Medium accuracy | 34.6909% | 34.6909% | 0.000e+00 |
| Tail accuracy | 7.0094% | 7.0094% | 0.000e+00 |

## All 35 fixed-weight candidates

Deltas are percentage-point changes from the uniform row. Candidate IDs follow lexicographic integer-unit ordering and are stable across runs.

| ID | Weights (CE, LAL, BS, Mixup) | Accuracy | BA | Head | Medium | Tail | ΔBA | ΔHead | ΔMedium | ΔTail | Both BA+Tail? | Pareto? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| fixed_000 | (0.00, 0.00, 0.00, 1.00) | 58.6292% | 31.0707% | 61.7710% | 25.8687% | 1.3228% | -4.8620 pp | -0.1951 pp | -8.8222 pp | -5.6866 pp | no | no |
| fixed_001 | (0.00, 0.00, 0.25, 0.75) | 60.9497% | 35.2999% | 63.5909% | 31.9713% | 6.1772% | -0.6328 pp | +1.6248 pp | -2.7196 pp | -0.8321 pp | no | no |
| fixed_002 | (0.00, 0.00, 0.50, 0.50) | 57.1077% | 34.8858% | 59.2129% | 32.9878% | 8.7185% | -1.0470 pp | -2.7532 pp | -1.7031 pp | +1.7091 pp | no | no |
| fixed_003 | (0.00, 0.00, 0.75, 0.25) | 52.0977% | 33.2701% | 53.9261% | 32.1980% | 10.4222% | -2.6626 pp | -8.0400 pp | -2.4928 pp | +3.4128 pp | no | no |
| fixed_004 | (0.00, 0.00, 1.00, 0.00) | 47.7332% | 31.1445% | 49.5497% | 29.7757% | 11.2688% | -4.7882 pp | -12.4164 pp | -4.9152 pp | +4.2594 pp | no | no |
| fixed_005 | (0.00, 0.25, 0.00, 0.75) | 60.7192% | 35.1311% | 63.6629% | 32.0812% | 5.4022% | -0.8017 pp | +1.6968 pp | -2.6097 pp | -1.6071 pp | no | no |
| fixed_006 | (0.00, 0.25, 0.25, 0.50) | 60.1660% | 37.2134% | 62.8900% | 34.9384% | 9.9115% | +1.2807 pp | +0.9239 pp | +0.2475 pp | +2.9021 pp | yes | yes |
| fixed_007 | (0.00, 0.25, 0.50, 0.25) | 55.8322% | 36.4403% | 58.1737% | 33.3167% | 14.7290% | +0.5076 pp | -3.7924 pp | -1.3742 pp | +7.7196 pp | yes | yes |
| fixed_008 | (0.00, 0.25, 0.75, 0.00) | 51.0220% | 34.1670% | 52.7380% | 32.3149% | 14.6616% | -1.7658 pp | -9.2280 pp | -2.3760 pp | +7.6522 pp | no | no |
| fixed_009 | (0.00, 0.50, 0.00, 0.50) | 57.0155% | 35.4587% | 59.4333% | 34.1516% | 9.0133% | -0.4740 pp | -2.5328 pp | -0.5393 pp | +2.0040 pp | no | no |
| fixed_010 | (0.00, 0.50, 0.25, 0.25) | 55.9398% | 36.4236% | 58.4063% | 33.2279% | 14.5055% | +0.4908 pp | -3.5598 pp | -1.4630 pp | +7.4962 pp | yes | no |
| fixed_011 | (0.00, 0.50, 0.50, 0.00) | 52.3436% | 35.4372% | 54.4225% | 32.4557% | 16.7660% | -0.4956 pp | -7.5435 pp | -2.2352 pp | +9.7566 pp | no | yes |
| fixed_012 | (0.00, 0.75, 0.00, 0.25) | 51.9133% | 33.5177% | 54.2487% | 31.1714% | 12.0689% | -2.4150 pp | -7.7174 pp | -3.5195 pp | +5.0595 pp | no | no |
| fixed_013 | (0.00, 0.75, 0.25, 0.00) | 50.9759% | 34.4013% | 52.8850% | 31.6105% | 16.0928% | -1.5315 pp | -9.0810 pp | -3.0804 pp | +9.0835 pp | no | no |
| fixed_014 | (0.00, 1.00, 0.00, 0.00) | 47.2261% | 30.6923% | 49.2259% | 27.4539% | 12.8481% | -5.2404 pp | -12.7402 pp | -7.2370 pp | +5.8387 pp | no | no |
| fixed_015 | (0.25, 0.00, 0.00, 0.75) | 61.5491% | 33.9972% | 64.5479% | 30.2286% | 2.7513% | -1.9356 pp | +2.5818 pp | -4.4623 pp | -4.2581 pp | no | no |
| fixed_016 | (0.25, 0.00, 0.25, 0.50) | 60.6424% | 34.0139% | 63.6509% | 31.9896% | 1.7989% | -1.9189 pp | +1.6848 pp | -2.7013 pp | -5.2104 pp | no | no |
| fixed_017 | (0.25, 0.00, 0.50, 0.25) | 57.3690% | 34.3381% | 59.7854% | 32.9629% | 6.2541% | -1.5946 pp | -2.1807 pp | -1.7280 pp | -0.7553 pp | no | no |
| fixed_018 | (0.25, 0.00, 0.75, 0.00) | 53.1120% | 33.3413% | 55.2879% | 32.3685% | 8.8717% | -2.5915 pp | -6.6781 pp | -2.3224 pp | +1.8623 pp | no | no |
| fixed_019 | (0.25, 0.25, 0.00, 0.50) | 60.7807% | 35.2515% | 63.3235% | 34.0856% | 3.8612% | -0.6812 pp | +1.3574 pp | -0.6053 pp | -3.1481 pp | no | no |
| fixed_020 | (0.25, 0.25, 0.25, 0.25) | 59.4283% | 35.9328% | 61.9661% | 34.6909% | 7.0094% | +0.0000 pp | +0.0000 pp | +0.0000 pp | +0.0000 pp | no | no |
| fixed_021 | (0.25, 0.25, 0.50, 0.00) | 55.8014% | 35.1688% | 57.8657% | 33.2162% | 10.9671% | -0.7640 pp | -4.1003 pp | -1.4747 pp | +3.9577 pp | no | no |
| fixed_022 | (0.25, 0.50, 0.00, 0.25) | 57.2768% | 34.3564% | 59.7348% | 31.9816% | 7.5186% | -1.5764 pp | -2.2312 pp | -2.7093 pp | +0.5093 pp | no | no |
| fixed_023 | (0.25, 0.50, 0.25, 0.00) | 55.8783% | 34.8274% | 58.2668% | 33.6753% | 8.8256% | -1.1053 pp | -3.6993 pp | -1.0156 pp | +1.8163 pp | no | no |
| fixed_024 | (0.25, 0.75, 0.00, 0.00) | 52.7893% | 32.9187% | 55.0465% | 30.8958% | 9.4631% | -3.0140 pp | -6.9196 pp | -3.7951 pp | +2.4537 pp | no | no |
| fixed_025 | (0.50, 0.00, 0.00, 0.50) | 58.2911% | 31.8148% | 61.2812% | 27.9857% | 1.9048% | -4.1179 pp | -0.6849 pp | -6.7052 pp | -5.1046 pp | no | no |
| fixed_026 | (0.50, 0.00, 0.25, 0.25) | 57.9530% | 32.6739% | 60.8927% | 30.1539% | 2.6918% | -3.2589 pp | -1.0734 pp | -4.5370 pp | -4.3176 pp | no | no |
| fixed_027 | (0.50, 0.00, 0.50, 0.00) | 55.8629% | 33.0177% | 58.4031% | 31.5580% | 5.1045% | -2.9150 pp | -3.5630 pp | -3.1329 pp | -1.9049 pp | no | no |
| fixed_028 | (0.50, 0.25, 0.00, 0.25) | 57.6610% | 33.1635% | 60.4073% | 30.7185% | 4.2316% | -2.7692 pp | -1.5587 pp | -3.9724 pp | -2.7778 pp | no | no |
| fixed_029 | (0.50, 0.25, 0.25, 0.00) | 56.7850% | 33.9285% | 59.1866% | 31.9263% | 6.7964% | -2.0043 pp | -2.7795 pp | -2.7646 pp | -0.2130 pp | no | no |
| fixed_030 | (0.50, 0.50, 0.00, 0.00) | 55.4633% | 33.1541% | 57.8735% | 30.9307% | 6.9090% | -2.7786 pp | -4.0926 pp | -3.7602 pp | -0.1004 pp | no | no |
| fixed_031 | (0.75, 0.00, 0.00, 0.25) | 54.8179% | 30.0007% | 57.8769% | 25.0522% | 3.2516% | -5.9321 pp | -4.0892 pp | -9.6387 pp | -3.7578 pp | no | no |
| fixed_032 | (0.75, 0.00, 0.25, 0.00) | 55.0330% | 31.3397% | 57.6580% | 28.3646% | 4.1059% | -4.5931 pp | -4.3081 pp | -6.3263 pp | -2.9034 pp | no | no |
| fixed_033 | (0.75, 0.25, 0.00, 0.00) | 55.0791% | 31.6240% | 57.6874% | 28.6316% | 4.7078% | -4.3088 pp | -4.2787 pp | -6.0593 pp | -2.3016 pp | no | no |
| fixed_034 | (1.00, 0.00, 0.00, 0.00) | 52.0055% | 28.5016% | 54.8905% | 23.6979% | 3.3189% | -7.4311 pp | -7.0755 pp | -10.9929 pp | -3.6905 pp | no | no |

## Joint improvement and BA–Tail trade-offs

- Candidates improving both BA and Tail: **3** (fixed_006, fixed_007, fixed_010).
- Candidates that improve BA while reducing Tail: none.
- Candidates that improve Tail while reducing BA: fixed_002, fixed_003, fixed_004, fixed_008, fixed_009, fixed_011, fixed_012, fixed_013, fixed_014, fixed_018, fixed_021, fixed_022, fixed_023, fixed_024.
- Candidates that reduce both BA and Tail: fixed_000, fixed_001, fixed_005, fixed_015, fixed_016, fixed_017, fixed_019, fixed_025, fixed_026, fixed_027, fixed_028, fixed_029, fixed_030, fixed_031, fixed_032, fixed_033, fixed_034.
- Candidates that have no strict change or a zero-delta boundary: fixed_020.
- Head/Medium deltas for the joint-improvement candidates: fixed_006 (+0.9239 pp Head, +0.2475 pp Medium); fixed_007 (-3.7924 pp Head, -1.3742 pp Medium); fixed_010 (-3.5598 pp Head, -1.4630 pp Medium).

## Observed Pareto frontier

This is the complete non-dominated frontier among the 35 predefined grid points only; it is not the frontier of all convex weight vectors. Exact BA–Tail ties are retained deterministically.

| ID | Weights | BA | Tail | Head | Medium | ΔBA | ΔTail |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_006 | (0.00, 0.25, 0.25, 0.50) | 37.2134% | 9.9115% | 62.8900% | 34.9384% | +1.2807 pp | +2.9021 pp |
| fixed_007 | (0.00, 0.25, 0.50, 0.25) | 36.4403% | 14.7290% | 58.1737% | 33.3167% | +0.5076 pp | +7.7196 pp |
| fixed_011 | (0.00, 0.50, 0.50, 0.00) | 35.4372% | 16.7660% | 54.4225% | 32.4557% | -0.4956 pp | +9.7566 pp |

## Limitations and next-step implication

- Inner fold 0 was excluded completely from new Task 3E-A calculations. Earlier Task 3C exploratory diagnostics did inspect it, so the router-selection partition cannot be described as untouched for all prior research decisions.
- The reserved outer-evaluation population and the original CIFAR-100 test set were not accessed. No checkpoint, method, or final weight choice should be selected from this report alone.
- The one-third three-expert vector `(0, 1/3, 1/3, 1/3)` is not an allowed grid point and was not silently added as a 36th candidate.
- The next soft-mixture feasibility experiment must remain a separate exploratory analysis; these fixed global weights do not establish the value of adaptive per-sample weighting.
