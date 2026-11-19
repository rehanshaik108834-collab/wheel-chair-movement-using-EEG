# A1-A3: honest LEMC-style combination, fusion with Mam's classifier, paper protocol

## Chosen on session-T CV (5 models)

| Rule | T-CV acc |
|---|---|
| PLAIN | 78.82 |
| TEMP | 79.24 |
| LEMC | 79.48 |

Chosen rule: **LEMC** (keep needs +0.5 over PLAIN)

Fusion with Mam's classifier on T-CV: weight 0 -> 79.48, best weight 0.1 -> 79.98; kept weight **0.1**

## Session E (15 models), mean of 9 subjects

| Combination | EA ref: all E trials | EA ref: first 48 trials | vs old 76.16 (offline) |
|---|---|---|---|
| plain mean | 79.48 | 77.74 | 8/9, p = 0.012 |
| per-model temperature | 79.55 | 77.93 | 8/9, p = 0.012 |
| temperature + weighting MLP (honest LEMC) | 80.13 | 78.28 | 8/9, p = 0.012 |
| chosen rule + Mam's classifier (w=0.1, as decided on T) | 80.09 | 78.24 | 8/9, p = 0.012 |
| chosen rule + Mam's classifier (w=0.1, T-CV best, for reference) | 80.09 | 78.24 | 8/9, p = 0.012 |

## Side comparison: the paper's own protocol (temperatures + MLP fitted on the session-E labels; NOT honest)

- 15 nets, deterministic: 80.09
- 15 nets, MC dropout (as LEMC-FTDB): 80.21
- Paper LEMC-FTDB (EEGNet, this protocol): 80.40; paper honest EMCD (15 nets): 75.35

## Rejection on the final system (temperature + entropy threshold from T-CV of the final system)

| EA reference | Acc | ECE raw -> cal | AUROC | 80% cov: coverage / accepted acc / false per 100 | 85% target: coverage / accepted acc / false per 100 |
|---|---|---|---|---|---|
| within | 80.09 | 3.25 -> 1.68 | 0.822 | 82.48 / 86.72 / 11.0 | 86.81 / 85.07 / 13.0 |
| within_online48 | 77.22 | 1.44 -> 3.88 | 0.813 | 83.10 / 83.23 / 13.9 | 88.56 / 81.44 / 16.4 |

Per subject (all E trials): accuracy | 80% cov accepted acc | 85% target coverage

- A01: 89.93 | 89.90 | 100.00
- A02: 63.19 | 72.87 | 72.92
- A03: 92.71 | 93.36 | 100.00
- A04: 73.26 | 82.44 | 81.94
- A05: 70.49 | 75.40 | 90.97
- A06: 62.85 | 91.04 | 37.50
- A07: 90.97 | 91.61 | 99.31
- A08: 89.58 | 91.17 | 99.31
- A09: 87.85 | 88.73 | 99.31