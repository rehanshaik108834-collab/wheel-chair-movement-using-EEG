# Improved model (EA + S&R EEGNet): ensemble size, MC dropout, rejection

Rule chosen on T-CV (5-model ensemble): plain 78.82, MC dropout 78.70 (gain -0.12, keep needs +0.5) -> **AUGSR**

## Session E (train T -> test E), mean of 9 subjects

| Rule | Models | EA reference | 1 model | Ensemble | vs old 76.16: better for, p |
|---|---|---|---|---|---|
| plain (chosen) | 5 | offline | 75.60 | 79.13 | 7/9, p = 0.039 |
| plain (chosen) | 5 | online48 | 73.67 | 76.93 | 6/9, p = 0.375 |
| plain (chosen) | 15 | offline | 74.84 | 79.48 | 8/9, p = 0.012 |
| plain (chosen) | 15 | online48 | 72.89 | 77.74 | 6/9, p = 0.121 |
| MC dropout | 5 | offline | 75.53 | 79.13 | 7/9, p = 0.039 |
| MC dropout | 5 | online48 | 73.65 | 76.93 | 6/9, p = 0.344 |
| MC dropout | 15 | offline | 74.75 | 79.75 | 8/9, p = 0.012 |
| MC dropout | 15 | online48 | 72.83 | 77.43 | 6/9, p = 0.234 |

Paper (Nzakuna et al. 2025, EEGNet): LEMC-FTDB 80.40 (temperatures/weights fitted on test labels), EMCD 15 nets honest 75.35, MCD 71.14

## Rejection (temperature + entropy threshold from T-CV, one threshold for all users)

| Setting | Models | Acc | ECE raw → cal | AUROC | 80% cov: coverage / accepted acc / false per 100 | 85% target: coverage / accepted acc / false per 100 |
|---|---|---|---|---|---|---|
| within | 1 | 75.60 | 4.76 → 2.64 | 0.808 | 81.18 / 82.40 / 14.3 | 75.15 / 84.83 / 11.4 |
| within | 5 | 79.13 | 3.57 → 2.94 | 0.823 | 82.45 / 85.87 / 11.7 | 85.57 / 84.85 / 13.0 |
| within | 15 | 79.48 | 5.10 → 1.83 | 0.827 | 80.67 / 86.61 / 10.8 | 83.83 / 85.64 / 12.0 |
| within_online48 | 1 | 72.86 | 7.48 → 3.31 | 0.801 | 82.71 / 79.37 / 17.1 | 76.71 / 81.79 / 14.0 |
| within_online48 | 5 | 75.97 | 2.64 → 3.74 | 0.820 | 83.24 / 82.26 / 14.8 | 86.11 / 80.97 / 16.4 |
| within_online48 | 15 | 76.94 | 2.96 → 1.99 | 0.815 | 80.19 / 84.30 / 12.6 | 83.84 / 82.83 / 14.4 |

Per subject (offline EA reference, largest ensemble): accuracy | 80% cov accepted acc | 85% target coverage

- A01: 88.19 | 88.81 | 100.00
- A02: 62.50 | 73.56 | 66.32
- A03: 93.06 | 94.01 | 99.31
- A04: 71.88 | 80.30 | 73.26
- A05: 69.10 | 75.21 | 86.46
- A06: 61.81 | 85.71 | 32.99
- A07: 90.62 | 91.20 | 98.96
- A08: 90.28 | 91.49 | 98.61
- A09: 87.85 | 89.61 | 98.61

Old Stage B (EA ensemble of 5, before S&R): accuracy 76.16, 80% coverage 82.9 / 82.0 / 14.9; 85% target not reached (82.8 at 81.2% coverage)