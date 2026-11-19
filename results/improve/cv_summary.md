# Part 1 screening: session-T CV (EA EEGNet), accuracy at the best epoch of the smoothed mean curve

Keep rule: at least +0.5 points over BASE on the same folds and seeds.

| Variant | Best epoch | T-CV acc | vs BASE | Subjects better | Wilcoxon p | Runs | Keep |
|---|---|---|---|---|---|---|---|
| AUGSR | 290 | 75.15 | +2.91 | 8/9 | 0.008 | 54 | yes |
| BASE | 290 | 72.25 | +0.00 | 0/9 | nan | 54 |  |
| AUGNS | 180 | 71.86 | -0.39 | 5/9 | 0.820 | 54 |  |
| RECIPE | 370 | 71.77 | -0.48 | 3/9 | 0.398 | 54 |  |
| CLEAN | 390 | 70.25 | -1.99 | 2/9 | 0.027 | 54 |  |
| FTDB | 230 | 69.43 | -2.82 | 1/9 | 0.008 | 54 |  |
| PREPD | 400 | 67.04 | -5.21 | 3/9 | 0.359 | 54 |  |
