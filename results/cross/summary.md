# A new user (leave-one-subject-out), BCI IV-2a, 3-model ensembles

Offline: EA reference from all 288 E trials of the new user, scored on all 288.
Online: EA reference (and BN / fine-tuning data) from the first 48 E trials only, scored on trials 49-288.

| Setting | Method | 1 model | Ensemble | vs old: better for, p | Rejection 80% cov: coverage / accepted acc / false per 100 | 65% target: coverage / accepted acc |
|---|---|---|---|---|---|---|
| offline | old EA model, no adaptation | 52.85 | 54.63 | 0/9, p = 1.000 | 79.01 / 59.52 / 32.0 | 57.87 / 62.80 |
| offline | + S&R (B4), no adaptation | 54.44 | 55.98 | 5/9, p = 0.445 | 79.28 / 60.58 / 31.2 | 61.07 / 62.73 |
| offline | + BatchNorm adaptation on all unlabelled trials (B5) | 57.50 | 59.49 | 8/9, p = 0.012 | 78.97 / 64.53 / 28.0 | 78.86 / 63.45 |
| online | old EA model, no adaptation | 50.54 | 52.50 | 0/9, p = 1.000 | 79.07 / 56.32 / 34.5 | 49.03 / 60.06 |
| online | + S&R (B4), no adaptation | 52.53 | 54.35 | 5/9, p = 0.195 | 78.84 / 58.19 / 33.0 | 52.31 / 63.36 |
| online | + BatchNorm adaptation on the first 48 unlabelled trials (B5) | 55.52 | 57.13 | 8/9, p = 0.027 | 78.84 / 62.89 / 29.3 | 70.56 / 64.11 |
| online | + fine-tuning on the first 48 labelled trials (B6) | 62.04 | 63.75 | 9/9, p = 0.004 | 78.61 / 70.91 / 22.9 | 98.75 / 64.04 |

Per subject (ensemble): A01, A02, A03, A04, A05, A06, A07, A08, A09

- OLD_offline: 71.18, 34.03, 79.17, 46.18, 35.42, 47.92, 40.28, 72.57, 64.93
- NONE_offline: 73.26, 32.29, 82.99, 44.79, 43.75, 44.79, 45.49, 70.49, 65.97
- BN_offline: 78.47, 35.76, 82.99, 46.88, 43.06, 50.00, 56.25, 71.88, 70.14
- OLD_online: 75.00, 35.00, 72.92, 40.42, 34.17, 40.42, 42.08, 68.33, 64.17
- NONE_online: 71.25, 33.33, 77.08, 42.92, 42.50, 38.33, 47.92, 71.67, 64.17
- BN_online: 75.83, 32.08, 83.33, 45.83, 42.50, 41.67, 55.00, 71.25, 66.67
- FT48: 76.25, 43.75, 84.17, 54.17, 47.08, 46.67, 69.58, 76.67, 75.42