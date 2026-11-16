# Stage B (calibration + rejection): summary

Temperature T and thresholds from calibration data only; test = session E / unseen subject.

| Setting | Model | Acc | T | ECE raw → cal | AUROC (entropy) | AURC | 80% cov: coverage / accepted acc / false per 100 | Target acc: coverage / accepted acc / false per 100 | 80% + artifact rule: coverage / accepted acc / false per 100 |
|---|---|---|---|---|---|---|---|---|---|
| within_noEA | single | 68.2 | 1.56 | 10.8 → 3.2 | 0.762 | 0.147 | 82.6 / 73.9 / 21.6 | 59.1 / 81.9 / 10.7 | 83.3 / 71.2 / 24.0 |
| within_noEA | ensemble | 73.5 | 0.91 | 2.8 → 1.8 | 0.783 | 0.107 | 83.3 / 78.5 / 17.9 | 72.5 / 82.6 / 12.6 | 83.3 / 76.4 / 19.7 |
| within_EA | single | 71.4 | 1.48 | 7.4 → 2.7 | 0.785 | 0.120 | 81.6 / 78.2 / 17.8 | 66.2 / 83.5 / 10.9 | 83.1 / 75.4 / 20.4 |
| within_EA | ensemble | 76.2 | 0.79 | 4.2 → 2.9 | 0.807 | 0.085 | 82.9 / 82.0 / 14.9 | 81.2 / 82.8 / 14.0 | 83.7 / 79.5 / 17.2 |
| within_noEA_from48 | single | 67.5 | 1.56 | 11.3 → 3.5 | 0.761 | 0.152 | 82.6 / 73.1 / 22.2 | 59.0 / 81.4 / 11.0 | 83.1 / 70.5 / 24.5 |
| within_noEA_from48 | ensemble | 73.1 | 0.91 | 3.2 → 2.0 | 0.780 | 0.111 | 83.0 / 78.0 / 18.2 | 71.9 / 82.3 / 12.7 | 83.2 / 76.1 / 19.9 |
| within_EA_online48 | single | 69.2 | 1.48 | 10.0 → 3.9 | 0.773 | 0.139 | 83.4 / 75.0 / 20.8 | 67.7 / 81.0 / 12.9 | 84.3 / 72.4 / 23.3 |
| within_EA_online48 | ensemble | 73.1 | 0.79 | 2.7 → 3.8 | 0.805 | 0.102 | 82.6 / 79.7 / 16.8 | 81.1 / 80.0 / 16.2 | 83.1 / 76.6 / 19.5 |
| cross_noEA | single | 44.4 | 4.07 | 25.2 → 6.2 | 0.552 | 0.513 | 78.7 / 47.5 / 41.3 | unreachable | 78.9 / 46.5 / 42.2 |
| cross_noEA | ensemble | 45.6 | 3.21 | 19.6 → 6.8 | 0.555 | 0.502 | 78.4 / 49.7 / 39.5 | unreachable | 78.9 / 48.5 / 40.6 |
| cross_EA | single | 52.9 | 2.07 | 17.3 → 2.4 | 0.644 | 0.351 | 79.2 / 57.6 / 33.6 | 49.5 / 62.0 / 18.9 | 79.4 / 56.0 / 34.9 |
| cross_EA | ensemble | 54.6 | 1.67 | 11.3 → 2.7 | 0.654 | 0.332 | 79.0 / 59.5 / 32.0 | 57.9 / 62.8 / 21.5 | 79.4 / 58.3 / 33.1 |
| cross_noEA_from48 | single | 44.0 | 4.24 | 25.7 → 6.8 | 0.542 | 0.530 | 78.8 / 46.8 / 41.9 | unreachable | 79.0 / 46.0 / 42.7 |
| cross_noEA_from48 | ensemble | 45.0 | 3.32 | 20.3 → 6.9 | 0.546 | 0.522 | 78.6 / 48.9 / 40.2 | unreachable | 78.9 / 47.5 / 41.4 |
| cross_EA_online48 | single | 50.5 | 2.33 | 20.5 → 2.8 | 0.641 | 0.384 | 79.2 / 55.4 / 35.3 | 40.5 / 61.8 / 15.5 | 79.4 / 54.1 / 36.5 |
| cross_EA_online48 | ensemble | 52.5 | 1.89 | 14.3 → 3.4 | 0.641 | 0.367 | 79.1 / 56.3 / 34.5 | 49.0 / 60.1 / 19.6 | 78.9 / 55.1 / 35.4 |
| cross_noEA_srcval | single | 44.4 | 0.93 | 25.2 → 26.8 | 0.626 | 0.439 | 88.2 / 46.4 / 47.3 | 100.0 / 44.5 / 55.5 | 86.2 / 45.5 / 47.0 |
| cross_EA_srcval | single | 52.9 | 1.01 | 17.3 → 17.2 | 0.671 | 0.327 | 73.9 / 59.0 / 30.3 | 99.9 / 52.9 / 47.1 | 76.5 / 56.9 / 33.0 |

Paired tests over the 9 subjects (Wilcoxon):
- within_EA vs within_noEA | ensemble | aurc: 0.1182 vs 0.1387, better for 7/9, p = 0.098
- within_EA vs within_noEA | ensemble | ece_cal: 0.0614 vs 0.0778, better for 5/9, p = 0.301
- within_EA vs within_noEA | ensemble | false_per_100_at_cov80: 14.9306 vs 17.9012, better for 6/9, p = 0.312
- within_EA vs within_noEA | ensemble | accepted_accuracy_at_cov80: 0.8016 vs 0.7733, better for 5/9, p = 0.359
- within_EA vs within_noEA | single | aurc: 0.1692 vs 0.1987, better for 7/9, p = 0.074
- within_EA vs within_noEA | single | ece_cal: 0.0595 vs 0.0789, better for 8/9, p = 0.008
- within_EA vs within_noEA | single | false_per_100_at_cov80: 17.8086 vs 21.5664, better for 7/9, p = 0.039
- within_EA vs within_noEA | single | accepted_accuracy_at_cov80: 0.7613 vs 0.7216, better for 8/9, p = 0.027
- within_EA_online48 vs within_noEA_from48 | ensemble | aurc: 0.1351 vs 0.1492, better for 5/9, p = 0.734
- within_EA_online48 vs within_noEA_from48 | ensemble | ece_cal: 0.0747 vs 0.0839, better for 4/9, p = 1.000
- within_EA_online48 vs within_noEA_from48 | ensemble | false_per_100_at_cov80: 16.7593 vs 18.2407, better for 3/9, p = 0.984
- within_EA_online48 vs within_noEA_from48 | ensemble | accepted_accuracy_at_cov80: 0.7788 vs 0.7660, better for 4/9, p = 0.910
- within_EA_online48 vs within_noEA_from48 | single | aurc: 0.1891 vs 0.2076, better for 5/9, p = 0.426
- within_EA_online48 vs within_noEA_from48 | single | ece_cal: 0.0719 vs 0.0864, better for 6/9, p = 0.164
- within_EA_online48 vs within_noEA_from48 | single | false_per_100_at_cov80: 20.8241 vs 22.1852, better for 4/9, p = 0.844
- within_EA_online48 vs within_noEA_from48 | single | accepted_accuracy_at_cov80: 0.7269 vs 0.7129, better for 5/9, p = 0.734
- cross_EA vs cross_noEA | ensemble | aurc: 0.3674 vs 0.4531, better for 9/9, p = 0.004
- cross_EA vs cross_noEA | ensemble | ece_cal: 0.1327 vs 0.1848, better for 6/9, p = 0.164
- cross_EA vs cross_noEA | ensemble | false_per_100_at_cov80: 31.9830 vs 39.4676, better for 7/9, p = 0.129
- cross_EA vs cross_noEA | ensemble | accepted_accuracy_at_cov80: 0.5731 vs 0.4881, better for 9/9, p = 0.004
- cross_EA vs cross_noEA | single | aurc: 0.3841 vs 0.4750, better for 9/9, p = 0.004
- cross_EA vs cross_noEA | single | ece_cal: 0.1387 vs 0.1668, better for 5/9, p = 0.496
- cross_EA vs cross_noEA | single | false_per_100_at_cov80: 33.5905 vs 41.3194, better for 7/9, p = 0.074
- cross_EA vs cross_noEA | single | accepted_accuracy_at_cov80: 0.5559 vs 0.4699, better for 9/9, p = 0.004
- cross_EA_online48 vs cross_noEA_from48 | ensemble | aurc: 0.3846 vs 0.4624, better for 9/9, p = 0.004
- cross_EA_online48 vs cross_noEA_from48 | ensemble | ece_cal: 0.1489 vs 0.1834, better for 6/9, p = 0.164
- cross_EA_online48 vs cross_noEA_from48 | ensemble | false_per_100_at_cov80: 34.5370 vs 40.1852, better for 6/9, p = 0.250
- cross_EA_online48 vs cross_noEA_from48 | ensemble | accepted_accuracy_at_cov80: 0.5421 vs 0.4819, better for 7/9, p = 0.027
- cross_EA_online48 vs cross_noEA_from48 | single | aurc: 0.3968 vs 0.4825, better for 8/9, p = 0.008
- cross_EA_online48 vs cross_noEA_from48 | single | ece_cal: 0.1483 vs 0.1692, better for 5/9, p = 0.426
- cross_EA_online48 vs cross_noEA_from48 | single | false_per_100_at_cov80: 35.3241 vs 41.8673, better for 7/9, p = 0.098
- cross_EA_online48 vs cross_noEA_from48 | single | accepted_accuracy_at_cov80: 0.5365 vs 0.4647, better for 9/9, p = 0.004