# Mentor pipeline: overnight summary

FINAL EEGNet variant: **BASE** (our pipeline (option A)); kept additions: []
Her classifier: **bp_csp_b9_all_s2_energy**, rule R4, CV accuracy 0.578
Fusion weight 0.9 (kept: True)

| Pipeline | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE (single) | 81.5 | 46.9 | 89.7 | 59.5 | 54.4 | 49.9 | 70.8 | 78.8 | 82.3 | **68.20** |
| BASE (ensemble) | 87.2 | 48.6 | 92.7 | 67.4 | 61.8 | 60.8 | 78.5 | 80.2 | 84.7 | **73.53** |
| E1 (single) | 78.4 | 46.9 | 87.5 | 50.9 | 49.3 | 47.8 | 69.1 | 76.0 | 80.8 | **65.21** |
| E1 (ensemble) | 82.3 | 50.7 | 89.2 | 57.6 | 56.6 | 51.0 | 75.0 | 77.1 | 81.9 | **69.06** |
| E2 (single) | 79.2 | 49.7 | 88.3 | 56.9 | 35.0 | 48.5 | 71.2 | 76.9 | 79.7 | **65.04** |
| E2 (ensemble) | 84.4 | 54.5 | 88.9 | 62.5 | 40.6 | 53.5 | 76.7 | 80.2 | 81.9 | **69.25** |
| E3 (single) | 83.6 | 48.5 | 85.6 | 61.5 | 54.6 | 50.1 | 77.0 | 77.2 | 81.5 | **68.84** |
| E3 (ensemble) | 87.5 | 50.7 | 87.8 | 69.1 | 60.4 | 56.6 | 79.5 | 81.2 | 85.4 | **73.15** |
| SPARSE (single) | 64.2 | 37.8 | 59.0 | 49.3 | 36.5 | 41.7 | 64.9 | 61.5 | 58.7 | **52.62** |
| FINAL_FUSED (single) | 82.3 | 48.1 | 88.5 | 64.5 | 55.3 | 51.0 | 75.0 | 81.5 | 84.7 | **70.10** |
| FINAL_FUSED (ensemble) | 86.8 | 48.3 | 89.9 | 68.4 | 58.7 | 53.1 | 82.3 | 83.7 | 86.5 | **73.07** |

Cross-subject: {"sparse": 0.2863}
Her protocol (segment-level, random split): {"none_b0_motor_s2_energy": 0.2738}