# Project handoff: EEG Motor-Imagery Wheelchair Control (BTP)

> **For a new Claude session:** read this file first. It summarises everything decided and done so far (2026-09-23 → 2026-10-20). The full raw chat history is in `claude_context/` of the transfer zip.

## 1. The project
**EEG-Based Motor-Imagery Wheelchair Control with Cross-Subject Generalization and Uncertainty-Aware Command Rejection.**
- **Research question:** can cross-subject EEG generalization (Euclidean Alignment, EA) combined with uncertainty-aware command rejection (temperature scaling → predictive entropy → reject option) make EEG wheelchair control more reliable for unseen users?
- **Dataset:** BCI Competition IV-2a: 9 subjects, 22 EEG + 3 EOG channels, 250 Hz, 4 classes (left hand, right hand, feet, tongue), 2 sessions (T and E, recorded on different days), 288 trials per session (6 runs × 48, 72 per class).
- **Wheelchair mapping (ours, not in the dataset):** left hand → LEFT, right hand → RIGHT, feet → FORWARD, tongue → STOP.
- **Core papers (`Research papers/`):** Thapa 2026 (TFormerEEG wheelchair), An 2025 (multimodal wheelchair), Junqueira 2024 (EA + deep learning), Nzakuna 2025 (LEMC-FTDB uncertainty; **Table 2 transcribed** in `results/reference_nzakuna2025_table2.csv`). Mentor's paper: Sreeja & Samanta 2019, Neurocomputing (sparsity + wavelet energy + OMP).
- **Novelty claim (careful):** we do not invent EA, EEGNet or entropy rejection. The contribution is a controlled evaluation combining alignment with uncertainty-aware selective command acceptance, including how EA changes rejection behaviour.

## 2. User preferences (important)
- **Plan first, show the plan, implement only after approval.** The user decides scope.
- **Do not add Claude as co-author** on commits (no `Co-Authored-By` line).
- Do not push the dataset, `docs/` or papers to GitHub. Show the file list before pushing.
- Keep our pipeline; mentor techniques were **added**, not substituted.
- Main metric: **per-subject + average accuracy** on the standard protocol (train T → test E). Cross-subject (LOSO) is also reported.
- Long training runs unattended with the overnight orchestrator (no Claude needed while training).

## 3. Environment
```
conda env create -f environment.yml        # creates "eeg-bci" (Python 3.11)
python -m ipykernel install --user --name eeg-bci --display-name "Python (eeg-bci)"
```
- PyTorch CUDA 12.6 wheels (original machine: RTX 4060 Laptop, 8 GB). Change the index URL if the GPU or CUDA differs; CPU works but is slow.
- **BLAS pinned to OpenBLAS:** conda-forge MKL crashed (0xC06D007F) when the env was not activated (Jupyter kernels).
- `data_processed/` is **not** in the zip. It is rebuilt automatically from `Dataset/` the first time it is needed (`src/preprocess.py: load_config_data`).
- Original Windows paths: project `C:\Users\karpu\OneDrive\Desktop\BTP\wheel-chair-movement-using-EEG`, env Python `C:\Users\karpu\anaconda3\envs\eeg-bci\python.exe`. All code uses paths relative to the project root.
- GitHub: `https://github.com/rehanshaik108834-collab/wheel-chair-movement-using-EEG` (main; last push = commit cf2b4bd, notebooks 01–04 + code up to then). **Not yet pushed:** notebook 05, Stage A code/results, tests, mentor pipeline.

## 4. Pipeline (final, "option A")
Raw GDF → drop 3 EOG → **8–30 Hz zero-phase FIR** (continuous signal) → epoch **0–4 s after cue** (cue at trial t = 2 s) → 22 × 1000 µV → per-channel z-score (training data only) → **EEGNet** (own PyTorch implementation of Lawhern 2018: F1 = 8, D = 2, F2 = 16, temporal kernel 125, separable kernel 31, 4,172 parameters; verified **identical outputs** to braindecode's EEGNet).
- Within-subject training: Adam 1e-3, batch 32, dropout 0.5, **446 epochs** (chosen by CV inside T), no early stopping.
- Cross-subject (LOSO): train on 8 subjects' T+E, 20% held out for early stopping, dropout 0.25, batch 64.
- cuDNN benchmark mode (1.8× faster; reproducible up to floating-point noise).
- **E-session labels:** official `true_labels.zip` from bbci.de → `Dataset/true_labels/` (the E GDFs only carry cue 783).

## 5. Results so far (session E unless stated; chance = 25%)
| Notebook | What | Key numbers |
|---|---|---|
| 01 | Data exploration | 264/2592 T trials expert-marked artifacts (kept); no clipping inside MI windows |
| 02 | Preprocessing | C3 mu ERD: right hand −17% vs left −1.3% (A01) |
| 03 | EEGNet baseline (T only, preliminary) | within 60.8% (ours) / 60.4% (braindecode), r = 0.99; LOSO 43.6% |
| 04 | **Standard protocol T → E** | **68.20%** single (5 seeds; κ 0.576, F1 0.673); option A chosen on T-CV (70.5%) at 446 epochs; options B/C/D on E: 68.0/65.8/66.7; Nzakuna MCD 71.14% (difference not significant, p = 0.25); LOSO no-EA 44.66% |
| (from 04 logits) | 5-seed ensemble | **73.53%** |
| 05 | **Mentor's pipeline added** | Motor channels −3.0, Chebyshev II filter −3.2, 2 s windows +0.6 (none kept by the +0.5 pp CV rule). Her classifier best = 9 sub-bands + CSP + wavelet energy + OMP (R4): 52.6% on E. **Fusion EEGNet + her classifier: 70.10% single** (8/9 subjects improve), ensemble 73.07%. Her segment-random split inflates accuracy: 78.9% vs 57.8% trial-level CV vs 52.6% E. Her classifier LOSO 28.6% |
| Stage A (`results/ea/summary.md`, **notebook 06 not built yet**) | **Euclidean Alignment** | **LOSO: 44.44 → 52.85% (+8.4, p = 0.004)**; pseudo-online 48 trials +6.6 (p = 0.008), 24 trials +6.1 (p = 0.004); LOSO ensemble 54.63%. EA needs ~43% fewer epochs (272 vs 476). **Within-subject EA: 71.40% single, 76.16% ensemble (new best; > Nzakuna EMCD 75.35%)**, using all unlabelled E trials for the reference; realistic 48-trial version 69.18 / 73.06%. Weak subjects gain most (A02 46.9 → 59.1, A06 LOSO 28.7 → 43.5) |

## 6. Code map
| Path | Purpose |
|---|---|
| `src/data.py` | Loading GDF, channel names, events, official labels (`labelled_events`) |
| `src/preprocess.py` | Pipeline options A/B/C/D/AC, cached arrays (`load_config_data`) |
| `src/models.py` | EEGNet |
| `src/train.py` | Training loop, ChannelScaler, metrics, seeds |
| `src/plots.py` | Per-subject curve/confusion/recall grids |
| `src/results_io.py` | Loading saved runs |
| `src/mentor_pipeline.py` | Mentor's pipeline (bands, CSP, segmentation, wavelet energy, batched OMP classifier R1–R4) |
| `src/alignment.py` | Euclidean Alignment |
| `tests/` | `test_mentor_pipeline.py`, `test_alignment.py` (all pass) |
| `experiments/run_experiments.py` | Notebook 04 runs (select / choose / t2e / loso), `run_all.ps1` |
| `experiments/run_mentor.py` | Notebook 05 runs; `extra_p0.py` (protocol comparison) |
| `experiments/run_ea.py` | Stage A runs (within_cv / within_choose / within_t2e / loso / final) |
| `experiments/overnight.py --plan mentor|ea` + `launch_overnight.py` | Unattended orchestrator: retries, stall watchdog, keeps PC awake, status in `results/overnight_status*.txt` |
| `notebooks/01–05` | Executed notebooks with outputs |
| `results/` | All metrics/predictions/logits (`runs/`, `mentor/`, `ea/`), figures, CSVs |
| `checkpoints/` | Every trained model (weights + scaler + settings) |
| `docs/` | Mentor meeting script (md + pdf), note for mentor on her pipeline |

Every experiment phase **resumes**: finished runs (one JSON each) are skipped when re-run. `--smoke` runs a tiny end-to-end test.

## 7. Next steps (in order)
1. **Notebook 06 (Stage A write-up)** from `results/ea/`: tables, covariance heatmaps before/after EA, convergence curves (stored `history` incl. `target_acc_monitor`), class-collapse check (A05/A06 confusion matrices), pseudo-online vs offline. No training needed.
2. **Stage B: calibration + rejection.** Temperature scaling (fit T on training-subject validation data only), ECE/Brier/NLL, reliability diagrams, predictive entropy, reject threshold chosen on validation data, coverage, selective risk, false actionable commands (also per command, e.g. false FORWARD), risk–coverage curves, AURC, misclassification AUROC. All logits/probs are already saved (`results/runs/`, `results/mentor/`, `results/ea/`).
3. **Stage C:** EA + rejection vs no-EA + rejection under LOSO: does EA change rejection behaviour?
4. Stage D only if needed: MC Dropout → LEMC-FTDB.
5. Push notebooks 05–06 and new code to GitHub (ask the user first; no Claude co-author).
6. Update the mentor meeting script/PDF with notebook 05/06 results if asked.
