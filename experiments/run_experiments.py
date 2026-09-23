"""Experiment runner for notebook 04 (standard protocol on BCI IV-2a).

Every training run is saved as soon as it finishes:
  results/runs/<phase>/<config>/<run>.json   metrics, predictions, logits, training history
  checkpoints/<phase>/<config>/<run>.pt      model weights (state_dict) + run settings
Re-running a command skips runs whose .json already exists, so an interrupted job resumes.

Phases
  select   Phase 2: preprocessing options A-D, cross-validation inside the T session only
           (6 folds: train 5 runs, evaluate the held-out run after every epoch, 500 epochs).
  choose   Pick the option and epoch count with the best mean held-out accuracy (T data only).
  t2e      Phases 3/4: train on all 288 T trials for the chosen epochs, test once on E.
  loso     Phase 5: train on 8 subjects (T+E), test on the unseen subject (early stopping on
           20% of the training subjects' trials).

Usage (from the project root, with the eeg-bci Python):
  python experiments/run_experiments.py select --configs A B C D
  python experiments/run_experiments.py choose
  python experiments/run_experiments.py t2e --configs B --seeds 0 1 2 3 4
  python experiments/run_experiments.py loso --configs B --seeds 0 1 2
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import SUBJECTS  # noqa: E402
from models import EEGNet  # noqa: E402
from preprocess import CONFIGS, load_config_data  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from train import (DEVICE, ChannelScaler, _batches, classification_metrics,  # noqa: E402
                   evaluate_loss, predict_logits, set_seed, to_device, train_model)

RUNS_DIR = ROOT / "results" / "runs"
CKPT_DIR = ROOT / "checkpoints"
SELECTION_FILE = ROOT / "results" / "selection.json"
LOG_FILE = RUNS_DIR / "log.txt"
SFREQ = 250
RUN_OF_TRIAL = np.repeat(np.arange(6), 48)

FAST = True   # cuDNN benchmark mode: ~1.8x faster; seeds still fix initialisation and batch order
WITHIN = {"lr": 1e-3, "batch_size": 32, "dropout": 0.5, "epochs": 500, "cudnn_benchmark": FAST}
CROSS = {"lr": 1e-3, "batch_size": 64, "dropout": 0.25, "max_epochs": 300, "patience": 50,
         "cudnn_benchmark": FAST}
SMOOTH = 25   # epochs; moving average used when reading the mean CV accuracy curve


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_times(config):
    c = CONFIGS[config]
    return int(round((c["tmax"] - c["tmin"]) * SFREQ))


def run_paths(phase, config, name):
    return (RUNS_DIR / phase / config / f"{name}.json", CKPT_DIR / phase / config / f"{name}.pt")


def save_run(phase, config, name, record, model, scaler=None):
    """Checkpoint = weights + settings + the per-channel scaler, i.e. everything needed to reuse the model."""
    jpath, cpath = run_paths(phase, config, name)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "config": config, "settings": record["settings"]}
    if scaler is not None:
        ckpt["scaler_mean"], ckpt["scaler_std"] = scaler.mean_, scaler.std_
    torch.save(ckpt, cpath)
    tmp = jpath.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(jpath)            # atomic: a half-written file never counts as "done"


def to_list(a):
    return np.asarray(a).tolist()


def train_fixed_with_monitor(model, X_tr, y_tr, X_mon, y_mon, *, epochs, lr, batch_size, seed):
    """Train for a fixed number of epochs; after every epoch record loss/accuracy on a monitor set.

    The monitor set never influences training (no early stopping, no weight selection).
    """
    from torch import nn
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    X_tr, y_tr = to_device(X_tr, y_tr)
    X_mon, y_mon = to_device(X_mon, y_mon)
    hist = {"train_loss": [], "train_acc": [], "mon_loss": [], "mon_acc": []}
    for _ in range(epochs):
        model.train()
        run_loss, run_correct = 0.0, 0
        for xb, yb in _batches(X_tr, y_tr, batch_size, shuffle=True, rng=rng):
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            model.apply_constraints()
            run_loss += loss.item() * len(yb)
            run_correct += (logits.argmax(1) == yb).sum().item()
        hist["train_loss"].append(run_loss / len(X_tr))
        hist["train_acc"].append(run_correct / len(X_tr))
        m_loss, m_acc = evaluate_loss(model, X_mon, y_mon)
        hist["mon_loss"].append(m_loss)
        hist["mon_acc"].append(m_acc)
    return model, hist


# --------------------------------------------------------------------------- Phase 2
def phase_select(configs, seed=0):
    for config in configs:
        for subject in SUBJECTS:
            X, y = load_config_data(config, subject, "T")
            for test_run in range(6):
                name = f"{subject}_run{test_run + 1}_seed{seed}"
                if run_paths("select", config, name)[0].exists():
                    continue
                t0 = time.time()
                te = RUN_OF_TRIAL == test_run
                tr = ~te
                scaler = ChannelScaler().fit(X[tr])
                set_seed(seed * 100 + test_run, fast=FAST)
                model, hist = train_fixed_with_monitor(
                    EEGNet(n_times=X.shape[2], dropout=WITHIN["dropout"]),
                    scaler.transform(X[tr]), y[tr], scaler.transform(X[te]), y[te],
                    epochs=WITHIN["epochs"], lr=WITHIN["lr"], batch_size=WITHIN["batch_size"],
                    seed=seed * 100 + test_run)
                record = {"phase": "select", "config": config, "subject": subject, "held_out_run": test_run + 1,
                          "seed": seed, "settings": {**WITHIN, "n_train": int(tr.sum()), "n_times": X.shape[2]},
                          "history": hist, "final_heldout_acc": hist["mon_acc"][-1],
                          "seconds": round(time.time() - t0, 1)}
                save_run("select", config, name, record, model, scaler)
                log(f"select {config} {name}: final held-out acc {hist['mon_acc'][-1]:.3f} ({record['seconds']} s)")


def smooth(curve, k=SMOOTH):
    """Centered moving average (shorter window at the edges)."""
    c = np.asarray(curve, dtype=float)
    out = np.empty_like(c)
    h = k // 2
    for i in range(len(c)):
        out[i] = c[max(0, i - h): i + h + 1].mean()
    return out


def load_select_curves(config, seed=0):
    curves = {}
    for subject in SUBJECTS:
        for r in range(1, 7):
            p = run_paths("select", config, f"{subject}_run{r}_seed{seed}")[0]
            if p.exists():
                curves[(subject, r)] = json.loads(p.read_text())["history"]["mon_acc"]
    return curves


def phase_choose(configs, seed=0):
    summary = {}
    for config in configs:
        curves = load_select_curves(config, seed)
        expected = len(SUBJECTS) * 6
        if len(curves) != expected:
            raise SystemExit(f"config {config}: only {len(curves)}/{expected} selection runs finished")
        mean_curve = np.mean(list(curves.values()), axis=0)
        sm = smooth(mean_curve)
        best_epoch = int(np.argmax(sm)) + 1
        summary[config] = {"best_epochs": best_epoch, "cv_acc_at_best": float(sm[best_epoch - 1]),
                           "cv_acc_final_epoch": float(mean_curve[-1]), "label": CONFIGS[config]["label"]}
        log(f"choose {config}: best smoothed CV accuracy {sm[best_epoch - 1]:.3f} at epoch {best_epoch}")
    chosen = max(summary, key=lambda c: summary[c]["cv_acc_at_best"])
    result = {"chosen_config": chosen, "configs": summary, "smoothing_window": SMOOTH,
              "criterion": "max of the smoothed mean held-out-run accuracy (T session only, 9 subjects x 6 folds)"}
    SELECTION_FILE.write_text(json.dumps(result, indent=2))
    log(f"choose: selected config {chosen} ({CONFIGS[chosen]['label']}), {summary[chosen]['best_epochs']} epochs")
    return result


# --------------------------------------------------------------------------- Phases 3/4
def phase_t2e(configs, seeds):
    selection = json.loads(SELECTION_FILE.read_text())
    for config in configs:
        epochs = selection["configs"][config]["best_epochs"]
        for seed in seeds:
            for subject in SUBJECTS:
                name = f"{subject}_seed{seed}"
                if run_paths("t2e", config, name)[0].exists():
                    continue
                t0 = time.time()
                X_tr, y_tr = load_config_data(config, subject, "T")
                X_te, y_te = load_config_data(config, subject, "E")
                scaler = ChannelScaler().fit(X_tr)                    # fitted on T only
                set_seed(seed, fast=FAST)
                # E is only monitored (for plotting); it never affects training or weight choice.
                model, hist = train_fixed_with_monitor(
                    EEGNet(n_times=X_tr.shape[2], dropout=WITHIN["dropout"]),
                    scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te,
                    epochs=epochs, lr=WITHIN["lr"], batch_size=WITHIN["batch_size"], seed=seed)
                logits = predict_logits(model, scaler.transform(X_te))
                pred = logits.argmax(1)
                metrics = classification_metrics(y_te, pred)
                record = {"phase": "t2e", "config": config, "subject": subject, "seed": seed,
                          "settings": {**WITHIN, "epochs": epochs, "n_train": len(y_tr), "n_times": X_tr.shape[2]},
                          "metrics": metrics, "y_true": to_list(y_te), "y_pred": to_list(pred),
                          "logits": to_list(np.round(logits, 5)), "history": hist,
                          "seconds": round(time.time() - t0, 1)}
                save_run("t2e", config, name, record, model, scaler)
                log(f"t2e {config} {name}: E accuracy {metrics['accuracy']:.3f} kappa {metrics['kappa']:.3f} "
                    f"({record['seconds']} s)")


# --------------------------------------------------------------------------- Phase 5
def phase_loso(configs, seeds):
    for config in configs:
        data = {(s, sess): load_config_data(config, s, sess) for s in SUBJECTS for sess in "TE"}
        for seed in seeds:
            for target in SUBJECTS:
                name = f"{target}_seed{seed}"
                if run_paths("loso", config, name)[0].exists():
                    continue
                t0 = time.time()
                sources = [s for s in SUBJECTS if s != target]
                parts = [(data[(s, sess)], i, j) for i, s in enumerate(sources) for j, sess in enumerate("TE")]
                X_src = np.concatenate([p[0][0] for p in parts])
                y_src = np.concatenate([p[0][1] for p in parts])
                strata = np.concatenate([np.full(288, (i * 2 + j) * 4) for _, i, j in parts]) + y_src
                tr_idx, va_idx = train_test_split(np.arange(len(y_src)), test_size=0.2,
                                                  random_state=seed, stratify=strata)
                scaler = ChannelScaler().fit(X_src[tr_idx])            # training subjects only
                set_seed(seed, fast=FAST)
                model, hist = train_model(
                    EEGNet(n_times=X_src.shape[2], dropout=CROSS["dropout"]),
                    *to_device(scaler.transform(X_src[tr_idx]), y_src[tr_idx]),
                    *to_device(scaler.transform(X_src[va_idx]), y_src[va_idx]),
                    lr=CROSS["lr"], batch_size=CROSS["batch_size"], max_epochs=CROSS["max_epochs"],
                    patience=CROSS["patience"], seed=seed, full_train_eval=False)
                out = {}
                for sess in "TE":
                    X_t, y_t = data[(target, sess)]
                    logits = predict_logits(model, scaler.transform(X_t))
                    out[sess] = {"metrics": classification_metrics(y_t, logits.argmax(1)),
                                 "y_true": to_list(y_t), "y_pred": to_list(logits.argmax(1)),
                                 "logits": to_list(np.round(logits, 5))}
                record = {"phase": "loso", "config": config, "subject": target, "seed": seed,
                          "settings": {**CROSS, "n_train": len(tr_idx), "n_val": len(va_idx),
                                       "n_times": X_src.shape[2]},
                          "target": out, "history": hist, "seconds": round(time.time() - t0, 1)}
                save_run("loso", config, name, record, model, scaler)
                log(f"loso {config} {name}: target E acc {out['E']['metrics']['accuracy']:.3f}, "
                    f"target T acc {out['T']['metrics']['accuracy']:.3f}, best epoch {hist['best_epoch'] + 1} "
                    f"({record['seconds']} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["preprocess", "select", "choose", "t2e", "loso"])
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end test (2 subjects, few epochs) written to results/runs_smoke/")
    args = ap.parse_args()
    if args.smoke:
        RUNS_DIR = ROOT / "results" / "runs_smoke"
        CKPT_DIR = ROOT / "checkpoints_smoke"
        SELECTION_FILE = RUNS_DIR / "selection.json"
        LOG_FILE = RUNS_DIR / "log.txt"
        SUBJECTS = SUBJECTS[:2]
        WITHIN["epochs"], CROSS["max_epochs"], CROSS["patience"] = 3, 3, 2
    log(f"=== {args.phase} configs={args.configs} seeds={args.seeds} device={DEVICE} ===")

    if args.phase == "preprocess":
        for c in args.configs or list(CONFIGS):
            for s in SUBJECTS:
                for sess in "TE":
                    X, y = load_config_data(c, s, sess)
                log(f"preprocess {c} {s}: X{X.shape}")
    elif args.phase == "select":
        phase_select(args.configs or list(CONFIGS))
    elif args.phase == "choose":
        phase_choose(args.configs or list(CONFIGS))
    else:
        configs = args.configs or [json.loads(SELECTION_FILE.read_text())["chosen_config"]]
        (phase_t2e if args.phase == "t2e" else phase_loso)(configs, args.seeds)
    log(f"=== {args.phase} finished ===")
