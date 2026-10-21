"""Stage A experiments: Euclidean Alignment (EA) with our EEGNet pipeline (option A: 8-30 Hz, 0-4 s).

Within-subject (train session T, test session E of the same person)
  within_cv      EA inside T-CV: training runs aligned with their own reference, the held-out run
                 aligned with ITS own reference (as a new session would be); accuracy every epoch
  within_choose  epoch count = peak of the smoothed mean CV curve (T only)
  within_t2e     T aligned with R_T; E aligned with R_E from all E trials (offline) and from the
                 first 24 / 48 E trials (pseudo-online); 5 seeds
Cross-subject (train 8 subjects' T+E, test the unseen subject's E)
  loso           NOEA and EA, 3 seeds, up to 600 epochs, early stopping (patience 80) on 20% of the
                 training subjects' trials. EA: every training subject-session aligned with its own
                 reference; the unseen subject aligned offline / with its first 24 / 48 trials.
  final          summary tables -> results/ea/summary.json + summary.md
No labels of a test session / test subject are ever used (EA is unsupervised).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import alignment as ea  # noqa: E402
import run_mentor as rm  # noqa: E402
from data import SUBJECTS  # noqa: E402
from preprocess import load_config_data  # noqa: E402
from run_mentor import Trainer, acc, load_json, save_json, smooth  # noqa: E402

OUT = ROOT / "results" / "ea"
CKPT = ROOT / "checkpoints" / "ea"
RUN_OF_TRIAL = np.repeat(np.arange(6), 48)
SPEC = {"windows": False}
WITHIN = {"lr": 1e-3, "batch_size": 32, "dropout": 0.5, "epochs": 500}
CROSS = {"lr": 1e-3, "batch_size": 64, "dropout": 0.25, "max_epochs": 600, "patience": 80}
REFS = {"offline": None, "online24": 24, "online48": 48}
SMOOTH = 25
LOG = OUT / "log.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def align_target(X, k):
    """Align a test session with a reference from all its trials (k=None) or its first k trials."""
    return ea.align(X, ea.reference_covariance(X if k is None else X[:k]))


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


# ------------------------------------------------------------------------------ within-subject
def phase_within_cv(seed=0):
    from train import ChannelScaler
    for s in SUBJECTS:
        X, y = load_config_data("A", s, "T")
        for run in range(6):
            path = OUT / "within_cv" / f"{s}_run{run + 1}.json"
            if path.exists():
                continue
            t0 = time.time()
            te = RUN_OF_TRIAL == run
            Xtr, _ = ea.align_self(X[~te])                    # training runs: their own reference
            Xte, _ = ea.align_self(X[te])                     # held-out run: its own reference
            sc = ChannelScaler().fit(Xtr)
            tr = Trainer(SPEC, 22, seed * 100 + run, WITHIN["dropout"], WITHIN["lr"], WITHIN["batch_size"])
            Xt, yt = tr.to_dev(sc.transform(Xtr), y[~te])
            Xm = tr.prepare_eval(sc.transform(Xte))
            mon = []
            for _ in range(WITHIN["epochs"]):
                tr.epoch(Xt, yt)
                mon.append(acc(y[te], tr.trial_probs(Xm)))
            save_json(path, {"subject": s, "held_out_run": run + 1, "mon_acc": mon,
                             "seconds": round(time.time() - t0, 1)})
            log(f"within_cv EA {s} run{run + 1}: final held-out acc {mon[-1]:.3f} ({time.time() - t0:.1f} s)")


def phase_within_choose():
    runs = [load_json(OUT / "within_cv" / f"{s}_run{r}.json") for s in SUBJECTS for r in range(1, 7)]
    sm = smooth(np.mean([x["mon_acc"] for x in runs], axis=0), SMOOTH)
    best = int(np.argmax(sm)) + 1
    base = rm.cv_summary("BASE") if (rm.OUT / "eeg_cv" / "BASE").exists() else None
    sel = {"best_epochs": best, "cv_acc": float(sm[best - 1]), "base_cv": base}
    save_json(OUT / "within_selection.json", sel)
    log(f"within_choose: EA CV accuracy {sm[best - 1]:.3f} at epoch {best}"
        + (f" (no-EA BASE: {base['cv_acc']:.3f})" if base else ""))


def phase_within_t2e(seeds):
    from train import ChannelScaler
    epochs = load_json(OUT / "within_selection.json")["best_epochs"]
    for seed in seeds:
        for s in SUBJECTS:
            path = OUT / "within_t2e" / f"{s}_seed{seed}.json"
            if path.exists():
                continue
            t0 = time.time()
            X, y = load_config_data("A", s, "T")
            Xe, ye = load_config_data("A", s, "E")
            Xtr, _ = ea.align_self(X)
            sc = ChannelScaler().fit(Xtr)
            tr = Trainer(SPEC, 22, seed, WITHIN["dropout"], WITHIN["lr"], WITHIN["batch_size"])
            Xt, yt = tr.to_dev(sc.transform(Xtr), y)
            for _ in range(epochs):
                tr.epoch(Xt, yt)
            probs = {r: tr.trial_probs(tr.prepare_eval(sc.transform(align_target(Xe, k)))) for r, k in REFS.items()}
            res = {"subject": s, "seed": seed, "epochs": epochs, "y": ye.tolist(),
                   "probs": {r: np.round(p, 5).tolist() for r, p in probs.items()},
                   "acc": {r: acc(ye, p) for r, p in probs.items()}, "seconds": round(time.time() - t0, 1)}
            rm.save_ckpt(CKPT / "within_t2e" / f"{s}_seed{seed}.pt", tr, sc, {"epochs": epochs, "ea": True})
            save_json(path, res)
            log(f"within_t2e EA {s} seed{seed}: E acc offline {res['acc']['offline']:.3f}, "
                f"online48 {res['acc']['online48']:.3f}, online24 {res['acc']['online24']:.3f}")


# ------------------------------------------------------------------------------ cross-subject
def phase_loso(seeds, variants=("NOEA", "EA")):
    from sklearn.model_selection import train_test_split
    from train import ChannelScaler
    data = {(s, sess): load_config_data("A", s, sess) for s in SUBJECTS for sess in "TE"}
    aligned = {k: ea.align_self(v[0])[0] for k, v in data.items()}
    for seed in seeds:
        for variant in variants:
            for target in SUBJECTS:
                path = OUT / "loso" / variant / f"{target}_seed{seed}.json"
                if path.exists():
                    continue
                t0 = time.time()
                src = [(s, sess) for s in SUBJECTS if s != target for sess in "TE"]
                X = np.concatenate([(aligned[k] if variant == "EA" else data[k][0]) for k in src])
                y = np.concatenate([data[k][1] for k in src])
                strata = np.concatenate([np.full(288, i * 4) for i in range(len(src))]) + y
                tri, vai = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=strata)
                sc = ChannelScaler().fit(X[tri])
                tr = Trainer(SPEC, 22, seed, CROSS["dropout"], CROSS["lr"], CROSS["batch_size"])
                Xt, yt = tr.to_dev(sc.transform(X[tri]), y[tri])
                Xv = tr.prepare_eval(sc.transform(X[vai]))
                Xe, ye = data[(target, "E")]
                Xmon = tr.prepare_eval(sc.transform(align_target(Xe, None) if variant == "EA" else Xe))
                hist = {"val_loss": [], "val_acc": [], "target_acc_monitor": []}
                best, best_state, best_ep, wait = np.inf, None, 0, 0
                for ep in range(1, CROSS["max_epochs"] + 1):
                    tr.epoch(Xt, yt)
                    pv = tr.trial_probs(Xv)
                    hist["val_loss"].append(nll(pv, y[vai])); hist["val_acc"].append(acc(y[vai], pv))
                    hist["target_acc_monitor"].append(acc(ye, tr.trial_probs(Xmon)))   # plotting only
                    if hist["val_loss"][-1] < best:
                        best, best_ep, wait = hist["val_loss"][-1], ep, 0
                        best_state = {k: t.detach().clone() for k, t in tr.model.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= CROSS["patience"]:
                            break
                tr.model.load_state_dict(best_state)
                if variant == "EA":
                    probs = {r: tr.trial_probs(tr.prepare_eval(sc.transform(align_target(Xe, k)))) for r, k in REFS.items()}
                else:
                    probs = {"none": tr.trial_probs(tr.prepare_eval(sc.transform(Xe)))}
                res = {"variant": variant, "subject": target, "seed": seed, "best_epoch": best_ep,
                       "epochs_run": len(hist["val_loss"]), "y": ye.tolist(), "history": hist,
                       "probs": {r: np.round(p, 5).tolist() for r, p in probs.items()},
                       "acc": {r: acc(ye, p) for r, p in probs.items()}, "seconds": round(time.time() - t0, 1)}
                rm.save_ckpt(CKPT / "loso" / variant / f"{target}_seed{seed}.pt", tr, sc,
                             {"variant": variant, "best_epoch": best_ep})
                save_json(path, res)
                log(f"loso {variant} {target} seed{seed}: E acc {', '.join(f'{r} {a:.3f}' for r, a in res['acc'].items())}"
                    f"; best epoch {best_ep}/{res['epochs_run']} ({res['seconds']} s)")


# ------------------------------------------------------------------------------ summary
def phase_final():
    from scipy.stats import wilcoxon
    out = {}

    def per_subject(runs, ref, from_trial=0):
        by = {}
        for r in runs:
            y = np.array(r["y"])[from_trial:]
            p = np.array(r["probs"][ref])[from_trial:]
            by.setdefault(r["subject"], []).append((y, p))
        single = {s: float(np.mean([acc(y, p) for y, p in by[s]])) for s in SUBJECTS}
        ens = {s: acc(by[s][0][0], np.mean([p for _, p in by[s]], 0)) for s in SUBJECTS}
        single["mean"], ens["mean"] = float(np.mean(list(single.values()))), float(np.mean(list(ens.values())))
        return {"single": single, "ensemble": ens}

    # within-subject: BASE (no EA) from notebook 04 runs
    base = []
    for f in sorted((ROOT / "results" / "runs" / "t2e" / "A").glob("*.json")):
        r = load_json(f)
        lg = np.array(r["logits"]); p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        base.append({"subject": r["subject"], "y": r["y_true"], "probs": {"none": p.tolist()}})
    within = [load_json(f) for f in sorted((OUT / "within_t2e").glob("*.json"))]
    loso = {v: [load_json(f) for f in sorted((OUT / "loso" / v).glob("*.json"))] for v in ("NOEA", "EA")}
    tables = {}
    for name, runs, ref in [("within_noEA", base, "none"), ("within_EA_offline", within, "offline"),
                            ("loso_noEA", loso["NOEA"], "none"), ("loso_EA_offline", loso["EA"], "offline")]:
        tables[name] = per_subject(runs, ref)
    for k in (24, 48):           # pseudo-online: score only trials after the calibration block, for every method
        tables[f"within_noEA_from{k}"] = per_subject(base, "none", k)
        tables[f"within_EA_online{k}"] = per_subject(within, f"online{k}", k)
        tables[f"loso_noEA_from{k}"] = per_subject(loso["NOEA"], "none", k)
        tables[f"loso_EA_online{k}"] = per_subject(loso["EA"], f"online{k}", k)
    out["tables"] = tables
    tests = {}
    for a, b in [("loso_EA_offline", "loso_noEA"), ("loso_EA_online48", "loso_noEA_from48"),
                 ("loso_EA_online24", "loso_noEA_from24"), ("within_EA_offline", "within_noEA"),
                 ("within_EA_online48", "within_noEA_from48")]:
        x = [tables[a]["single"][s] for s in SUBJECTS]; z = [tables[b]["single"][s] for s in SUBJECTS]
        try:
            p = float(wilcoxon(x, z).pvalue)
        except ValueError:                       # e.g. all differences zero
            p = float("nan")
        tests[f"{a} vs {b}"] = {"mean_diff_pp": 100 * float(np.mean(np.subtract(x, z))), "wilcoxon_p": p}
    out["tests"] = tests
    out["loso_best_epoch"] = {v: float(np.mean([r["best_epoch"] for r in loso[v]])) for v in loso}
    out["within_selection"] = load_json(OUT / "within_selection.json")
    save_json(OUT / "summary.json", out)
    lines = ["# Stage A (Euclidean Alignment): summary", "",
             "| Setting | " + " | ".join(SUBJECTS) + " | Mean |", "|" + "---|" * (len(SUBJECTS) + 2)]
    for name, t in tables.items():
        for kind in ("single", "ensemble"):
            lines.append(f"| {name} ({kind}) | " + " | ".join(f"{100 * t[kind][s]:.1f}" for s in SUBJECTS)
                         + f" | **{100 * t[kind]['mean']:.2f}** |")
    lines += ["", "Paired tests (single-model accuracy, 9 subjects):"]
    lines += [f"- {k}: {v['mean_diff_pp']:+.2f} pp, Wilcoxon p = {v['wilcoxon_p']:.3f}" for k, v in tests.items()]
    lines += ["", f"Mean best epoch (LOSO): {out['loso_best_epoch']}"]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log("final: summary written to results/ea/summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        OUT, CKPT = ROOT / "results" / "ea_smoke", ROOT / "checkpoints" / "ea_smoke"
        SUBJECTS = SUBJECTS[:2]
        rm.SUBJECTS = SUBJECTS
        WITHIN["epochs"] = 6
        CROSS.update({"max_epochs": 4, "patience": 2})
        SMOOTH = 3
    LOG = OUT / "logs" / f"{a.phase}.txt"
    log(f"=== {a.phase}{' (smoke)' if a.smoke else ''} ===")
    {"within_cv": phase_within_cv, "within_choose": phase_within_choose,
     "within_t2e": lambda: phase_within_t2e(a.seeds or [0, 1, 2, 3, 4]),
     "loso": lambda: phase_loso(a.seeds or [0, 1, 2]), "final": phase_final}[a.phase]()
    log(f"=== {a.phase} finished ===")
