"""Stage B experiments: calibration + uncertainty-aware rejection on top of Stage A (EA) and the no-EA baseline.

Labelled calibration data never comes from the test session / test subject:
  within-subject   leave-one-run-out CV inside session T (held-out logits at the final epoch count)
  cross-subject    "others": the unseen-subject predictions of the OTHER 8 LOSO folds (new-user data)
                   "srcval": the 20% validation split of the 8 training subjects (in-distribution)
Phases
  within_cv   CV inside T with held-out logits: NOEA (446 epochs, notebook 04) and EA (238 epochs,
              Stage A), seeds 0-4
  export      logits of the saved final models, recomputed from their checkpoints (no training):
              within-subject EA on E (aligned offline / first 48 / first 24 trials), cross-subject NOEA / EA
              on the unseen subject's E and on the training subjects' 20% validation split
  artifacts   per-trial artifact features from the raw recording (EEG peak-to-peak, EOG power), expert marks
  final       temperature scaling, calibration metrics, rejection at pre-set operating points, wheelchair
              safety metrics, EA vs no EA -> results/stageB/summary.json + summary.md
Operating points (fixed in advance): 80% coverage; accepted commands >= 85% correct (within-subject) or
>= 65% (cross-subject); 80% coverage with an extra artifact rule (robust z > 3 of EEG peak-to-peak or EOG power).
"""
import argparse
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import alignment as ea  # noqa: E402
import uncertainty as U  # noqa: E402
from data import SUBJECTS  # noqa: E402
from preprocess import load_config_data  # noqa: E402
from run_mentor import Trainer, load_json, save_json  # noqa: E402

OUT = ROOT / "results" / "stageB"
RES_EA = ROOT / "results" / "ea"
CKPT_EA = ROOT / "checkpoints" / "ea"
RUN_OF_TRIAL = np.repeat(np.arange(6), 48)
SPEC = {"windows": False}
WITHIN = {"lr": 1e-3, "batch_size": 32, "dropout": 0.5}
SEEDS_W, SEEDS_X = [0, 1, 2, 3, 4], [0, 1, 2]
REFS = {"offline": None, "online48": 48, "online24": 24}
COVERAGE = 0.80
ACC_TARGET = {"within": 0.85, "cross": 0.65}
ART_Z = 3.0
EPOCHS_OVERRIDE = None
LOG = OUT / "log.txt"

# name: (kind, variant, EA reference, first scored trial, calibration data)
SETTINGS = {
    "within_noEA": ("within", "NOEA", "none", 0, "cv"),
    "within_EA": ("within", "EA", "offline", 0, "cv"),
    "within_noEA_from48": ("within", "NOEA", "none", 48, "cv"),
    "within_EA_online48": ("within", "EA", "online48", 48, "cv"),
    "cross_noEA": ("cross", "NOEA", "none", 0, "others"),
    "cross_EA": ("cross", "EA", "offline", 0, "others"),
    "cross_noEA_from48": ("cross", "NOEA", "none", 48, "others"),
    "cross_EA_online48": ("cross", "EA", "online48", 48, "others"),
    "cross_noEA_srcval": ("cross", "NOEA", "none", 0, "srcval"),
    "cross_EA_srcval": ("cross", "EA", "offline", 0, "srcval"),
}
OPERATING_POINTS = ("cov80", "acc_target", "cov80_artifact")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cv_epochs():
    if EPOCHS_OVERRIDE:
        return {"NOEA": EPOCHS_OVERRIDE, "EA": EPOCHS_OVERRIDE}
    return {"NOEA": load_json(ROOT / "results" / "selection.json")["configs"]["A"]["best_epochs"],
            "EA": load_json(RES_EA / "within_selection.json")["best_epochs"]}


def align_target(X, k):
    """Align a test session with a reference from all its trials (k=None) or its first k trials."""
    return ea.align(X, ea.reference_covariance(X if k is None else X[:k]))


def model_logits(model, X):
    import torch
    from train import DEVICE
    Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=DEVICE)
    model.eval()
    with torch.no_grad():
        return torch.cat([model(Xt[i:i + 1024]) for i in range(0, len(Xt), 1024)]).cpu().numpy().astype(np.float64)


# ============================================================================== calibration data
def phase_within_cv(seeds):
    from train import ChannelScaler
    E = cv_epochs()
    for seed in seeds:
        for variant in ("NOEA", "EA"):
            for s in SUBJECTS:
                X, y = load_config_data("A", s, "T")
                for run in range(6):
                    path = OUT / "within_cv" / variant / f"{s}_run{run + 1}_seed{seed}.json"
                    if path.exists():
                        continue
                    t0 = time.time()
                    te = RUN_OF_TRIAL == run
                    if variant == "EA":       # as in Stage A: each block aligned with its own reference
                        Xtr, Xte = ea.align_self(X[~te])[0], ea.align_self(X[te])[0]
                    else:
                        Xtr, Xte = X[~te], X[te]
                    sc = ChannelScaler().fit(Xtr)
                    tr = Trainer(SPEC, 22, seed * 100 + run, WITHIN["dropout"], WITHIN["lr"], WITHIN["batch_size"])
                    Xt, yt = tr.to_dev(sc.transform(Xtr), y[~te])
                    for _ in range(E[variant]):
                        tr.epoch(Xt, yt)
                    lg = model_logits(tr.model, sc.transform(Xte))
                    acc = float((lg.argmax(1) == y[te]).mean())
                    save_json(path, {"variant": variant, "subject": s, "held_out_run": run + 1, "seed": seed,
                                     "epochs": E[variant], "y": y[te].tolist(), "logits": np.round(lg, 5).tolist(),
                                     "acc": acc, "seconds": round(time.time() - t0, 1)})
                    log(f"within_cv {variant} {s} run{run + 1} seed{seed}: held-out acc {acc:.3f} ({time.time() - t0:.1f} s)")


def load_model(path):
    import torch
    from models import EEGNet
    from train import DEVICE, ChannelScaler
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = EEGNet(n_channels=22, n_times=1000).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    sc = ChannelScaler()
    sc.mean_, sc.std_ = np.asarray(ck["scaler_mean"]), np.asarray(ck["scaler_std"])
    return model, sc


def agreement(logits, probs):
    return float((np.asarray(logits).argmax(1) == np.asarray(probs).argmax(1)).mean())


def phase_export():
    from sklearn.model_selection import train_test_split
    for s in SUBJECTS:
        Xe, ye = load_config_data("A", s, "E")
        for seed in SEEDS_W:
            path = OUT / "export" / "within_EA" / f"{s}_seed{seed}.json"
            if path.exists():
                continue
            model, sc = load_model(CKPT_EA / "within_t2e" / f"{s}_seed{seed}.pt")
            saved = load_json(RES_EA / "within_t2e" / f"{s}_seed{seed}.json")
            lg = {r: model_logits(model, sc.transform(align_target(Xe, k))) for r, k in REFS.items()}
            agree = {r: agreement(lg[r], saved["probs"][r]) for r in lg}
            save_json(path, {"subject": s, "seed": seed, "y": ye.tolist(),
                             "logits": {r: np.round(v, 5).tolist() for r, v in lg.items()}, "agree_with_saved": agree})
            log(f"export within EA {s} seed{seed}: prediction agreement with saved run {agree}")

    data = {(s, sess): load_config_data("A", s, sess) for s in SUBJECTS for sess in "TE"}
    aligned = {k: ea.align_self(v[0])[0] for k, v in data.items()}
    for variant in ("NOEA", "EA"):
        for seed in SEEDS_X:
            for target in SUBJECTS:
                path = OUT / "export" / "loso" / variant / f"{target}_seed{seed}.json"
                if path.exists():
                    continue
                model, sc = load_model(CKPT_EA / "loso" / variant / f"{target}_seed{seed}.pt")
                saved = load_json(RES_EA / "loso" / variant / f"{target}_seed{seed}.json")
                Xe, ye = data[(target, "E")]
                refs = REFS if variant == "EA" else {"none": None}
                lg = {r: model_logits(model, sc.transform(align_target(Xe, k) if variant == "EA" else Xe))
                      for r, k in refs.items()}
                # the 20% validation split of the training subjects, rebuilt exactly as in run_ea.phase_loso
                src = [(s, sess) for s in SUBJECTS if s != target for sess in "TE"]
                X = np.concatenate([(aligned[k] if variant == "EA" else data[k][0]) for k in src])
                y = np.concatenate([data[k][1] for k in src])
                strata = np.concatenate([np.full(288, i * 4) for i in range(len(src))]) + y
                _, vai = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=strata)
                lv = model_logits(model, sc.transform(X[vai]))
                val_acc = float((lv.argmax(1) == y[vai]).mean())
                hist_val_acc = saved["history"]["val_acc"][saved["best_epoch"] - 1]
                agree = {r: agreement(lg[r], saved["probs"][r]) for r in lg}
                save_json(path, {"variant": variant, "subject": target, "seed": seed, "y": ye.tolist(),
                                 "logits": {r: np.round(v, 5).tolist() for r, v in lg.items()},
                                 "src": [f"{a}{b}" for a, b in src], "val_index": vai.tolist(),
                                 "val_y": y[vai].tolist(), "val_logits": np.round(lv, 5).tolist(),
                                 "val_acc": val_acc, "val_acc_in_training_history": hist_val_acc,
                                 "agree_with_saved": agree})
                log(f"export loso {variant} {target} seed{seed}: agreement {agree}; validation acc "
                    f"{val_acc:.4f} (training history at best epoch {hist_val_acc:.4f})")


def phase_artifacts():
    from data import labelled_events, load_raw, trial_rejected_mask
    path = OUT / "artifacts.json"
    if path.exists():
        return
    out = {}
    for s in SUBJECTS:
        for sess in "TE":
            raw = load_raw(s, sess)
            events, _ = labelled_events(raw, s, sess)
            d = raw.get_data() * 1e6                                   # 22 EEG + 3 EOG, microvolts, as recorded
            W = np.stack([d[:, e:e + 1000] for e in events[:, 0]])      # 0-4 s after each cue
            expert = trial_rejected_mask(raw)
            assert len(expert) == 288, (s, sess, len(expert))
            out[f"{s}{sess}"] = {"eeg_ptp": (np.nanmax(W[:, :22], 2) - np.nanmin(W[:, :22], 2)).max(1).round(3).tolist(),
                                 "eog_logvar": np.log(np.nanvar(W[:, 22:], 2).mean(1)).round(4).tolist(),
                                 "expert": expert.astype(int).tolist()}
            log(f"artifacts {s}{sess}: {int(expert.sum())} expert-marked trials")
    save_json(path, out)


# ============================================================================== analysis
@lru_cache(maxsize=None)
def J(path):
    return load_json(path)


@lru_cache(maxsize=None)
def artifact_score(rec):
    """Robust z of EEG peak-to-peak and EOG log-power, statistics from the session's own unlabelled trials."""
    a = J(str(OUT / "artifacts.json"))[rec]
    z = np.maximum(U.robust_z(a["eeg_ptp"]), U.robust_z(a["eog_logvar"]))
    return z, np.array(a["expert"], dtype=bool)


def _cv(variant, s, seed):
    parts = [J(str(OUT / "within_cv" / variant / f"{s}_run{r}_seed{seed}.json")) for r in range(1, 7)]
    return np.concatenate([p["logits"] for p in parts]), np.concatenate([p["y"] for p in parts])


def _within_test(variant, ref, s, seed):
    if variant == "NOEA":
        r = J(str(ROOT / "results" / "runs" / "t2e" / "A" / f"{s}_seed{seed}.json"))
        return np.array(r["logits"]), np.array(r["y_true"])
    r = J(str(OUT / "export" / "within_EA" / f"{s}_seed{seed}.json"))
    return np.array(r["logits"][ref]), np.array(r["y"])


def _cross(variant, s, seed):
    return J(str(OUT / "export" / "loso" / variant / f"{s}_seed{seed}.json"))


def units(setting, members):
    """Per test subject: calibration data and test data (logits of each member model; members = seeds)."""
    kind, variant, ref, start, calib = SETTINGS[setting]
    out = []
    for s in SUBJECTS:
        z_test, expert = artifact_score(f"{s}E")
        u = {"subject": s, "test_art": z_test[start:], "test_expert": expert[start:]}
        if kind == "within":
            cal = [_cv(variant, s, k) for k in members]
            test = [_within_test(variant, ref, s, k) for k in members]
            u.update(cal_logits=[c[0] for c in cal], cal_y=cal[0][1], cal_art=artifact_score(f"{s}T")[0],
                     test_logits=[t[0][start:] for t in test], test_y=test[0][1][start:])
        else:
            test = [_cross(variant, s, k) for k in members]
            u.update(test_logits=[np.array(r["logits"][ref])[start:] for r in test],
                     test_y=np.array(test[0]["y"])[start:])
            if calib == "others":          # the other folds' unseen subjects (their own LOSO models)
                others = [o for o in SUBJECTS if o != s]
                u.update(cal_logits=[np.concatenate([np.array(_cross(variant, o, k)["logits"][ref])[start:] for o in others])
                                     for k in members],
                         cal_y=np.concatenate([np.array(_cross(variant, o, members[0])["y"])[start:] for o in others]),
                         cal_art=np.concatenate([artifact_score(f"{o}E")[0][start:] for o in others]))
            else:                          # srcval: the training subjects' 20% validation split (single model)
                assert len(members) == 1
                r = test[0]
                idx = np.array(r["val_index"])
                u.update(cal_logits=[np.array(r["val_logits"])], cal_y=np.array(r["val_y"]),
                         cal_art=np.array([artifact_score(r["src"][i // 288])[0][i % 288] for i in idx]))
        out.append(u)
    return out


def _combine(member_logits):
    """Single model: its logits. Ensemble: log of the members' mean probability (then calibrated as one model)."""
    if len(member_logits) == 1:
        return np.asarray(member_logits[0], dtype=np.float64)
    return np.log(np.mean([U.softmax(z) for z in member_logits], 0) + 1e-12)


def thresholds(cal_p, cal_y, cal_art, target):
    """Entropy thresholds for the operating points, chosen on calibration data only."""
    H = U.entropy(cal_p)
    correct = cal_p.argmax(1) == cal_y
    clean = cal_art <= ART_Z
    q = min(1.0, COVERAGE / max(clean.mean(), 1e-9))
    return {"cov80": U.threshold_for_coverage(H, COVERAGE),
            "acc_target": U.threshold_for_accuracy(H, correct, target),
            "cov80_artifact": U.threshold_for_coverage(H[clean], q) if clean.any() else -np.inf}


def evaluate(setting, members):
    """Calibrate, score and apply the rejection rule; returns the per-subject units with test results."""
    kind, calib = SETTINGS[setting][0], SETTINGS[setting][4]
    us = units(setting, members)
    for u in us:
        zc, zt = _combine(u["cal_logits"]), _combine(u["test_logits"])
        u["T"] = U.fit_temperature(zc, u["cal_y"])
        u["cal_p"], u["test_p_raw"], u["test_p"] = U.softmax(zc / u["T"]), U.softmax(zt), U.softmax(zt / u["T"])
        if len(members) > 1:
            u["test_mi"] = U.mutual_information([U.softmax(z) for z in u["test_logits"]])
    if calib == "cv":        # one threshold for every user, from the pooled session-T CV data
        th = thresholds(*(np.concatenate([u[k] for u in us]) for k in ("cal_p", "cal_y", "cal_art")), ACC_TARGET[kind])
        for u in us:
            u["th"] = th
    else:                    # per test subject, from its own calibration data
        for u in us:
            u["th"] = thresholds(u["cal_p"], u["cal_y"], u["cal_art"], ACC_TARGET[kind])
    for u in us:
        H = U.entropy(u["test_p"])
        u["test_H"] = H
        th = u["th"]
        u["accept"] = {"cov80": H <= th["cov80"],
                       "acc_target": None if th["acc_target"] is None else H <= th["acc_target"],
                       "cov80_artifact": (H <= th["cov80_artifact"]) & (u["test_art"] <= ART_Z)}
    return us


def metrics(us):
    cat = lambda k: np.concatenate([u[k] for u in us])
    y, p, p_raw, H = cat("test_y"), cat("test_p"), cat("test_p_raw"), cat("test_H")
    correct = p.argmax(1) == y
    m = {"accuracy": float(correct.mean()), "T_mean": float(np.mean([u["T"] for u in us])),
         "ece_raw": U.ece(p_raw, y), "ece_cal": U.ece(p, y), "nll_raw": U.nll(p_raw, y), "nll_cal": U.nll(p, y),
         "brier_raw": U.brier(p_raw, y), "brier_cal": U.brier(p, y),
         "auroc": {"entropy": U.auroc_error(H, correct), "maxprob": U.auroc_error(U.one_minus_maxprob(p), correct)},
         "aurc": U.aurc(H, correct), "no_rejection_false_per_100": float(100 * (1 - correct.mean()))}
    if "test_mi" in us[0]:
        m["auroc"]["mutual_information"] = U.auroc_error(cat("test_mi"), correct)
    for op in OPERATING_POINTS:
        if any(u["accept"][op] is None for u in us):
            m[op] = None
            continue
        m[op] = U.selective_metrics(p, y, np.concatenate([u["accept"][op] for u in us]))
    expert, art = cat("test_expert"), cat("test_art")
    for op in ("cov80", "cov80_artifact"):
        acc = np.concatenate([u["accept"][op] for u in us])
        m[op]["rejected_expert_marked"] = float(1 - acc[expert].mean())
        m[op]["rejected_clean"] = float(1 - acc[~expert].mean())
    flagged = art > ART_Z
    m["artifact_detector"] = {"flagged": float(flagged.mean()), "expert_marked_flagged": float(flagged[expert].mean()),
                              "flagged_that_are_expert_marked": float(expert[flagged].mean()) if flagged.any() else float("nan")}
    m["by_subject"] = {}
    for u in us:
        c = u["test_p"].argmax(1) == u["test_y"]
        row = {"accuracy": float(c.mean()), "T": u["T"], "ece_raw": U.ece(u["test_p_raw"], u["test_y"]),
               "ece_cal": U.ece(u["test_p"], u["test_y"]), "aurc": U.aurc(u["test_H"], c)}
        for op in OPERATING_POINTS:
            if u["accept"][op] is not None:
                sm = U.selective_metrics(u["test_p"], u["test_y"], u["accept"][op])
                row[op] = {k: sm[k] for k in ("coverage", "accepted_accuracy", "false_per_100")}
        m["by_subject"][u["subject"]] = row
    return m


def average(ms):
    """Mean of nested metric dicts (one per seed); a None entry makes that entry None."""
    if any(x is None for x in ms):
        return None
    if isinstance(ms[0], dict):
        return {k: average([x[k] for x in ms]) for k in ms[0]}
    return float(np.nanmean(ms)) if not isinstance(ms[0], str) else ms[0]


def phase_final():
    from scipy.stats import wilcoxon
    res = {}
    for name, (kind, *_rest, calib) in SETTINGS.items():
        seeds = SEEDS_W if kind == "within" else SEEDS_X
        res[name] = {"single": average([metrics(evaluate(name, [k])) for k in seeds])}
        if calib != "srcval":
            res[name]["ensemble"] = metrics(evaluate(name, seeds))
        log(f"final {name}: done")
    tests = {}
    pairs = [("within_EA", "within_noEA"), ("within_EA_online48", "within_noEA_from48"),
             ("cross_EA", "cross_noEA"), ("cross_EA_online48", "cross_noEA_from48")]
    for a, b in pairs:
        for kind in ("ensemble", "single"):
            for key, get in [("aurc", lambda r: r["aurc"]), ("ece_cal", lambda r: r["ece_cal"]),
                             ("false_per_100_at_cov80", lambda r: r["cov80"]["false_per_100"]),
                             ("accepted_accuracy_at_cov80", lambda r: r["cov80"]["accepted_accuracy"])]:
                xa = [get(res[a][kind]["by_subject"][s]) for s in SUBJECTS]
                xb = [get(res[b][kind]["by_subject"][s]) for s in SUBJECTS]
                try:
                    pv = float(wilcoxon(xa, xb).pvalue)
                except ValueError:
                    pv = float("nan")
                tests[f"{a} vs {b} | {kind} | {key}"] = {"mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
                                                         "a_better_for": int(sum((x < z) if key != "accepted_accuracy_at_cov80" else (x > z)
                                                                                 for x, z in zip(xa, xb))),
                                                         "wilcoxon_p": pv}
    save_json(OUT / "summary.json", {"settings": res, "tests": tests,
                                     "operating_points": {"coverage": COVERAGE, "accuracy_target": ACC_TARGET, "artifact_z": ART_Z}})
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}"
    lines = ["# Stage B (calibration + rejection): summary", "",
             "Temperature T and thresholds from calibration data only; test = session E / unseen subject.", "",
             "| Setting | Model | Acc | T | ECE raw → cal | AUROC (entropy) | AURC | 80% cov: coverage / accepted acc / false per 100 "
             "| Target acc: coverage / accepted acc / false per 100 | 80% + artifact rule: coverage / accepted acc / false per 100 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        for kind, m in r.items():
            ops = []
            for op in OPERATING_POINTS:
                o = m[op]
                ops.append("unreachable" if o is None else f"{pct(o['coverage'])} / {pct(o['accepted_accuracy'])} / {o['false_per_100']:.1f}")
            lines.append(f"| {name} | {kind} | {pct(m['accuracy'])} | {m['T_mean']:.2f} | {pct(m['ece_raw'])} → {pct(m['ece_cal'])} "
                         f"| {m['auroc']['entropy']:.3f} | {m['aurc']:.3f} | " + " | ".join(ops) + " |")
    lines += ["", "Paired tests over the 9 subjects (Wilcoxon):"]
    lines += [f"- {k}: {v['mean_a']:.4f} vs {v['mean_b']:.4f}, better for {v['a_better_for']}/9, p = {v['wilcoxon_p']:.3f}"
              for k, v in tests.items()]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log("final: summary written to results/stageB/summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        OUT = ROOT / "results" / "stageB_smoke"
        SUBJECTS = SUBJECTS[:2]
        SEEDS_W, SEEDS_X = [0, 1], [0, 1]
        EPOCHS_OVERRIDE = 3
    LOG = OUT / "logs" / f"{a.phase}.txt"
    log(f"=== {a.phase}{' (smoke)' if a.smoke else ''} ===")
    {"within_cv": lambda: phase_within_cv(a.seeds or SEEDS_W), "export": phase_export,
     "artifacts": phase_artifacts, "final": phase_final}[a.phase]()
    log(f"=== {a.phase} finished ===")
