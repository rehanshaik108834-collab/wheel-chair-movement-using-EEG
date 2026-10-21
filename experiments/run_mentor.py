"""Notebook 05 experiments: adding the mentor's pipeline (Sreeja & Samanta, 2019) to ours.

All decisions are made from the T session only (run-based cross-validation); E is used only to test.
Every unit of work is saved as soon as it finishes and skipped when re-run, so jobs resume.

CPU lane (her sparse-representation classifier)
  sparse_cv      every grid configuration, leave-one-run-out CV inside T (trial-level scores)
  sparse_choose  best configuration + rule by mean CV trial accuracy; probability scale for fusion
  sparse_p0      her own protocol (segments split randomly into 10 folds) to reproduce her behaviour
  sparse_t2e     train on T, test on E (all configurations; the chosen one is the main result)
  sparse_loso    chosen configuration, train on 8 subjects (T+E), test on the unseen subject's E
GPU lane (her techniques added to our EEGNet pipeline)
  eeg_cv         BASE (our pipeline) and variants E1 motor channels, E2 her Chebyshev filter,
                 E3 segmentation windows: CV inside T, held-out accuracy recorded every epoch
  eeg_choose     keep rule (+0.5 pp mean CV accuracy over BASE); runs COMBO CV if >= 2 kept; picks FINAL
  eeg_t2e        train on all 288 T trials for the variant's CV-chosen epochs, test on E, 5 seeds
  eeg_loso       FINAL variant, cross-subject
Analysis
  final          tables, seed ensembles, EEGNet + sparse fusion (weight chosen on T-CV) -> summary
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mentor_pipeline as mp  # noqa: E402
from data import SUBJECTS  # noqa: E402

OUT = ROOT / "results" / "mentor"
CKPT = ROOT / "checkpoints" / "mentor"
LOG = OUT / "log.txt"
RUN_OF_TRIAL = np.repeat(np.arange(6), 48)
KEEP_MARGIN = 0.005          # keep an addition if mean CV accuracy improves by >= 0.5 percentage points
SMOOTH = 25
SMOKE = False


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def acc(y, scores):
    return float((np.asarray(scores).argmax(1) == np.asarray(y)).mean())


def smooth(c, k=SMOOTH):
    c = np.asarray(c, float)
    h = k // 2
    return np.array([c[max(0, i - h): i + h + 1].mean() for i in range(len(c))])


# =============================================================================== sparse lane
def sparse_grid():
    g = []
    for ch in ("all", "motor"):                                   # Case 3: no filtering
        for seg in (0.2, 0.5):
            g.append({"case": "none", "bands": 0, "channels": ch, "seg": seg, "feature": "energy"})
    for nb in (1, 2, 3, 6, 9):                                    # Case 2: band-pass only
        for ch in ("all", "motor"):
            g.append({"case": "bp", "bands": nb, "channels": ch, "seg": 0.2, "feature": "energy"})
    for nb in (1, 2, 3, 6, 9):                                    # Case 1: band-pass + CSP (10 filters)
        g.append({"case": "bp_csp", "bands": nb, "channels": "all", "seg": 0.2, "feature": "energy"})
    g.append({"case": "none", "bands": 0, "channels": "all", "seg": 0.2, "feature": "energy+entropy"})
    g.append({"case": "bp", "bands": 1, "channels": "all", "seg": 0.2, "feature": "energy+entropy"})
    for c in g:
        c["name"] = f"{c['case']}_b{c['bands']}_{c['channels']}_s{int(c['seg'] * 10)}_{c['feature'].replace('+', '')}"
    if SMOKE:
        g = [g[2], g[4], g[14]]          # no filter + motor (her best setting), band-pass, band-pass + CSP
    return g


def bands_of(cfg):
    return [None] if cfg["case"] == "none" else [tuple(b) for b in mp.BAND_SETS[cfg["bands"]]]


def all_bands():
    s = {None}
    for nb in mp.BAND_SETS:
        s |= set(map(tuple, mp.BAND_SETS[nb]))
    return sorted(s, key=lambda b: (-1, -1) if b is None else b)


class SubjectData:
    """All band-filtered epochs of one recording, loaded once (both channel sets derive from 'all')."""

    def __init__(self, subject, session, bands=None):
        self.X, self.y = mp.load_band_epochs(subject, session, bands or all_bands(), channels="all")

    def bands(self, cfg, trials=slice(None)):
        out = {}
        for b in bands_of(cfg):
            X = self.X[b][trials]
            out[b] = X[:, mp.MOTOR_IDX] if cfg["channels"] == "motor" else X
        return out


def sparse_features(cfg, Xb_train, y_train, Xb_test):
    """(train feats (n_tr, n_seg, m), test feats (n_te, n_seg, m)); CSP is fitted on train only."""
    if cfg["case"] == "bp_csp":
        csps = mp.fit_csp(Xb_train, y_train)
        Xb_train, Xb_test = mp.apply_csp(csps, Xb_train), mp.apply_csp(csps, Xb_test)
    return mp.features(Xb_train, cfg["seg"], cfg["feature"]), mp.features(Xb_test, cfg["seg"], cfg["feature"])


def sparse_fit_score(F_tr, y_tr, F_te):
    n_tr, n_seg, m = F_tr.shape
    clf = mp.SparseRepresentationClassifier().fit(F_tr.reshape(-1, m), np.repeat(y_tr, n_seg))
    seg = clf.scores(F_te.reshape(-1, m))
    return mp.trial_scores(seg, F_te.shape[0], n_seg), seg


def phase_sparse_cv():
    grid = sparse_grid()
    for s in SUBJECTS:
        todo = [c for c in grid if not (OUT / "sparse_cv" / c["name"] / f"{s}.json").exists()]
        if not todo:
            continue
        d = SubjectData(s, "T")
        for cfg in todo:
            t0 = time.time()
            scores = {r: np.zeros((288, 4)) for r in mp.CLASS_RULES}
            for run in range(6):
                te = RUN_OF_TRIAL == run
                F_tr, F_te = sparse_features(cfg, d.bands(cfg, ~te), d.y[~te], d.bands(cfg, te))
                tr_scores, _ = sparse_fit_score(F_tr, d.y[~te], F_te)
                for r in mp.CLASS_RULES:
                    scores[r][te] = tr_scores[r]
            res = {"config": cfg, "subject": s, "y": d.y.tolist(),
                   "scores": {r: np.round(v, 5).tolist() for r, v in scores.items()},
                   "acc": {r: acc(d.y, v) for r, v in scores.items()}, "seconds": round(time.time() - t0, 1)}
            save_json(OUT / "sparse_cv" / cfg["name"] / f"{s}.json", res)
            log(f"sparse_cv {cfg['name']} {s}: R4 acc {res['acc']['R4']:.3f} ({res['seconds']} s)")


def fit_prob_scale(scores, y):
    """Scale k so that softmax(k * standardized scores) is calibrated (min NLL on CV trials)."""
    z = (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-9)
    best = (np.inf, 1.0)
    for k in np.exp(np.linspace(np.log(0.01), np.log(20), 200)):
        p = np.exp(k * z - (k * z).max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        best = min(best, (nll, k))
    return float(best[1])


def sparse_probs(scores, k):
    z = (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-9)
    p = np.exp(k * z - (k * z).max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def phase_sparse_choose():
    table = {}
    for cfg in sparse_grid():
        res = [load_json(OUT / "sparse_cv" / cfg["name"] / f"{s}.json") for s in SUBJECTS]
        table[cfg["name"]] = {"config": cfg, **{r: float(np.mean([x["acc"][r] for x in res])) for r in mp.CLASS_RULES},
                              "per_subject_R4": {x["subject"]: x["acc"]["R4"] for x in res}}
    name, rule = max(((n, r) for n in table for r in mp.CLASS_RULES), key=lambda nr: table[nr[0]][nr[1]])
    res = [load_json(OUT / "sparse_cv" / name / f"{s}.json") for s in SUBJECTS]
    S = np.concatenate([np.array(x["scores"][rule]) for x in res])
    y = np.concatenate([np.array(x["y"]) for x in res])
    k = fit_prob_scale(S, y)
    sel = {"chosen": name, "rule": rule, "cv_acc": table[name][rule], "prob_scale": k, "table": table,
           "criterion": "max mean trial-level CV accuracy inside T (9 subjects x 6 runs), over configs x rules"}
    save_json(OUT / "sparse_selection.json", sel)
    log(f"sparse_choose: {name} with rule {rule}, CV accuracy {table[name][rule]:.3f}, prob scale {k:.2f}")


def phase_sparse_p0():
    """Her protocol: segments shuffled and split into 10 folds within T (segments of a trial may be
    in both training and test). Run for her best setting (no filter, motor channels, 0.2 s) and ours."""
    from sklearn.model_selection import KFold
    sel = load_json(OUT / "sparse_selection.json")
    cfgs = [c for c in sparse_grid() if c["name"] in {"none_b0_motor_s2_energy", sel["chosen"]}]
    for cfg in cfgs:
        if cfg["case"] == "bp_csp":
            continue                                  # CSP needs trials; her P0 table uses Case 3 anyway
        for s in SUBJECTS:
            path = OUT / "sparse_p0" / cfg["name"] / f"{s}.json"
            if path.exists():
                continue
            d = SubjectData(s, "T")
            F = mp.features(d.bands(cfg), cfg["seg"], cfg["feature"])
            n_seg, m = F.shape[1], F.shape[2]
            Fs, ys = F.reshape(-1, m), np.repeat(d.y, n_seg)
            pred = np.zeros(len(ys), dtype=int)
            for tr, te in KFold(10, shuffle=True, random_state=0).split(Fs):
                clf = mp.SparseRepresentationClassifier().fit(Fs[tr], ys[tr])
                pred[te] = clf.scores(Fs[te])["R4"].argmax(1)
            res = {"config": cfg, "subject": s, "segment_acc": float((pred == ys).mean()), "n_segments": len(ys)}
            save_json(path, res)
            log(f"sparse_p0 {cfg['name']} {s}: segment-level accuracy with her split {res['segment_acc']:.3f}")


def phase_sparse_t2e():
    grid = sparse_grid()
    for s in SUBJECTS:
        todo = [c for c in grid if not (OUT / "sparse_t2e" / c["name"] / f"{s}.json").exists()]
        if not todo:
            continue
        dT, dE = SubjectData(s, "T"), SubjectData(s, "E")
        for cfg in todo:
            t0 = time.time()
            F_tr, F_te = sparse_features(cfg, dT.bands(cfg), dT.y, dE.bands(cfg))
            sc, _ = sparse_fit_score(F_tr, dT.y, F_te)
            res = {"config": cfg, "subject": s, "y": dE.y.tolist(),
                   "scores": {r: np.round(v, 5).tolist() for r, v in sc.items()},
                   "acc": {r: acc(dE.y, v) for r, v in sc.items()}, "seconds": round(time.time() - t0, 1)}
            save_json(OUT / "sparse_t2e" / cfg["name"] / f"{s}.json", res)
            log(f"sparse_t2e {cfg['name']} {s}: E accuracy R4 {res['acc']['R4']:.3f}")


def phase_sparse_loso(seg_per_trial=4):
    """Dictionary from 8 subjects (T+E). To keep it tractable, `seg_per_trial` random segments of each
    training trial become atoms (all segments of the test subject are classified)."""
    sel = load_json(OUT / "sparse_selection.json")
    cfg = next(c for c in sparse_grid() if c["name"] == sel["chosen"])
    rng = np.random.default_rng(0)
    data = {}
    for s in SUBJECTS:
        if (OUT / "sparse_loso" / cfg["name"] / f"{s}.json").exists():
            continue
        if not data:
            data = {(x, sess): SubjectData(x, sess, bands_of(cfg)) for x in SUBJECTS for sess in "TE"}
        src = [(x, sess) for x in SUBJECTS if x != s for sess in "TE"]
        Xb_tr = {b: np.concatenate([data[k].bands(cfg)[b] for k in src]) for b in bands_of(cfg)}
        y_tr = np.concatenate([data[k].y for k in src])
        F_tr, F_te = sparse_features(cfg, Xb_tr, y_tr, data[(s, "E")].bands(cfg))
        n_tr, n_seg, m = F_tr.shape
        pick = np.stack([rng.choice(n_seg, size=min(seg_per_trial, n_seg), replace=False) for _ in range(n_tr)])
        atoms = F_tr[np.arange(n_tr)[:, None], pick].reshape(-1, m)
        clf = mp.SparseRepresentationClassifier(batch=512).fit(atoms, np.repeat(y_tr, pick.shape[1]))
        seg = clf.scores(F_te.reshape(-1, m))
        sc = mp.trial_scores(seg, F_te.shape[0], n_seg)
        yE = data[(s, "E")].y
        res = {"config": cfg, "subject": s, "y": yE.tolist(), "atoms": int(len(atoms)),
               "scores": {r: np.round(v, 5).tolist() for r, v in sc.items()},
               "acc": {r: acc(yE, v) for r, v in sc.items()}}
        save_json(OUT / "sparse_loso" / cfg["name"] / f"{s}.json", res)
        log(f"sparse_loso {cfg['name']} {s}: E accuracy ({sel['rule']}) {res['acc'][sel['rule']]:.3f}")


# =============================================================================== EEGNet lane
VARIANTS = {
    "BASE": {"data": "A", "channels": "all", "windows": False, "label": "our pipeline (option A)"},
    "E1": {"data": "A", "channels": "motor", "windows": False, "label": "+ motor-area channels (14)"},
    "E2": {"data": "AC", "channels": "all", "windows": False, "label": "+ her Chebyshev II filter"},
    "E3": {"data": "A", "channels": "all", "windows": True, "label": "+ segmentation (2 s windows, 0.5 s step)"},
}
WIN, STRIDE = 500, 125
WITHIN = {"lr": 1e-3, "batch_size": 32, "dropout": 0.5, "epochs": 500, "epochs_windows": 300}
CROSS = {"lr": 1e-3, "batch_size": 64, "dropout": 0.25, "max_epochs": 300, "patience": 50}
PROB_EVERY = 10


def variant_spec(name):
    if name in VARIANTS:
        return VARIANTS[name]
    sel = load_json(OUT / "eeg_selection.json")
    return sel["combo_spec"]


def load_variant(spec, subject, session):
    from preprocess import load_config_data
    X, y = load_config_data(spec["data"], subject, session)
    if spec["channels"] == "motor":
        X = X[:, mp.MOTOR_IDX]
    return X, y


def windows(X):
    """(n, ch, 1000) -> (n, n_win, ch, WIN)."""
    starts = range(0, X.shape[-1] - WIN + 1, STRIDE)
    return np.stack([X[..., s:s + WIN] for s in starts], axis=1)


def n_epochs(spec):
    return WITHIN["epochs_windows"] if spec["windows"] else WITHIN["epochs"]


class Trainer:
    """EEGNet training with optional windowing; trial predictions average the windows' softmax."""

    def __init__(self, spec, n_ch, seed, dropout, lr, batch_size):
        import torch
        from models import EEGNet
        from train import DEVICE, set_seed
        set_seed(seed, fast=True)
        self.torch, self.dev, self.spec = torch, DEVICE, spec
        n_times = WIN if spec["windows"] else 1000
        self.model = EEGNet(n_channels=n_ch, n_times=n_times, dropout=dropout).to(DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.CrossEntropyLoss()
        self.rng = np.random.default_rng(seed)
        self.bs = batch_size

    def _expand(self, X, y):
        if not self.spec["windows"]:
            return X, y
        W = windows(X)
        return W.reshape(-1, *W.shape[2:]), np.repeat(y, W.shape[1])

    def to_dev(self, X, y=None):
        t = self.torch.as_tensor(X, device=self.dev)
        return t if y is None else (t, self.torch.as_tensor(y, device=self.dev))

    def epoch(self, Xt, yt):
        self.model.train()
        idx = self.torch.as_tensor(self.rng.permutation(len(Xt)), device=self.dev)
        for s in range(0, len(Xt), self.bs):
            b = idx[s:s + self.bs]
            self.opt.zero_grad()
            self.loss(self.model(Xt[b]), yt[b]).backward()
            self.opt.step()
            self.model.apply_constraints()

    def trial_probs(self, Xt_trials):
        """Xt_trials: device tensor (n, ch, T) or windows (n, n_win, ch, WIN) -> (n, 4) numpy probs."""
        with self.torch.no_grad():
            self.model.eval()
            if self.spec["windows"]:
                n, w = Xt_trials.shape[:2]
                out = []
                flat = Xt_trials.reshape(n * w, *Xt_trials.shape[2:])
                for s in range(0, len(flat), 1024):
                    out.append(self.torch.softmax(self.model(flat[s:s + 1024]), 1))
                return self.torch.cat(out).reshape(n, w, -1).mean(1).cpu().numpy()
            out = [self.torch.softmax(self.model(Xt_trials[s:s + 1024]), 1) for s in range(0, len(Xt_trials), 1024)]
            return self.torch.cat(out).cpu().numpy()

    def prepare_eval(self, X):
        return self.to_dev(windows(X) if self.spec["windows"] else X)


def save_ckpt(path, trainer, scaler, extra):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": trainer.model.state_dict(), "scaler_mean": scaler.mean_,
                "scaler_std": scaler.std_, **extra}, path)


def phase_eeg_cv(variants, seed=0):
    from train import ChannelScaler
    for v in variants:
        spec = variant_spec(v)
        E = n_epochs(spec)
        for s in SUBJECTS:
            X, y = load_variant(spec, s, "T")
            for run in range(6):
                path = OUT / "eeg_cv" / v / f"{s}_run{run + 1}.json"
                if path.exists():
                    continue
                t0 = time.time()
                te = RUN_OF_TRIAL == run
                sc = ChannelScaler().fit(X[~te])
                tr = Trainer(spec, X.shape[1], seed * 100 + run, WITHIN["dropout"], WITHIN["lr"], WITHIN["batch_size"])
                Xt, yt = tr.to_dev(*tr._expand(sc.transform(X[~te]), y[~te]))
                Xm = tr.prepare_eval(sc.transform(X[te]))
                mon_acc, mon_nll, probs = [], [], {}
                for ep in range(1, E + 1):
                    tr.epoch(Xt, yt)
                    p = tr.trial_probs(Xm)
                    mon_acc.append(acc(y[te], p))
                    mon_nll.append(float(-np.log(p[np.arange(len(p)), y[te]] + 1e-12).mean()))
                    if ep % PROB_EVERY == 0:
                        probs[ep] = np.round(p, 4).tolist()
                res = {"variant": v, "spec": spec, "subject": s, "held_out_run": run + 1, "epochs": E,
                       "y": y[te].tolist(), "mon_acc": mon_acc, "mon_nll": mon_nll, "probs": probs,
                       "seconds": round(time.time() - t0, 1)}
                save_ckpt(CKPT / "eeg_cv" / v / f"{s}_run{run + 1}.pt", tr, sc, {"spec": spec, "epochs": E})
                save_json(path, res)
                log(f"eeg_cv {v} {s} run{run + 1}: final held-out acc {mon_acc[-1]:.3f} ({res['seconds']} s)")


def cv_summary(v):
    runs = [load_json(OUT / "eeg_cv" / v / f"{s}_run{r}.json") for s in SUBJECTS for r in range(1, 7)]
    curve = np.mean([x["mon_acc"] for x in runs], axis=0)
    sm = smooth(curve)
    best = int(np.argmax(sm)) + 1
    return {"best_epochs": best, "cv_acc": float(sm[best - 1]), "cv_acc_final": float(curve[-1])}


def phase_eeg_choose():
    summ = {v: cv_summary(v) for v in ["BASE", "E1", "E2", "E3"]}
    base = summ["BASE"]["cv_acc"]
    kept = [v for v in ["E1", "E2", "E3"] if summ[v]["cv_acc"] >= base + KEEP_MARGIN]
    sel = {"summary": summ, "kept": kept, "keep_margin": KEEP_MARGIN, "combo_spec": None}
    if len(kept) >= 2:
        spec = {"data": "AC" if "E2" in kept else "A", "channels": "motor" if "E1" in kept else "all",
                "windows": "E3" in kept, "label": "combination of kept additions: " + " + ".join(kept)}
        sel["combo_spec"] = spec
        save_json(OUT / "eeg_selection.json", sel)
        phase_eeg_cv(["COMBO"])
        summ["COMBO"] = cv_summary("COMBO")
    candidates = ["BASE"] + kept + (["COMBO"] if len(kept) >= 2 else [])
    final = max(candidates, key=lambda v: summ[v]["cv_acc"])
    sel.update({"summary": summ, "final": final, "candidates": candidates})
    save_json(OUT / "eeg_selection.json", sel)
    log(f"eeg_choose: BASE {base:.3f}; " + ", ".join(f"{v} {summ[v]['cv_acc']:.3f}" for v in summ if v != "BASE")
        + f"; kept {kept}; FINAL = {final}")


def phase_eeg_t2e(variants, seeds):
    from train import ChannelScaler
    sel = load_json(OUT / "eeg_selection.json")
    for v in variants:
        if v == "COMBO" and not sel.get("combo_spec"):
            continue
        spec = variant_spec(v)
        epochs = sel["summary"][v]["best_epochs"]
        for seed in seeds:
            for s in SUBJECTS:
                path = OUT / "eeg_t2e" / v / f"{s}_seed{seed}.json"
                if path.exists():
                    continue
                t0 = time.time()
                X, y = load_variant(spec, s, "T")
                Xe, ye = load_variant(spec, s, "E")
                sc = ChannelScaler().fit(X)
                tr = Trainer(spec, X.shape[1], seed, WITHIN["dropout"], WITHIN["lr"], WITHIN["batch_size"])
                Xt, yt = tr.to_dev(*tr._expand(sc.transform(X), y))
                for _ in range(epochs):
                    tr.epoch(Xt, yt)
                p = tr.trial_probs(tr.prepare_eval(sc.transform(Xe)))
                res = {"variant": v, "spec": spec, "subject": s, "seed": seed, "epochs": epochs, "y": ye.tolist(),
                       "probs": np.round(p, 5).tolist(), "acc": acc(ye, p), "seconds": round(time.time() - t0, 1)}
                save_ckpt(CKPT / "eeg_t2e" / v / f"{s}_seed{seed}.pt", tr, sc, {"spec": spec, "epochs": epochs})
                save_json(path, res)
                log(f"eeg_t2e {v} {s} seed{seed}: E accuracy {res['acc']:.3f} ({res['seconds']} s)")


def phase_eeg_loso():
    from sklearn.model_selection import train_test_split
    from train import ChannelScaler
    sel = load_json(OUT / "eeg_selection.json")
    v = sel["final"]
    spec = variant_spec(v)
    seeds = [0] if spec["windows"] else [0, 1, 2]
    if v == "BASE":
        log("eeg_loso: FINAL is BASE -> cross-subject result already exists (notebook 04, 44.66%); nothing to run")
        return
    data = {(s, sess): load_variant(spec, s, sess) for s in SUBJECTS for sess in "TE"}
    for seed in seeds:
        for target in SUBJECTS:
            path = OUT / "eeg_loso" / v / f"{target}_seed{seed}.json"
            if path.exists():
                continue
            t0 = time.time()
            src = [(s, sess) for s in SUBJECTS if s != target for sess in "TE"]
            X = np.concatenate([data[k][0] for k in src])
            y = np.concatenate([data[k][1] for k in src])
            strata = np.concatenate([np.full(288, i * 4) for i in range(len(src))]) + y
            tri, vai = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=strata)
            sc = ChannelScaler().fit(X[tri])
            tr = Trainer(spec, X.shape[1], seed, CROSS["dropout"], CROSS["lr"], CROSS["batch_size"])
            Xt, yt = tr.to_dev(*tr._expand(sc.transform(X[tri]), y[tri]))
            Xv = tr.prepare_eval(sc.transform(X[vai]))
            best, best_state, best_ep, wait = np.inf, None, 0, 0
            for ep in range(1, CROSS["max_epochs"] + 1):
                tr.epoch(Xt, yt)
                pv = tr.trial_probs(Xv)
                nll = float(-np.log(pv[np.arange(len(vai)), y[vai]] + 1e-12).mean())
                if nll < best:
                    best, best_ep, wait = nll, ep, 0
                    best_state = {k: t.detach().clone() for k, t in tr.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= CROSS["patience"]:
                        break
            tr.model.load_state_dict(best_state)
            Xe, ye = data[(target, "E")]
            p = tr.trial_probs(tr.prepare_eval(sc.transform(Xe)))
            res = {"variant": v, "spec": spec, "subject": target, "seed": seed, "best_epoch": best_ep,
                   "y": ye.tolist(), "probs": np.round(p, 5).tolist(), "acc": acc(ye, p),
                   "seconds": round(time.time() - t0, 1)}
            save_ckpt(CKPT / "eeg_loso" / v / f"{target}_seed{seed}.pt", tr, sc, {"spec": spec, "best_epoch": best_ep})
            save_json(path, res)
            log(f"eeg_loso {v} {target} seed{seed}: E accuracy {res['acc']:.3f}, best epoch {best_ep} ({res['seconds']} s)")


# =============================================================================== analysis
def phase_final():
    """Assemble every result into results/mentor/summary.json + summary.md (no training)."""
    eeg_sel = load_json(OUT / "eeg_selection.json")
    sp_sel = load_json(OUT / "sparse_selection.json")
    final, rule, k = eeg_sel["final"], sp_sel["rule"], sp_sel["prob_scale"]
    summary = {"eeg_selection": eeg_sel, "sparse_selection": {kk: sp_sel[kk] for kk in ("chosen", "rule", "cv_acc", "prob_scale")}}

    # ---- T->E per variant (BASE comes from notebook 04)
    def base_runs():
        out = {}
        for f in sorted((ROOT / "results" / "runs" / "t2e" / "A").glob("*.json")):
            r = load_json(f)
            lg = np.array(r["logits"])
            p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            out.setdefault(r["subject"], []).append((np.array(r["y_true"]), p))
        return out

    def variant_runs(v):
        out = {}
        for f in sorted((OUT / "eeg_t2e" / v).glob("*.json")):
            r = load_json(f)
            out.setdefault(r["subject"], []).append((np.array(r["y"]), np.array(r["probs"])))
        return out

    t2e = {"BASE": base_runs()}
    for v in ["E1", "E2", "E3", "COMBO"]:
        if (OUT / "eeg_t2e" / v).exists():
            t2e[v] = variant_runs(v)
    rows = {}
    for v, runs in t2e.items():
        rows[v] = {"single": {s: float(np.mean([acc(y, p) for y, p in runs[s]])) for s in SUBJECTS},
                   "ensemble": {s: acc(runs[s][0][0], np.mean([p for _, p in runs[s]], 0)) for s in SUBJECTS}}
        for kind in ("single", "ensemble"):
            rows[v][kind]["mean"] = float(np.mean([rows[v][kind][s] for s in SUBJECTS]))

    # ---- her classifier on E
    sp = {s: load_json(OUT / "sparse_t2e" / sp_sel["chosen"] / f"{s}.json") for s in SUBJECTS}
    rows["SPARSE"] = {"single": {s: sp[s]["acc"][rule] for s in SUBJECTS}}
    rows["SPARSE"]["single"]["mean"] = float(np.mean([rows["SPARSE"]["single"][s] for s in SUBJECTS]))

    # ---- fusion weight from T-CV (EEGNet FINAL CV probs at its chosen epoch + sparse CV probs)
    ep = eeg_sel["summary"][final]["best_epochs"]
    snap = str(min(range(PROB_EVERY, n_epochs(variant_spec(final)) + 1, PROB_EVERY), key=lambda e: abs(e - ep)))
    cv_eeg, cv_sp, cv_y = [], [], []
    for s in SUBJECTS:
        spcv = load_json(OUT / "sparse_cv" / sp_sel["chosen"] / f"{s}.json")
        S = sparse_probs(np.array(spcv["scores"][rule]), k)
        for r in range(1, 7):
            x = load_json(OUT / "eeg_cv" / final / f"{s}_run{r}.json")
            cv_eeg.append(np.array(x["probs"][snap])); cv_y.append(np.array(x["y"]))
            cv_sp.append(S[RUN_OF_TRIAL == r - 1])
    cv_eeg, cv_sp, cv_y = map(np.concatenate, (cv_eeg, cv_sp, cv_y))
    weights = np.round(np.arange(0, 2.01, 0.1), 2)
    cv_fused = {float(w): acc(cv_y, np.log(cv_eeg + 1e-9) + w * np.log(cv_sp + 1e-9)) for w in weights}
    w_best = max(cv_fused, key=cv_fused.get)
    use_fusion = cv_fused[w_best] >= cv_fused[0.0] + KEEP_MARGIN
    summary["fusion"] = {"cv_acc_by_weight": cv_fused, "weight": w_best, "kept": bool(use_fusion),
                         "cv_snapshot_epoch": int(snap)}

    fin_runs = t2e[final]
    fused = {"single": {}, "ensemble": {}}
    for s in SUBJECTS:
        Ssp = sparse_probs(np.array(sp[s]["scores"][rule]), k)
        w = w_best if use_fusion else 0.0
        fused["single"][s] = float(np.mean([acc(y, np.log(p + 1e-9) + w * np.log(Ssp + 1e-9)) for y, p in fin_runs[s]]))
        pe = np.mean([p for _, p in fin_runs[s]], 0)
        fused["ensemble"][s] = acc(fin_runs[s][0][0], np.log(pe + 1e-9) + w * np.log(Ssp + 1e-9))
    for kind in fused:
        fused[kind]["mean"] = float(np.mean([fused[kind][s] for s in SUBJECTS]))
    rows["FINAL_FUSED"] = fused
    summary["t2e"] = rows

    # ---- cross-subject
    loso = {}
    if (OUT / "eeg_loso" / final).exists():
        res = [load_json(f) for f in (OUT / "eeg_loso" / final).glob("*.json")]
        loso["eeg_final"] = {s: float(np.mean([r["acc"] for r in res if r["subject"] == s])) for s in SUBJECTS}
        loso["eeg_final"]["mean"] = float(np.mean([loso["eeg_final"][s] for s in SUBJECTS]))
    if (OUT / "sparse_loso" / sp_sel["chosen"]).exists():
        res = {s: load_json(OUT / "sparse_loso" / sp_sel["chosen"] / f"{s}.json") for s in SUBJECTS}
        loso["sparse"] = {s: res[s]["acc"][rule] for s in SUBJECTS}
        loso["sparse"]["mean"] = float(np.mean([loso["sparse"][s] for s in SUBJECTS]))
    summary["loso"] = loso
    p0 = {}
    for d in (OUT / "sparse_p0").glob("*"):
        p0[d.name] = float(np.mean([load_json(f)["segment_acc"] for f in d.glob("*.json")]))
    summary["p0_her_protocol_segment_acc"] = p0
    save_json(OUT / "summary.json", summary)

    lines = ["# Mentor pipeline: overnight summary", "",
             f"FINAL EEGNet variant: **{final}** ({variant_spec(final)['label']}); kept additions: {eeg_sel['kept']}",
             f"Her classifier: **{sp_sel['chosen']}**, rule {rule}, CV accuracy {sp_sel['cv_acc']:.3f}",
             f"Fusion weight {w_best} (kept: {use_fusion})", "",
             "| Pipeline | " + " | ".join(SUBJECTS) + " | Mean |", "|" + "---|" * (len(SUBJECTS) + 2)]
    for v, r in rows.items():
        for kind in ("single", "ensemble"):
            if kind in r:
                lines.append(f"| {v} ({kind}) | " + " | ".join(f"{100 * r[kind][s]:.1f}" for s in SUBJECTS)
                             + f" | **{100 * r[kind]['mean']:.2f}** |")
    lines += ["", f"Cross-subject: {json.dumps({kk: round(vv['mean'], 4) for kk, vv in loso.items()})}",
              f"Her protocol (segment-level, random split): {json.dumps({kk: round(vv, 4) for kk, vv in p0.items()})}"]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"final: summary written to {OUT / 'summary.md'}")


# =============================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--variants", nargs="+", default=["BASE", "E1", "E2", "E3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        SMOKE = True
        OUT = ROOT / "results" / "mentor_smoke"
        CKPT = ROOT / "checkpoints" / "mentor_smoke"
        LOG = OUT / "log.txt"
        SUBJECTS = SUBJECTS[:2]
        WITHIN.update({"epochs": 12, "epochs_windows": 12})
        CROSS.update({"max_epochs": 3, "patience": 2})
        SMOOTH = 3
    LOG = OUT / "logs" / f"{a.phase}.txt"          # one log per phase: parallel lanes never share a file
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log(f"=== {a.phase} {'(smoke)' if a.smoke else ''} ===")
    fn = {"sparse_cv": phase_sparse_cv, "sparse_choose": phase_sparse_choose, "sparse_p0": phase_sparse_p0,
          "sparse_t2e": phase_sparse_t2e, "sparse_loso": phase_sparse_loso,
          "eeg_cv": lambda: phase_eeg_cv(a.variants), "eeg_choose": phase_eeg_choose,
          "eeg_t2e": lambda: phase_eeg_t2e(a.variants, a.seeds), "eeg_loso": phase_eeg_loso,
          "final": phase_final}[a.phase]
    fn()
    log(f"=== {a.phase} finished ===")
