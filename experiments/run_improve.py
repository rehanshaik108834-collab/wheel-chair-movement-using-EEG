"""Part 1: improving the within-subject EEGNet (same network as Nzakuna et al. 2025), EA throughout.

Every choice is made on session T only (6-fold leave-one-run-out CV); session E is used once, in t2e.
Variants (combine with '+', e.g. FTDB+AUGSR):
  BASE    our EA EEGNet (option A: 8-30 Hz, 0-4 s; dropout 0.5, Adam 1e-3, batch 32)
  FTDB    FT-DropBlock instead of dropout (p 0.5, block 15, as in Nzakuna et al.)
  PREPD   their preprocessing (no band-pass, -0.5..4 s, exponential moving standardisation) + EA
  AUGSR   segmentation & recombination: half of each epoch's trials are rebuilt from 8 time segments
          of random same-class training trials (Lotte 2015)
  AUGNS   amplitude scaling (x U(0.9, 1.1)), Gaussian noise (0.05 sd) and time shift (+-20 samples)
  RECIPE  label smoothing 0.1 + cosine learning-rate decay to 0 over the training length
  CLEAN   expert-marked artifact trials removed from the TRAINING trials only (test trials untouched)
Phases
  cv      --variants V... --seeds S...   held-out-run accuracy every 10 epochs up to 400
  summary                                 per variant: best epoch (smoothed mean curve), CV accuracy,
                                          paired difference to BASE, subjects improved, Wilcoxon p
  t2e     --variants V --epochs N --seeds S...  train on all T, test on E (offline / first-48 EA)
  t2e_summary                             accuracy of 1 model and of the seed ensemble
"""
import argparse
import json
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
from data import SUBJECTS  # noqa: E402
from preprocess import load_config_data  # noqa: E402
from run_mentor import acc, load_json, save_json  # noqa: E402

SMOKE = "--smoke" in sys.argv
OUT = ROOT / "results" / ("improve_smoke" if SMOKE else "improve")
RUN_OF_TRIAL = np.repeat(np.arange(6), 48)
MAX_EPOCHS = 30 if SMOKE else 400
EVAL_EVERY = 10
SUBJ = SUBJECTS[:2] if SMOKE else SUBJECTS
REFS = {"offline": None, "online48": 48}
KEEP_RULE = 0.005                                   # +0.5 points on T-CV to keep a change
PAPER = {"LEMC-FTDB (test labels used)": 0.8040, "EMCD (15 nets, honest)": 0.7535, "MCD (1 net)": 0.7114}
OPTIONS = {"BASE": {}, "FTDB": {"drop": "ftdb"}, "PREPD": {"config": "D"}, "AUGSR": {"sr": True},
           "AUGNS": {"ns": True}, "RECIPE": {"smooth": 0.1, "cosine": True}, "CLEAN": {"clean": True}}


def spec(variant):
    s = {"config": "A", "drop": "dropout", "sr": False, "ns": False, "smooth": 0.0, "cosine": False, "clean": False}
    for part in variant.split("+"):
        s.update(OPTIONS[part])
    return s


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


_EXPERT = None


def expert_marks(rec):
    global _EXPERT
    if _EXPERT is None:
        _EXPERT = json.loads((ROOT / "results" / "stageB" / "artifacts.json").read_text())
    return np.array(_EXPERT[rec]["expert"], dtype=bool)


class Net:
    """EEGNet training with the variant's options; data live on the GPU."""

    def __init__(self, sp, n_times, seed, epochs, lr=1e-3, batch_size=32, dropout=0.5):
        import torch
        from models import EEGNet
        from train import DEVICE, set_seed
        set_seed(seed, fast=True)
        self.t, self.dev, self.sp, self.bs = torch, DEVICE, sp, batch_size
        self.model = EEGNet(n_channels=22, n_times=n_times, dropout=dropout, drop_type=sp["drop"]).to(DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs) if sp["cosine"] else None
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=sp["smooth"])
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def data(self, X, y=None):
        t = self.t.as_tensor(X, device=self.dev)
        return t if y is None else (t, self.t.as_tensor(y, device=self.dev))

    def _sr(self, X, y):
        """Half of the trials replaced by segment-recombined trials of the same class (8 segments)."""
        n, _, T = X.shape
        out = X.clone()
        pick = self.t.as_tensor(np.flatnonzero(self.rng.random(n) < 0.5), device=self.dev)
        if len(pick) == 0:
            return out
        yc = y.cpu().numpy()
        pools = {c: np.flatnonzero(yc == c) for c in np.unique(yc)}
        edges = np.linspace(0, T, 9).astype(int)
        donors = np.stack([self.rng.choice(pools[yc[i]], 8) for i in pick.cpu().numpy()])   # (m, 8)
        donors = self.t.as_tensor(donors, device=self.dev)
        for k in range(8):
            out[pick, :, edges[k]:edges[k + 1]] = X[donors[:, k], :, edges[k]:edges[k + 1]]
        return out

    def _ns(self, xb):
        b = xb.shape[0]
        scale = 0.9 + 0.2 * self.t.rand(b, 1, 1, device=self.dev, generator=self.gen)
        xb = xb * scale + 0.05 * self.t.randn(xb.shape, device=self.dev, generator=self.gen)
        shift = int(self.rng.integers(-20, 21))
        return self.t.roll(xb, shift, dims=2)

    def epoch(self, X, y):
        self.model.train()
        Xe = self._sr(X, y) if self.sp["sr"] else X
        idx = self.t.as_tensor(self.rng.permutation(len(X)), device=self.dev)
        for s in range(0, len(X), self.bs):
            b = idx[s:s + self.bs]
            xb = self._ns(Xe[b]) if self.sp["ns"] else Xe[b]
            self.opt.zero_grad()
            self.loss(self.model(xb), y[b]).backward()
            self.opt.step()
            self.model.apply_constraints()
        if self.sched:
            self.sched.step()

    def probs(self, X):
        with self.t.no_grad():
            self.model.eval()
            return self.t.cat([self.t.softmax(self.model(X[s:s + 512]), 1) for s in range(0, len(X), 512)]).cpu().numpy()


def phase_cv(variants, seeds):
    from train import ChannelScaler
    for v in variants:
        sp = spec(v)
        for seed in seeds:
            for s in SUBJ:
                X, y = load_config_data(sp["config"], s, "T")
                marked = expert_marks(f"{s}T")
                for run in range(6):
                    path = OUT / "cv" / v / f"{s}_run{run + 1}_seed{seed}.json"
                    if path.exists():
                        continue
                    t0 = time.time()
                    te = RUN_OF_TRIAL == run
                    trm = ~te & ~marked if sp["clean"] else ~te
                    Xtr, _ = ea.align_self(X[trm])
                    Xte, _ = ea.align_self(X[te])             # held-out run: its own reference, all its trials
                    sc = ChannelScaler().fit(Xtr)
                    net = Net(sp, X.shape[2], seed * 100 + run, MAX_EPOCHS)
                    Xt, yt = net.data(sc.transform(Xtr), y[trm])
                    Xm = net.data(sc.transform(Xte))
                    mon = []
                    for ep in range(1, MAX_EPOCHS + 1):
                        net.epoch(Xt, yt)
                        if ep % EVAL_EVERY == 0:
                            mon.append(acc(y[te], net.probs(Xm)))
                    save_json(path, {"variant": v, "subject": s, "run": run + 1, "seed": seed, "mon_acc": mon,
                                     "n_train": int(trm.sum()), "seconds": round(time.time() - t0, 1)})
                    log(f"cv {v} {s} run{run + 1} seed{seed}: acc@{MAX_EPOCHS} {mon[-1]:.3f}, "
                        f"best {max(mon):.3f} ({time.time() - t0:.1f} s)")


def cv_table(v):
    """{(subject, run, seed): accuracy curve} for one variant."""
    return {(r["subject"], r["run"], r["seed"]): np.array(r["mon_acc"])
            for r in (load_json(f) for f in sorted((OUT / "cv" / v).glob("*.json")))}


def smooth3(c):
    return np.convolve(np.pad(c, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def phase_summary():
    from scipy.stats import wilcoxon
    variants = sorted(p.name for p in (OUT / "cv").iterdir() if p.is_dir())
    base = cv_table("BASE")
    rows = {}
    for v in variants:
        tab = cv_table(v)
        keys = sorted(set(tab) & set(base))
        if not keys:
            continue
        curve = smooth3(np.mean([tab[k] for k in keys], 0))
        bcurve = smooth3(np.mean([base[k] for k in keys], 0))
        i, bi = int(np.argmax(curve)), int(np.argmax(bcurve))
        subj = sorted({k[0] for k in keys})
        sv = np.array([np.mean([smooth3(tab[k])[i] for k in keys if k[0] == s]) for s in subj])
        sb = np.array([np.mean([smooth3(base[k])[bi] for k in keys if k[0] == s]) for s in subj])
        d = sv - sb
        p = float(wilcoxon(sv, sb).pvalue) if v != "BASE" and np.any(d != 0) and len(subj) > 5 else float("nan")
        rows[v] = {"best_epoch": (i + 1) * EVAL_EVERY, "cv_acc": float(curve[i]), "base_cv_acc": float(bcurve[bi]),
                   "delta": float(curve[i] - bcurve[bi]), "subjects_better": int((d > 0).sum()), "n_subjects": len(subj),
                   "wilcoxon_p": p, "n_runs": len(keys), "per_subject": dict(zip(subj, sv.round(4).tolist())),
                   "keep": bool(v != "BASE" and curve[i] - bcurve[bi] >= KEEP_RULE),
                   "curve": curve.round(4).tolist()}
    save_json(OUT / "cv_summary.json", rows)
    lines = ["# Part 1 screening: session-T CV (EA EEGNet), accuracy at the best epoch of the smoothed mean curve", "",
             f"Keep rule: at least +{100 * KEEP_RULE:.1f} points over BASE on the same folds and seeds.", "",
             "| Variant | Best epoch | T-CV acc | vs BASE | Subjects better | Wilcoxon p | Runs | Keep |",
             "|---|---|---|---|---|---|---|---|"]
    for v, r in sorted(rows.items(), key=lambda kv: -kv[1]["cv_acc"]):
        lines.append(f"| {v} | {r['best_epoch']} | {100 * r['cv_acc']:.2f} | {100 * r['delta']:+.2f} | "
                     f"{r['subjects_better']}/{r['n_subjects']} | {r['wilcoxon_p']:.3f} | {r['n_runs']} | "
                     f"{'yes' if r['keep'] else ''} |")
    (OUT / "cv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("summary written to " + str(OUT / "cv_summary.md"))
    print("\n".join(lines))


def phase_t2e(variant, epochs, seeds):
    from train import ChannelScaler
    sp = spec(variant)
    for seed in seeds:
        for s in SUBJ:
            path = OUT / "t2e" / variant / f"{s}_seed{seed}.json"
            if path.exists():
                continue
            t0 = time.time()
            X, y = load_config_data(sp["config"], s, "T")
            Xe, ye = load_config_data(sp["config"], s, "E")
            trm = ~expert_marks(f"{s}T") if sp["clean"] else np.ones(len(y), bool)
            Xtr, _ = ea.align_self(X[trm])
            sc = ChannelScaler().fit(Xtr)
            net = Net(sp, X.shape[2], seed, epochs)
            Xt, yt = net.data(sc.transform(Xtr), y[trm])
            for _ in range(epochs):
                net.epoch(Xt, yt)
            probs = {}
            for r, k in REFS.items():
                Xa = ea.align(Xe, ea.reference_covariance(Xe if k is None else Xe[:k]))
                probs[r] = net.probs(net.data(sc.transform(Xa)))
            res = {"variant": variant, "subject": s, "seed": seed, "epochs": epochs, "y": ye.tolist(),
                   "probs": {r: np.round(p, 5).tolist() for r, p in probs.items()},
                   "acc": {r: acc(ye, p) for r, p in probs.items()}, "seconds": round(time.time() - t0, 1)}
            path.parent.mkdir(parents=True, exist_ok=True)
            import torch
            (ROOT / "checkpoints" / "improve" / variant).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.model.state_dict(), "scaler_mean": sc.mean_, "scaler_std": sc.std_,
                        "spec": sp, "epochs": epochs}, ROOT / "checkpoints" / "improve" / variant / f"{s}_seed{seed}.pt")
            save_json(path, res)
            log(f"t2e {variant} {s} seed{seed}: E acc offline {res['acc']['offline']:.3f}, "
                f"online48 {res['acc']['online48']:.3f} ({res['seconds']} s)")


def phase_t2e_summary():
    from scipy.stats import wilcoxon
    out = {}
    base = {}
    for f in (ROOT / "results" / "ea" / "within_t2e").glob("*.json"):
        r = load_json(f)
        base.setdefault(r["subject"], []).append(np.array(r["probs"]["offline"]))
    yb = {s: np.array(load_json(ROOT / "results" / "ea" / "within_t2e" / f"{s}_seed0.json")["y"]) for s in SUBJECTS}
    base_ens = {s: acc(yb[s], np.mean(base[s], 0)) for s in SUBJECTS}
    for vdir in sorted(p for p in (OUT / "t2e").iterdir() if p.is_dir()):
        runs = [load_json(f) for f in vdir.glob("*.json")]
        by = {}
        for r in runs:
            by.setdefault(r["subject"], []).append(r)
        subj = sorted(by)
        res = {}
        for ref in REFS:
            single = {s: float(np.mean([r["acc"][ref] for r in by[s]])) for s in subj}
            ens = {s: acc(np.array(by[s][0]["y"]), np.mean([np.array(r["probs"][ref]) for r in by[s]], 0)) for s in subj}
            res[ref] = {"single": float(np.mean(list(single.values()))), "ensemble": float(np.mean(list(ens.values()))),
                        "per_subject_ensemble": ens, "n_seeds": min(len(by[s]) for s in subj)}
        e = res["offline"]["per_subject_ensemble"]
        if len(subj) == 9:
            a, b = np.array([e[s] for s in subj]), np.array([base_ens[s] for s in subj])
            res["vs_old_ensemble"] = {"old": float(b.mean()), "new": float(a.mean()), "better_for": int((a > b).sum()),
                                      "wilcoxon_p": float(wilcoxon(a, b).pvalue) if np.any(a != b) else float("nan")}
        out[vdir.name] = res
    save_json(OUT / "t2e_summary.json", out)
    lines = ["# Part 1: session E (train T -> test E), mean of 9 subjects", "",
             "| Variant | Seeds | 1 model | Ensemble | Ensemble, EA from first 48 trials | vs old EA ensemble (76.16) |",
             "|---|---|---|---|---|---|"]
    for v, r in out.items():
        cmp = r.get("vs_old_ensemble")
        lines.append(f"| {v} | {r['offline']['n_seeds']} | {100 * r['offline']['single']:.2f} | {100 * r['offline']['ensemble']:.2f} | "
                     f"{100 * r['online48']['ensemble']:.2f} | "
                     + (f"{100 * (cmp['new'] - cmp['old']):+.2f}, better for {cmp['better_for']}/9, p = {cmp['wilcoxon_p']:.3f}" if cmp else "-")
                     + " |")
    lines += ["", "Nzakuna et al. 2025 (EEGNet, same protocol): " + ", ".join(f"{k} {100 * a:.2f}" for k, a in PAPER.items())]
    (OUT / "t2e_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cv", "summary", "t2e", "t2e_summary"])
    ap.add_argument("--variants", nargs="+", default=["BASE"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    log(f"=== {a.phase} {' '.join(a.variants)} seeds {a.seeds} ===")
    if a.phase == "cv":
        phase_cv(a.variants, a.seeds)
    elif a.phase == "summary":
        phase_summary()
    elif a.phase == "t2e":
        for v in a.variants:
            phase_t2e(v, a.epochs or load_json(OUT / "cv_summary.json")[v]["best_epoch"], a.seeds)
    else:
        phase_t2e_summary()
    log(f"=== {a.phase} finished ===")
