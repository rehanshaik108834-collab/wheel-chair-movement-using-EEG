"""B4-B6: a NEW user (leave-one-subject-out), with the improvements and with adaptation to the new user.

Same setup as Stage A LOSO (run_ea.phase_loso): train on sessions T+E of the 8 other subjects (each session aligned
with its own EA reference), early stopping on a stratified 20% split of those subjects (val NLL, patience 80, max 600),
dropout 0.25, batch 64, Adam 1e-3, seeds 0-2. Nothing is tuned on the new user; all recipes below are fixed in advance.

  train   B4: + segmentation & recombination (segments from the same source subject AND class)
  adapt   per new user (session E), from the saved models:
            NONE   no adaptation (EA reference from all E trials = offline, or from the first 48 = online)
            BN     B5: BatchNorm statistics recomputed on the new user's UNLABELLED trials (all, or first 48)
            FT48   B6: BN on the first 48 trials, then fine-tuning on those 48 LABELLED trials
                   (12 per class; Adam 1e-4, batch 16, 60 epochs, S&R on, all layers); scored on trials 49-288
  final   accuracy table (offline: all 288 trials; online: trials 49-288 for every method) vs the old EA LOSO models,
          then rejection (temperature + entropy threshold from the OTHER 8 new users' outputs, as in Stage B)
          -> results/cross/summary.json, summary.md
"""
import argparse
import shutil
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
import run_improve as ri  # noqa: E402
from data import SUBJECTS  # noqa: E402
from preprocess import load_config_data  # noqa: E402
from run_mentor import acc, load_json, save_json  # noqa: E402

SMOKE = "--smoke" in sys.argv
OUT = ROOT / "results" / ("cross_smoke" if SMOKE else "cross")
CKPT = ROOT / "checkpoints" / ("cross_smoke" if SMOKE else "cross")
SEEDS = [0] if SMOKE else [0, 1, 2]
SUBJ = SUBJECTS[:3] if SMOKE else SUBJECTS
CROSS = {"lr": 1e-3, "batch_size": 64, "dropout": 0.25, "max_epochs": 4 if SMOKE else 600, "patience": 2 if SMOKE else 80}
FT = {"lr": 1e-4, "batch_size": 16, "epochs": 3 if SMOKE else 60, "n": 48}
METHODS = {  # name: (logits key, first scored trial)
    "OLD_offline": ("OLD_offline", 0), "OLD_online": ("OLD_online48", 48),
    "NONE_offline": ("NONE_offline", 0), "NONE_online": ("NONE_online48", 48),
    "BN_offline": ("BN_offline", 0), "BN_online": ("BN_online48", 48),
    "FT48": ("FT48", 48)}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "logs" / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


class GroupNet(ri.Net):
    """ri.Net whose S&R draws segments from trials of the same group (= source subject x class)."""
    groups = None

    def _sr(self, X, y):
        return super()._sr(X, self.groups if self.groups is not None else y)


def nll(p, y):
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-9)))


def ref(X, k):
    return ea.align(X, ea.reference_covariance(X if k is None else X[:k]))


def phase_train(seeds, targets):
    from sklearn.model_selection import train_test_split
    import torch
    from train import ChannelScaler
    sp = ri.spec("AUGSR")
    data = {(s, sess): load_config_data("A", s, sess) for s in SUBJECTS for sess in "TE"}
    aligned = {k: ea.align_self(v[0])[0] for k, v in data.items()}
    for seed in seeds:
        for target in targets:
            path = OUT / "train" / f"{target}_seed{seed}.json"
            if path.exists():
                continue
            t0 = time.time()
            src = [(s, sess) for s in SUBJECTS if s != target for sess in "TE"]
            X = np.concatenate([aligned[k] for k in src])
            y = np.concatenate([data[k][1] for k in src])
            subj = np.concatenate([np.full(288, SUBJECTS.index(k[0])) for k in src])
            strata = np.concatenate([np.full(288, i * 4) for i in range(len(src))]) + y
            tri, vai = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=strata)
            sc = ChannelScaler().fit(X[tri])
            net = GroupNet(sp, X.shape[2], seed, CROSS["max_epochs"], lr=CROSS["lr"], batch_size=CROSS["batch_size"],
                           dropout=CROSS["dropout"])
            Xt, yt = net.data(sc.transform(X[tri]), y[tri])
            net.groups = torch.as_tensor(subj[tri] * 4 + y[tri], device=net.dev)
            Xv = net.data(sc.transform(X[vai]))
            hist, best, best_state, best_ep, wait = [], np.inf, None, 0, 0
            for ep in range(1, CROSS["max_epochs"] + 1):
                net.epoch(Xt, yt)
                hist.append(nll(net.probs(Xv), y[vai]))
                if hist[-1] < best:
                    best, best_ep, wait = hist[-1], ep, 0
                    best_state = {k: t.detach().clone() for k, t in net.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= CROSS["patience"]:
                        break
            net.model.load_state_dict(best_state)
            pv = net.probs(Xv)
            CKPT.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "scaler_mean": sc.mean_, "scaler_std": sc.std_, "best_epoch": best_ep},
                       CKPT / f"{target}_seed{seed}.pt")
            save_json(path, {"subject": target, "seed": seed, "best_epoch": best_ep, "epochs_run": len(hist),
                             "val_nll": hist, "val_acc": acc(y[vai], pv), "seconds": round(time.time() - t0, 1)})
            log(f"train {target} seed{seed}: best epoch {best_ep}/{len(hist)}, source val acc {acc(y[vai], pv):.3f} "
                f"({time.time() - t0:.1f} s)")


def bn_adapt(model, X):
    """Recompute every BatchNorm's running mean/variance on X (no labels, no gradient); dropout stays off."""
    import torch
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad():
        model(X)
    model.eval()


def phase_adapt():
    import torch
    from models import EEGNet
    from train import DEVICE, ChannelScaler
    sp = ri.spec("AUGSR")
    for target in SUBJ:
        Xe, ye = load_config_data("A", target, "E")
        for seed in SEEDS:
            path = OUT / "adapt" / f"{target}_seed{seed}.json"
            if path.exists():
                continue
            t0 = time.time()
            ck = torch.load(CKPT / f"{target}_seed{seed}.pt", map_location=DEVICE, weights_only=False)
            sc = ChannelScaler()
            sc.mean_, sc.std_ = np.asarray(ck["scaler_mean"]), np.asarray(ck["scaler_std"])
            T = lambda X: torch.as_tensor(np.ascontiguousarray(sc.transform(X), dtype=np.float32), device=DEVICE)
            Xoff, Xon = T(ref(Xe, None)), T(ref(Xe, FT["n"]))

            def fresh():
                m = EEGNet(n_channels=22, n_times=Xe.shape[2], dropout=CROSS["dropout"]).to(DEVICE)
                m.load_state_dict(ck["state_dict"]); m.eval()
                return m

            def logp(m, X):
                with torch.no_grad():
                    m.eval()
                    return torch.log_softmax(m(X), 1).cpu().numpy()

            lg = {}
            m = fresh(); lg["NONE_offline"], lg["NONE_online48"] = logp(m, Xoff), logp(m, Xon)
            m = fresh(); bn_adapt(m, Xoff); lg["BN_offline"] = logp(m, Xoff)
            m = fresh(); bn_adapt(m, Xon[:FT["n"]]); lg["BN_online48"] = logp(m, Xon)
            # FT48: start from the BN(first 48) model, fine-tune on the first 48 labelled trials
            net = ri.Net(sp, Xe.shape[2], 1000 + seed, FT["epochs"], lr=FT["lr"], batch_size=FT["batch_size"],
                         dropout=CROSS["dropout"])
            net.model = m
            net.opt = torch.optim.Adam(m.parameters(), lr=FT["lr"])
            yt = torch.as_tensor(ye[:FT["n"]], device=DEVICE)
            for _ in range(FT["epochs"]):
                net.epoch(Xon[:FT["n"]], yt)
            lg["FT48"] = logp(net.model, Xon)
            res = {"subject": target, "seed": seed, "y": ye.tolist(),
                   "logits": {k: np.round(v, 5).tolist() for k, v in lg.items()},
                   "acc": {k: acc(ye[METHODS_START(k):], np.exp(v[METHODS_START(k):])) for k, v in lg.items()},
                   "seconds": round(time.time() - t0, 1)}
            save_json(path, res)
            log(f"adapt {target} seed{seed}: " + ", ".join(f"{k} {a:.3f}" for k, a in res["acc"].items()))


def METHODS_START(key):
    return 0 if key.endswith("offline") else 48


def load_method(method, s, seed):
    """(log-probs over all 288 E trials, y) for one new user and seed."""
    key = METHODS[method][0]
    if key.startswith("OLD"):
        r = load_json(ROOT / "results" / "ea" / "loso" / "EA" / f"{s}_seed{seed}.json")
        return np.log(np.array(r["probs"][key.split("_")[1]]) + 1e-9), np.array(r["y"])
    r = load_json(OUT / "adapt" / f"{s}_seed{seed}.json")
    return np.array(r["logits"][key]), np.array(r["y"])


def phase_final():
    from scipy.stats import wilcoxon
    import run_calib as rc
    res = {"accuracy": {}, "rejection": {}}
    per = {}
    for m, (_, start) in METHODS.items():
        single, ens = {}, {}
        for s in SUBJ:
            runs = [load_method(m, s, k) for k in SEEDS]
            y = runs[0][1][start:]
            single[s] = float(np.mean([acc(y, np.exp(z[start:])) for z, _ in runs]))
            ens[s] = acc(y, np.mean([np.exp(z[start:]) for z, _ in runs], 0))
        per[m] = ens
        res["accuracy"][m] = {"single": float(np.mean(list(single.values()))), "ensemble": float(np.mean(list(ens.values()))),
                              "per_subject_ensemble": ens}
    for m in METHODS:
        base = "OLD_offline" if m.endswith("offline") else "OLD_online"
        a, b = np.array([per[m][s] for s in SUBJ]), np.array([per[base][s] for s in SUBJ])
        try:
            pv = float(wilcoxon(a, b).pvalue)
        except ValueError:
            pv = float("nan")
        res["accuracy"][m]["vs_old"] = {"vs": base, "better_for": int((a > b).sum()), "wilcoxon_p": pv}
    # rejection, calibrated on the other new users' outputs ("others" in Stage B)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "artifacts.json").exists():
        shutil.copy(ROOT / "results" / "stageB" / "artifacts.json", OUT / "artifacts.json")
    rc.OUT, rc.SUBJECTS = OUT, SUBJ

    def _cross(variant, s, seed):
        z, y = load_method(variant, s, seed)
        return {"logits": {"x": z}, "y": y}
    rc._cross = _cross
    rc.SETTINGS = {m: ("cross", m, "x", start, "others") for m, (_, start) in METHODS.items()}
    for m in METHODS:
        res["rejection"][m] = rc.metrics(rc.evaluate(m, SEEDS))
    save_json(OUT / "summary.json", res)
    write_md(res)
    log("final: summary written")


def write_md(res):
    pct = lambda v: "–" if v is None else f"{100 * v:.2f}"
    names = {"OLD_offline": "old EA model, no adaptation", "NONE_offline": "+ S&R (B4), no adaptation",
             "BN_offline": "+ BatchNorm adaptation on all unlabelled trials (B5)",
             "OLD_online": "old EA model, no adaptation", "NONE_online": "+ S&R (B4), no adaptation",
             "BN_online": "+ BatchNorm adaptation on the first 48 unlabelled trials (B5)",
             "FT48": "+ fine-tuning on the first 48 labelled trials (B6)"}
    L = ["# A new user (leave-one-subject-out), BCI IV-2a, 3-model ensembles", "",
         "Offline: EA reference from all 288 E trials of the new user, scored on all 288.",
         "Online: EA reference (and BN / fine-tuning data) from the first 48 E trials only, scored on trials 49-288.", "",
         "| Setting | Method | 1 model | Ensemble | vs old: better for, p | Rejection 80% cov: coverage / accepted acc / false per 100 "
         "| 65% target: coverage / accepted acc |", "|---|---|---|---|---|---|---|"]
    for m, n in names.items():
        a, r = res["accuracy"][m], res["rejection"][m]
        c = r["cov80"]
        t = r["acc_target"]
        L.append(f"| {'offline' if m.endswith('offline') else 'online'} | {n} | {pct(a['single'])} | {pct(a['ensemble'])} | "
                 f"{a['vs_old']['better_for']}/9, p = {a['vs_old']['wilcoxon_p']:.3f} | {pct(c['coverage'])} / "
                 f"{pct(c['accepted_accuracy'])} / {c['false_per_100']:.1f} | "
                 f"{'unreachable' if t is None else pct(t['coverage']) + ' / ' + pct(t['accepted_accuracy'])} |")
    L += ["", "Per subject (ensemble): " + ", ".join(SUBJ), ""]
    for m in names:
        L.append(f"- {m}: " + ", ".join(pct(res['accuracy'][m]['per_subject_ensemble'][s]) for s in SUBJ))
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "adapt", "final"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    log(f"=== {a.phase} ===")
    if a.phase == "train":
        phase_train(a.seeds or SEEDS, a.subjects or SUBJ)
    elif a.phase == "adapt":
        phase_adapt()
    else:
        phase_final()
    log(f"=== {a.phase} finished ===")
