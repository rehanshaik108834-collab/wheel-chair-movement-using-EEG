"""Part 1.6 + Stage B on the improved model (EA EEGNet + segmentation & recombination, "AUGSR", 290 epochs).

Every choice uses session T only; session E is scored once, in `final`.
Phases
  cv      --seeds S... [--subjects A01 ...]   leave-one-run-out CV inside T at 290 epochs. Per held-out run:
          deterministic logits (variant AUGSR) and the log of the MC-dropout mean probability over 50 passes
          (variant AUGSR_MC) -> results/stageB_aug/within_cv/{AUGSR,AUGSR_MC}/{s}_run{r}_seed{k}.json
  export  from the final models checkpoints/improve/AUGSR/{s}_seed{k}.pt (seeds 0-14, trained on all of T):
          E logits and MC-dropout mean (200 passes), EA reference from all / first 48 / first 24 E trials
  final   1) ensemble rule chosen on T-CV (5 seeds): plain mean vs MC-dropout mean; MC needs >= +0.5 points
          2) session E accuracy, 5 and 15 models, vs the old EA ensemble (76.16) and the paper
          3) Stage B (run_calib: temperature + entropy thresholds from T-CV) on the chosen rule
          -> results/stageB_aug/summary.json, summary.md
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
import run_calib as rc  # noqa: E402
import run_improve as ri  # noqa: E402
from data import SUBJECTS  # noqa: E402
from preprocess import load_config_data  # noqa: E402
from run_mentor import load_json, save_json  # noqa: E402

SMOKE = "--smoke" in sys.argv
VARIANT, EPOCHS = "AUGSR", (3 if SMOKE else 290)
OUT = ROOT / "results" / ("stageB_aug_smoke" if SMOKE else "stageB_aug")
CKPT = ROOT / "checkpoints" / "improve" / VARIANT
MC_CV, MC_E = (5, 5) if SMOKE else (50, 200)
SEEDS_CV = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
SEEDS_E = [0, 1] if SMOKE else list(range(15))
SUBJ = SUBJECTS[:2] if SMOKE else SUBJECTS
REFS = {"offline": None, "online48": 48, "online24": 24}
OLD = {"within_EA ensemble (Stage B)": "results/stageB/summary.json"}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "logs" / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def logits_and_mc(model, X, passes):
    """Deterministic logits and log of the MC-dropout mean probability (dropout on, BatchNorm in eval mode)."""
    import torch
    model.eval()
    with torch.no_grad():
        lg = model(X)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        p = sum(torch.softmax(model(X), 1) for _ in range(passes)) / passes
    model.eval()
    return lg.cpu().numpy().astype(np.float64), np.log(p.cpu().numpy().astype(np.float64) + 1e-12)


def phase_cv(seeds, subjects):
    from train import ChannelScaler
    sp = ri.spec(VARIANT)
    for seed in seeds:
        for s in subjects:
            X, y = load_config_data(sp["config"], s, "T")
            for run in range(6):
                paths = {v: OUT / "within_cv" / v / f"{s}_run{run + 1}_seed{seed}.json" for v in (VARIANT, VARIANT + "_MC")}
                if all(p.exists() for p in paths.values()):
                    continue
                t0 = time.time()
                te = ri.RUN_OF_TRIAL == run
                Xtr, Xte = ea.align_self(X[~te])[0], ea.align_self(X[te])[0]
                sc = ChannelScaler().fit(Xtr)
                net = ri.Net(sp, X.shape[2], seed * 100 + run, EPOCHS)
                Xt, yt = net.data(sc.transform(Xtr), y[~te])
                for _ in range(EPOCHS):
                    net.epoch(Xt, yt)
                lg, mc = logits_and_mc(net.model, net.data(sc.transform(Xte)), MC_CV)
                for v, z in ((VARIANT, lg), (VARIANT + "_MC", mc)):
                    save_json(paths[v], {"variant": v, "subject": s, "held_out_run": run + 1, "seed": seed, "epochs": EPOCHS,
                                         "y": y[te].tolist(), "logits": np.round(z, 5).tolist(),
                                         "acc": float((z.argmax(1) == y[te]).mean()), "seconds": round(time.time() - t0, 1)})
                log(f"cv {s} run{run + 1} seed{seed}: acc {(lg.argmax(1) == y[te]).mean():.3f}, "
                    f"MC {(mc.argmax(1) == y[te]).mean():.3f} ({time.time() - t0:.1f} s)")


def phase_export():
    import torch
    from models import EEGNet
    from train import DEVICE, ChannelScaler
    for s in SUBJ:
        Xe, ye = load_config_data("A", s, "E")
        for seed in SEEDS_E:
            paths = {v: OUT / "export" / v / f"{s}_seed{seed}.json" for v in (VARIANT, VARIANT + "_MC")}
            if all(p.exists() for p in paths.values()):
                continue
            ck = torch.load(CKPT / f"{s}_seed{seed}.pt", map_location=DEVICE, weights_only=False)
            model = EEGNet(n_channels=22, n_times=Xe.shape[2]).to(DEVICE)
            model.load_state_dict(ck["state_dict"])
            sc = ChannelScaler()
            sc.mean_, sc.std_ = np.asarray(ck["scaler_mean"]), np.asarray(ck["scaler_std"])
            out = {VARIANT: {}, VARIANT + "_MC": {}}
            for r, k in REFS.items():
                Xa = torch.as_tensor(np.ascontiguousarray(sc.transform(rc.align_target(Xe, k)), dtype=np.float32), device=DEVICE)
                out[VARIANT][r], out[VARIANT + "_MC"][r] = logits_and_mc(model, Xa, MC_E)
            saved = load_json(ROOT / "results" / "improve" / "t2e" / VARIANT / f"{s}_seed{seed}.json")
            agree = float((out[VARIANT]["offline"].argmax(1) == np.array(saved["probs"]["offline"]).argmax(1)).mean())
            for v, d in out.items():
                save_json(paths[v], {"variant": v, "subject": s, "seed": seed, "y": ye.tolist(),
                                     "logits": {r: np.round(z, 5).tolist() for r, z in d.items()},
                                     "agree_with_saved": agree})
            log(f"export {s} seed{seed}: agreement with the saved run {agree:.3f}")


def _cv(variant, s, seed):
    """T-CV logits. Seeds 5-14 have no CV models: they use the CV of seed % 5 (calibration = 5-model CV ensemble)."""
    seed = SEEDS_CV[seed % len(SEEDS_CV)]
    parts = [load_json(OUT / "within_cv" / variant / f"{s}_run{r}_seed{seed}.json") for r in range(1, 7)]
    return np.concatenate([p["logits"] for p in parts]), np.concatenate([p["y"] for p in parts])


def _within_test(variant, ref, s, seed):
    r = load_json(OUT / "export" / variant / f"{s}_seed{seed}.json")
    return np.array(r["logits"][ref]), np.array(r["y"])


def ens_acc(variant, s, seeds, getter, ref="offline"):
    zs = [getter(variant, s, k) if getter is _cv else _within_test(variant, ref, s, k) for k in seeds]
    p = np.mean([rc.U.softmax(z[0]) for z in zs], 0)
    return float((p.argmax(1) == zs[0][1]).mean())


def phase_final():
    from scipy.stats import wilcoxon
    res = {}
    # 1) rule on T-CV
    cv = {v: {s: ens_acc(v, s, SEEDS_CV, _cv) for s in SUBJ} for v in (VARIANT, VARIANT + "_MC")}
    cv_single = {v: float(np.mean([np.mean([ens_acc(v, s, [k], _cv) for k in SEEDS_CV]) for s in SUBJ]))
                 for v in (VARIANT, VARIANT + "_MC")}
    gain = np.mean(list(cv[VARIANT + "_MC"].values())) - np.mean(list(cv[VARIANT].values()))
    chosen = VARIANT + "_MC" if gain >= ri.KEEP_RULE else VARIANT
    res["rule_on_T_cv"] = {"ensemble5": {v: float(np.mean(list(d.values()))) for v, d in cv.items()},
                           "single": cv_single, "mc_gain": float(gain), "chosen": chosen}
    log(f"T-CV: plain {np.mean(list(cv[VARIANT].values())):.4f}, MC {np.mean(list(cv[VARIANT + '_MC'].values())):.4f} -> {chosen}")
    # 2) session E
    old = {}
    for f in (ROOT / "results" / "ea" / "within_t2e").glob("*.json"):
        r = load_json(f)
        old.setdefault(r["subject"], []).append((np.array(r["probs"]["offline"]), np.array(r["y"])))
    old_acc = {s: float((np.mean([p for p, _ in old[s]], 0).argmax(1) == old[s][0][1]).mean()) for s in SUBJ}
    e = {}
    for v in (VARIANT, VARIANT + "_MC"):
        for n, seeds in (("5", SEEDS_E[:5]), ("15", SEEDS_E)):
            for ref in ("offline", "online48"):
                per = {s: ens_acc(v, s, seeds, None, ref) for s in SUBJ}
                single = float(np.mean([np.mean([ens_acc(v, s, [k], None, ref) for k in seeds]) for s in SUBJ]))
                a, b = np.array([per[s] for s in SUBJ]), np.array([old_acc[s] for s in SUBJ])
                try:
                    pv = float(wilcoxon(a, b).pvalue)
                except ValueError:
                    pv = float("nan")
                e[f"{v}|{n}|{ref}"] = {"ensemble": float(a.mean()), "single_mean": single, "per_subject": per,
                                       "vs_old": {"old": float(b.mean()), "better_for": int((a > b).sum()), "wilcoxon_p": pv}}
    res["session_E"] = e
    # 3) Stage B on the chosen rule
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "artifacts.json").exists():
        shutil.copy(ROOT / "results" / "stageB" / "artifacts.json", OUT / "artifacts.json")
    rc.OUT, rc.SUBJECTS, rc._cv, rc._within_test = OUT, SUBJ, _cv, _within_test
    rc.SETTINGS = {"within": ("within", chosen, "offline", 0, "cv"), "within_online48": ("within", chosen, "online48", 48, "cv")}
    sb = {}
    for name in rc.SETTINGS:
        for n, seeds in (("1", None), ("5", SEEDS_E[:5]), ("15", SEEDS_E)):
            m = rc.average([rc.metrics(rc.evaluate(name, [k])) for k in SEEDS_E[:5]]) if seeds is None \
                else rc.metrics(rc.evaluate(name, seeds))
            sb[f"{name}|{n}"] = m
    res["stageB"] = sb
    save_json(OUT / "summary.json", res)
    write_md(res, chosen)
    log("final: written results/stageB_aug/summary.md")


def write_md(res, chosen):
    pct = lambda v: "–" if v is None else f"{100 * v:.2f}"
    r = res["rule_on_T_cv"]
    L = ["# Improved model (EA + S&R EEGNet): ensemble size, MC dropout, rejection", "",
         f"Rule chosen on T-CV (5-model ensemble): plain {pct(r['ensemble5'][VARIANT])}, MC dropout "
         f"{pct(r['ensemble5'][VARIANT + '_MC'])} (gain {100 * r['mc_gain']:+.2f}, keep needs +0.5) -> **{chosen}**", "",
         "## Session E (train T -> test E), mean of 9 subjects", "",
         "| Rule | Models | EA reference | 1 model | Ensemble | vs old 76.16: better for, p |", "|---|---|---|---|---|---|"]
    for k, v in res["session_E"].items():
        rule, n, ref = k.split("|")
        L.append(f"| {'MC dropout' if rule.endswith('_MC') else 'plain'}{' (chosen)' if rule == chosen else ''} | {n} | {ref} "
                 f"| {pct(v['single_mean'])} | {pct(v['ensemble'])} | {v['vs_old']['better_for']}/9, p = {v['vs_old']['wilcoxon_p']:.3f} |")
    L += ["", "Paper (Nzakuna et al. 2025, EEGNet): LEMC-FTDB 80.40 (temperatures/weights fitted on test labels), "
          "EMCD 15 nets honest 75.35, MCD 71.14", "",
          "## Rejection (temperature + entropy threshold from T-CV, one threshold for all users)", "",
          "| Setting | Models | Acc | ECE raw → cal | AUROC | 80% cov: coverage / accepted acc / false per 100 "
          "| 85% target: coverage / accepted acc / false per 100 |", "|---|---|---|---|---|---|---|"]
    for k, m in res["stageB"].items():
        name, n = k.split("|")
        ops = ["unreachable" if m[op] is None else f"{pct(m[op]['coverage'])} / {pct(m[op]['accepted_accuracy'])} / "
               f"{m[op]['false_per_100']:.1f}" for op in ("cov80", "acc_target")]
        L.append(f"| {name} | {n} | {pct(m['accuracy'])} | {pct(m['ece_raw'])} → {pct(m['ece_cal'])} "
                 f"| {m['auroc']['entropy']:.3f} | " + " | ".join(ops) + " |")
    key = "within|15"
    L += ["", "Per subject (offline EA reference, largest ensemble): accuracy | 80% cov accepted acc | 85% target coverage", ""]
    for s, row in res["stageB"][key]["by_subject"].items():
        at = row.get("acc_target")
        L.append(f"- {s}: {pct(row['accuracy'])} | {pct(row['cov80']['accepted_accuracy'])} | "
                 f"{pct(at['coverage']) if at else '–'}")
    L += ["", "Old Stage B (EA ensemble of 5, before S&R): accuracy 76.16, 80% coverage 82.9 / 82.0 / 14.9; "
          "85% target not reached (82.8 at 81.2% coverage)"]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cv", "export", "final"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    log(f"=== {a.phase} ===")
    if a.phase == "cv":
        phase_cv(a.seeds or SEEDS_CV, a.subjects or SUBJ)
    elif a.phase == "export":
        phase_export()
    else:
        phase_final()
    log(f"=== {a.phase} finished ===")
