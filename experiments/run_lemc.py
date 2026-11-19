"""A1-A3 on the improved model (EA + S&R EEGNet): honest LEMC-style combination, fusion with Mam's
classifier, and the baseline paper's own (test-label) protocol for a side comparison. No training of EEGNets:
uses the saved T-CV logits (seeds 0-4) and session-E logits (seeds 0-14) from run_ens.py.

A1 (honest, chosen on T-CV only; a rule needs >= +0.5 points over the plain mean to be kept)
   PLAIN   mean of the members' softmax
   TEMP    per-model temperature (per subject and seed), then mean.   T-CV: fitted on 5 runs, applied to the 6th
   LEMC    TEMP + weighting MLP (2 -> 16 -> 8 -> 1, softplus; inputs max-prob and entropy of each member,
           trained with NLL of the weighted ensemble), as in Nzakuna et al. T-CV: trained on the other 8 subjects
   For E: temperatures from all 6 CV runs (seeds 5-14 use seed % 5), MLP trained on all 9 subjects' T-CV.
A2 fusion with Mam's sparse classifier: log p + w log p_sparse, w in 0..2 chosen on T-CV, kept if >= +0.5
A3 the paper's protocol (NOT honest, side comparison only): temperatures and MLP fitted on session-E labels and
   evaluated on the same E trials, with MC-dropout members as in LEMC-FTDB.
-> results/lemc/summary.json, summary.md
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import uncertainty as U  # noqa: E402
from data import SUBJECTS  # noqa: E402
from run_mentor import load_json, save_json, sparse_probs  # noqa: E402

SRC = ROOT / "results" / "stageB_aug"
OUT = ROOT / "results" / "lemc"
SEEDS_CV, SEEDS_E = [0, 1, 2, 3, 4], list(range(15))
RUN = np.repeat(np.arange(6), 48)
KEEP = 0.005
SP = load_json(ROOT / "results" / "mentor" / "sparse_selection.json")
SPARSE_W = np.round(np.arange(0, 2.01, 0.1), 2)


def acc(p, y):
    return float((p.argmax(1) == y).mean())


def cv_logits(v, s, k):
    parts = [load_json(SRC / "within_cv" / v / f"{s}_run{r}_seed{k}.json") for r in range(1, 7)]
    return np.concatenate([p["logits"] for p in parts]), np.concatenate([p["y"] for p in parts])


def e_logits(v, s, k, ref="offline"):
    r = load_json(SRC / "export" / v / f"{s}_seed{k}.json")
    return np.array(r["logits"][ref]), np.array(r["y"])


def sparse(kind, s):
    r = load_json(ROOT / "results" / "mentor" / kind / SP["chosen"] / f"{s}.json")
    return sparse_probs(np.array(r["scores"][SP["rule"]]), SP["prob_scale"]), np.array(r["y"])


# ------------------------------------------------------------------ weighting MLP (their architecture)
def feats(P):
    """P: (members, n, 4) -> (members, n, 2) features: max probability and entropy."""
    return np.stack([P.max(-1), U.entropy(P.reshape(-1, 4)).reshape(P.shape[:2])], -1)


def fit_mlp(P_list, y_list, epochs=200, seed=0):
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8), torch.nn.ReLU(),
                              torch.nn.Linear(8, 1), torch.nn.Softplus())
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    P = torch.tensor(np.concatenate(P_list, 1), dtype=torch.float32)          # (M, N, 4)
    F = torch.tensor(feats(np.concatenate(P_list, 1)), dtype=torch.float32)   # (M, N, 2)
    y = torch.tensor(np.concatenate(y_list))
    for _ in range(epochs):
        w = net(F) + 1e-6                                                     # (M, N, 1)
        p = (w * P).sum(0) / w.sum(0)
        loss = -torch.log(p[torch.arange(len(y)), y] + 1e-9).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def apply_mlp(net, P):
    import torch
    with torch.no_grad():
        w = net(torch.tensor(feats(P), dtype=torch.float32)).numpy() + 1e-6
    return (w * P).sum(0) / w.sum(0)


# ------------------------------------------------------------------ A1 on T-CV
def cv_member_probs(v="AUGSR"):
    """Per subject: y, plain member probs (M, n, 4) and temperature-scaled ones (T fitted on the other 5 runs)."""
    out = {}
    for s in SUBJECTS:
        Z = [cv_logits(v, s, k) for k in SEEDS_CV]
        y = Z[0][1]
        plain = np.stack([U.softmax(z) for z, _ in Z])
        temp = np.zeros_like(plain)
        for m, (z, _) in enumerate(Z):
            for r in range(6):
                te = RUN == r
                T = U.fit_temperature(z[~te], y[~te])
                temp[m, te] = U.softmax(z[te] / T)
        out[s] = {"y": y, "plain": plain, "temp": temp, "T_all": [U.fit_temperature(z, y) for z, _ in Z]}
    return out


def rules_on_cv(cv):
    res = {s: {"PLAIN": cv[s]["plain"].mean(0), "TEMP": cv[s]["temp"].mean(0)} for s in SUBJECTS}
    for s in SUBJECTS:          # LEMC: MLP trained on the other 8 subjects' (temperature-scaled) CV outputs
        others = [o for o in SUBJECTS if o != s]
        net = fit_mlp([cv[o]["temp"] for o in others], [cv[o]["y"] for o in others])
        res[s]["LEMC"] = apply_mlp(net, cv[s]["temp"])
    return res


def mean_acc(probs_by_s, ys):
    return float(np.mean([acc(probs_by_s[s], ys[s]) for s in SUBJECTS]))


def rejection(cv_final, e_final):
    """Stage B (run_calib) on the final system treated as one model: calibration data = its T-CV outputs."""
    import shutil
    import run_calib as rc
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "artifacts.json").exists():
        shutil.copy(ROOT / "results" / "stageB" / "artifacts.json", OUT / "artifacts.json")
    rc.OUT = OUT
    rc._cv = lambda variant, s, seed: cv_final[s]
    rc._within_test = lambda variant, ref, s, seed: e_final[ref][s]
    rc.SETTINGS = {"within": ("within", "FINAL", "offline", 0, "cv"),
                   "within_online48": ("within", "FINAL", "online48", 48, "cv")}
    return {n: rc.metrics(rc.evaluate(n, [0])) for n in rc.SETTINGS}


# ------------------------------------------------------------------ main
def main():
    from scipy.stats import wilcoxon
    out = {}
    cv = cv_member_probs()
    ys = {s: cv[s]["y"] for s in SUBJECTS}
    cvr = rules_on_cv(cv)
    a1_cv = {r: mean_acc({s: cvr[s][r] for s in SUBJECTS}, ys) for r in ("PLAIN", "TEMP", "LEMC")}
    best = max(a1_cv, key=a1_cv.get)
    chosen = best if a1_cv[best] >= a1_cv["PLAIN"] + KEEP else "PLAIN"
    out["A1_T_cv"] = {"acc": a1_cv, "chosen": chosen}

    # A2 fusion on T-CV, on top of the chosen rule
    spcv = {s: sparse("sparse_cv", s) for s in SUBJECTS}
    assert all((spcv[s][1] == ys[s]).all() for s in SUBJECTS)
    fus_cv = {float(w): mean_acc({s: np.log(cvr[s][chosen] + 1e-9) + w * np.log(spcv[s][0] + 1e-9) for s in SUBJECTS}, ys)
              for w in SPARSE_W}
    w_best = max(fus_cv, key=fus_cv.get)
    w_use = w_best if fus_cv[w_best] >= fus_cv[0.0] + KEEP else 0.0
    out["A2_T_cv"] = {"acc_by_weight": fus_cv, "best_weight": float(w_best), "kept_weight": float(w_use)}

    # session E (offline and first-48 EA reference)
    net_all = fit_mlp([cv[s]["temp"] for s in SUBJECTS], [ys[s] for s in SUBJECTS])
    old = {}
    for f in (ROOT / "results" / "ea" / "within_t2e").glob("*.json"):
        r = load_json(f)
        old.setdefault(r["subject"], []).append((np.array(r["probs"]["offline"]), np.array(r["y"])))
    old_acc = np.array([acc(np.mean([p for p, _ in old[s]], 0), old[s][0][1]) for s in SUBJECTS])
    E, FINAL = {}, {}
    for ref in ("offline", "online48"):
        per = {k: {} for k in ("PLAIN", "TEMP", "LEMC", "FUSED_chosen", "FUSED_bestw")}
        for s in SUBJECTS:
            Z = [e_logits("AUGSR", s, k, ref) for k in SEEDS_E]
            y = Z[0][1]
            plain = np.stack([U.softmax(z) for z, _ in Z])
            temp = np.stack([U.softmax(z / cv[s]["T_all"][k % 5]) for k, (z, _) in enumerate(Z)])
            p = {"PLAIN": plain.mean(0), "TEMP": temp.mean(0), "LEMC": apply_mlp(net_all, temp)}
            spe, ysp = sparse("sparse_t2e", s)
            assert (ysp == y).all()
            for r in ("PLAIN", "TEMP", "LEMC"):
                per[r][s] = acc(p[r], y)
            fz = np.log(p[chosen] + 1e-9) + w_use * np.log(spe + 1e-9)
            FINAL.setdefault(ref, {})[s] = (fz, y)
            per["FUSED_chosen"][s] = acc(fz, y)
            per["FUSED_bestw"][s] = acc(np.log(p[chosen] + 1e-9) + w_best * np.log(spe + 1e-9), y)
        E[ref] = {}
        for r, d in per.items():
            a = np.array([d[s] for s in SUBJECTS])
            try:
                pv = float(wilcoxon(a, old_acc).pvalue)
            except ValueError:
                pv = float("nan")
            E[ref][r] = {"mean": float(a.mean()), "per_subject": d, "better_than_old": int((a > old_acc).sum()), "p_vs_old": pv}
    out["session_E"] = E

    # A3 the paper's protocol: temperatures + MLP fitted on E labels, scored on the same E trials (not honest)
    a3 = {}
    for v in ("AUGSR", "AUGSR_MC"):
        Ps, Yl = {}, {}
        for s in SUBJECTS:
            Z = [e_logits(v, s, k) for k in SEEDS_E]
            y = Z[0][1]
            Ps[s] = np.stack([U.softmax(z / U.fit_temperature(z, y)) for z, _ in Z])
            Yl[s] = y
        net = fit_mlp(list(Ps.values()), list(Yl.values()))
        a3[v] = mean_acc({s: apply_mlp(net, Ps[s]) for s in SUBJECTS}, Yl)
    out["A3_paper_protocol_E"] = a3
    cv_final = {s: (np.log(cvr[s][chosen] + 1e-9) + w_use * np.log(spcv[s][0] + 1e-9), ys[s]) for s in SUBJECTS}
    out["rejection"] = rejection(cv_final, FINAL)
    save_json(OUT / "summary.json", out)

    pct = lambda x: f"{100 * x:.2f}"
    L = ["# A1-A3: honest LEMC-style combination, fusion with Mam's classifier, paper protocol", "",
         "## Chosen on session-T CV (5 models)", "",
         "| Rule | T-CV acc |", "|---|---|"] + [f"| {r} | {pct(a)} |" for r, a in a1_cv.items()]
    L += ["", f"Chosen rule: **{chosen}** (keep needs +0.5 over PLAIN)", "",
          f"Fusion with Mam's classifier on T-CV: weight 0 -> {pct(fus_cv[0.0])}, best weight {w_best} -> "
          f"{pct(fus_cv[w_best])}; kept weight **{w_use}**", "",
          "## Session E (15 models), mean of 9 subjects", "",
          "| Combination | EA ref: all E trials | EA ref: first 48 trials | vs old 76.16 (offline) |", "|---|---|---|---|"]
    names = {"PLAIN": "plain mean", "TEMP": "per-model temperature", "LEMC": "temperature + weighting MLP (honest LEMC)",
             "FUSED_chosen": f"chosen rule + Mam's classifier (w={w_use}, as decided on T)",
             "FUSED_bestw": f"chosen rule + Mam's classifier (w={w_best}, T-CV best, for reference)"}
    for r, n in names.items():
        o, on = E["offline"][r], E["online48"][r]
        L.append(f"| {n} | {pct(o['mean'])} | {pct(on['mean'])} | {o['better_than_old']}/9, p = {o['p_vs_old']:.3f} |")
    L += ["", "## Side comparison: the paper's own protocol (temperatures + MLP fitted on the session-E labels; NOT honest)", "",
          f"- 15 nets, deterministic: {pct(a3['AUGSR'])}",
          f"- 15 nets, MC dropout (as LEMC-FTDB): {pct(a3['AUGSR_MC'])}",
          "- Paper LEMC-FTDB (EEGNet, this protocol): 80.40; paper honest EMCD (15 nets): 75.35"]
    L += ["", "## Rejection on the final system (temperature + entropy threshold from T-CV of the final system)", "",
          "| EA reference | Acc | ECE raw -> cal | AUROC | 80% cov: coverage / accepted acc / false per 100 "
          "| 85% target: coverage / accepted acc / false per 100 |", "|---|---|---|---|---|---|"]
    for name, m in out["rejection"].items():
        ops = ["unreachable" if m[op] is None else f"{pct(m[op]['coverage'])} / {pct(m[op]['accepted_accuracy'])} / "
               f"{m[op]['false_per_100']:.1f}" for op in ("cov80", "acc_target")]
        L.append(f"| {name} | {pct(m['accuracy'])} | {pct(m['ece_raw'])} -> {pct(m['ece_cal'])} | {m['auroc']['entropy']:.3f} | "
                 + " | ".join(ops) + " |")
    L += ["", "Per subject (all E trials): accuracy | 80% cov accepted acc | 85% target coverage", ""]
    for s, row in out["rejection"]["within"]["by_subject"].items():
        at = row.get("acc_target")
        L.append(f"- {s}: {pct(row['accuracy'])} | {pct(row['cov80']['accepted_accuracy'])} | {pct(at['coverage']) if at else '-'}")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
