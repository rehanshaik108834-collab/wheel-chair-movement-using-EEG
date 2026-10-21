"""Extra check: her evaluation protocol (segments randomly split into 10 folds, within T) applied to
settings whose features carry class information, compared with the trial-level CV used for our results.

For CSP settings, CSP is fitted on all T trials first (as a segment-level random split implies:
the spatial filters see every trial). Output: results/mentor/p0_extra.json
"""
import json
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import mentor_pipeline as mp  # noqa: E402
from data import SUBJECTS  # noqa: E402
from run_mentor import SubjectData, bands_of, load_json, sparse_grid  # noqa: E402

OUT = ROOT / "results" / "mentor"
names = ["bp_b1_all_s2_energy", "bp_csp_b9_all_s2_energy"]
res = {}
for name in names:
    cfg = next(c for c in sparse_grid() if c["name"] == name)
    res[name] = {}
    for s in SUBJECTS:
        d = SubjectData(s, "T", bands_of(cfg))
        Xb = d.bands(cfg)
        if cfg["case"] == "bp_csp":
            Xb = mp.apply_csp(mp.fit_csp(Xb, d.y), Xb)
        F = mp.features(Xb, cfg["seg"], cfg["feature"])
        n_seg, m = F.shape[1], F.shape[2]
        Fs, ys = F.reshape(-1, m), np.repeat(d.y, n_seg)
        pred = np.zeros(len(ys), dtype=int)
        for tr, te in KFold(10, shuffle=True, random_state=0).split(Fs):
            pred[te] = mp.SparseRepresentationClassifier().fit(Fs[tr], ys[tr]).scores(Fs[te])["R4"].argmax(1)
        cv = load_json(OUT / "sparse_cv" / name / f"{s}.json")["acc"]["R4"]
        res[name][s] = {"her_split_segment_acc": float((pred == ys).mean()), "trial_level_cv_acc": cv}
        print(name, s, res[name][s], flush=True)
    for k in ("her_split_segment_acc", "trial_level_cv_acc"):
        print(name, "MEAN", k, round(float(np.mean([res[name][s][k] for s in SUBJECTS])), 4), flush=True)
(OUT / "p0_extra.json").write_text(json.dumps(res, indent=1))
