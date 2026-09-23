"""Load the per-run results written by experiments/run_experiments.py."""
import json
from pathlib import Path

import numpy as np

from data import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def load_runs(phase: str, config: str) -> list:
    """All finished runs of one phase/config, as a list of dicts (sorted by file name)."""
    folder = RUNS_DIR / phase / config
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))] if folder.exists() else []


def selection() -> dict:
    return json.loads((PROJECT_ROOT / "results" / "selection.json").read_text())


def pooled_predictions(runs, key_true="y_true", key_pred="y_pred", session=None):
    """{subject: (y_true, y_pred)} with all seeds concatenated (for confusion matrices)."""
    out = {}
    for r in runs:
        src = r["target"][session] if session else r
        yt, yp = np.array(src[key_true]), np.array(src[key_pred])
        if r["subject"] in out:
            out[r["subject"]] = (np.concatenate([out[r["subject"]][0], yt]),
                                 np.concatenate([out[r["subject"]][1], yp]))
        else:
            out[r["subject"]] = (yt, yp)
    return dict(sorted(out.items()))
