"""Preprocessing pipeline: remove EOG -> 8-30 Hz band-pass -> 4 s epochs after each cue."""
from pathlib import Path

import mne
import numpy as np

from data import CLASS_NAMES, PROJECT_ROOT, SUBJECTS, cue_events, load_raw

PROCESSED_DIR = PROJECT_ROOT / "data_processed"

L_FREQ, H_FREQ = 8.0, 30.0   # mu (8-13 Hz) + beta (13-30 Hz)
TMIN, TMAX = 0.0, 4.0        # seconds relative to the cue (= 2-6 s of the trial)


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Keep the 22 EEG channels and band-pass filter the continuous signal."""
    raw = raw.copy().pick("eeg")
    raw.filter(L_FREQ, H_FREQ, method="fir", phase="zero", verbose="ERROR")
    return raw


def make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Cut one epoch per motor-imagery cue, TMIN..TMAX seconds after the cue."""
    events, event_id = cue_events(raw)
    sfreq = raw.info["sfreq"]
    return mne.Epochs(raw, events, event_id=event_id, tmin=TMIN, tmax=TMAX - 1 / sfreq,
                      baseline=None, preload=True, verbose="ERROR")


def preprocess_subject(subject: str, session: str = "T"):
    """Full pipeline for one recording. Returns X (trials, 22, 1000) in microvolts and y (0..3)."""
    epochs = make_epochs(preprocess_raw(load_raw(subject, session)))
    X = (epochs.get_data() * 1e6).astype(np.float32)
    y = epochs.events[:, 2].astype(np.int64)
    return X, y


def save_all(sessions=("T",)):
    """Preprocess every subject and save one .npz file per recording to data_processed/."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    for subject in SUBJECTS:
        for session in sessions:
            X, y = preprocess_subject(subject, session)
            np.savez_compressed(PROCESSED_DIR / f"{subject}{session}.npz", X=X, y=y,
                                classes=np.array(CLASS_NAMES))
            yield subject, session, X, y


def load_processed(subject: str, session: str = "T"):
    """Load a preprocessed recording saved by save_all()."""
    d = np.load(PROCESSED_DIR / f"{subject}{session}.npz")
    return d["X"], d["y"]


# ---------------------------------------------------------------------------
# Preprocessing options compared in notebook 04 (both sessions, official labels).
# Windows are relative to the cue; the cue is at t = 2 s of each trial.
# ---------------------------------------------------------------------------
CONFIGS = {
    "A": {"l_freq": 8.0, "h_freq": 30.0, "tmin": 0.0, "tmax": 4.0, "ems": False,
          "label": "8-30 Hz, 0-4 s after cue (Steps 2-3)"},
    "B": {"l_freq": 8.0, "h_freq": 30.0, "tmin": 0.5, "tmax": 4.5, "ems": False,
          "label": "8-30 Hz, 0.5-4.5 s after cue"},
    "C": {"l_freq": 4.0, "h_freq": 38.0, "tmin": 0.5, "tmax": 4.5, "ems": False,
          "label": "4-38 Hz, 0.5-4.5 s after cue"},
    "D": {"l_freq": None, "h_freq": None, "tmin": -0.5, "tmax": 4.0, "ems": True,
          "label": "no extra filter, -0.5-4 s, exp. moving standardization (Nzakuna et al.)"},
}


def preprocess_config(subject: str, session: str, config: str):
    """One recording -> X (288, 22, n_times) float32, y (288,) using official labels for T and E.

    Filters and exponential moving standardization run on the continuous signal before
    epoching. EMS is causal (uses only past samples), so it never looks ahead into later trials.
    """
    from braindecode.preprocessing import exponential_moving_standardize
    from data import labelled_events

    cfg = CONFIGS[config]
    raw = load_raw(subject, session).pick("eeg")
    if cfg["l_freq"] is not None:
        raw.filter(cfg["l_freq"], cfg["h_freq"], method="fir", phase="zero", verbose="ERROR")
    if cfg["ems"]:
        raw.apply_function(lambda x: exponential_moving_standardize(x * 1e6, factor_new=1e-3,
                                                                    init_block_size=1000),
                           channel_wise=False)
    events, event_id = labelled_events(raw, subject, session)
    sfreq = raw.info["sfreq"]
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=cfg["tmin"], tmax=cfg["tmax"] - 1 / sfreq,
                        baseline=None, preload=True, verbose="ERROR")
    assert len(epochs) == 288, f"{subject}{session}: {len(epochs)} epochs (some dropped?)"
    X = epochs.get_data()
    if not cfg["ems"]:
        X = X * 1e6                       # volts -> microvolts (EMS output is already unitless)
    return X.astype(np.float32), epochs.events[:, 2].astype(np.int64)


def load_config_data(config: str, subject: str, session: str):
    """Cached version of preprocess_config (saved under data_processed/config_<X>/)."""
    path = PROCESSED_DIR / f"config_{config}" / f"{subject}{session}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        X, y = preprocess_config(subject, session, config)
        np.savez_compressed(path, X=X, y=y)
    d = np.load(path)
    return d["X"], d["y"]
