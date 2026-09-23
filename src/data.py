"""Loading utilities for the BCI Competition IV-2a dataset (GDF files)."""
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Dataset" / "BCICIV_2a_gdf"
# Official labels from https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
LABEL_DIR = PROJECT_ROOT / "Dataset" / "true_labels"

SUBJECTS = [f"A0{i}" for i in range(1, 10)]
SFREQ = 250.0

# The GDF files label most channels just "EEG"; this is the documented 10-20 order.
EEG_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]

# Event codes stored in the GDF annotations.
EVENT_CODES = {
    "276": "Idle, eyes open",
    "277": "Idle, eyes closed",
    "768": "Start of a trial",
    "769": "Cue: left hand (class 1)",
    "770": "Cue: right hand (class 2)",
    "771": "Cue: feet (class 3)",
    "772": "Cue: tongue (class 4)",
    "783": "Cue: unknown (evaluation session)",
    "1023": "Rejected trial (expert-marked artifact)",
    "1072": "Eye movements (EOG calibration)",
    "32766": "Start of a new run",
}

# Motor-imagery cue code -> class name. Integer labels 0..3 are used for training.
CUE_CODES = {"769": "left_hand", "770": "right_hand", "771": "feet", "772": "tongue"}
CLASS_NAMES = list(CUE_CODES.values())


def gdf_path(subject: str, session: str = "T") -> Path:
    """Path to a subject's file; session is 'T' (training) or 'E' (evaluation)."""
    return DATA_DIR / f"{subject}{session}.gdf"


def load_raw(subject: str, session: str = "T", preload: bool = True) -> mne.io.BaseRaw:
    """Read one GDF file with proper channel names, EOG channel types and a 10-20 montage."""
    raw = mne.io.read_raw_gdf(gdf_path(subject, session), eog=EOG_CHANNELS,
                              preload=preload, verbose="ERROR")
    raw.rename_channels(dict(zip(raw.ch_names[:22], EEG_CHANNELS)))
    raw.set_montage("colin27_1020", on_missing="ignore")
    return raw


def cue_events(raw: mne.io.BaseRaw):
    """Return (events, event_id) for the four MI cues only (training sessions)."""
    present = {code: i for i, code in enumerate(CUE_CODES) if code in set(raw.annotations.description)}
    events, _ = mne.events_from_annotations(raw, event_id=present, verbose="ERROR")
    event_id = {CUE_CODES[code]: i for code, i in present.items()}
    return events, event_id


def true_labels(subject: str, session: str) -> np.ndarray:
    """Official class labels (0..3) from the competition's true_labels .mat files, in trial order."""
    from scipy.io import loadmat
    labels = loadmat(LABEL_DIR / f"{subject}{session}.mat")["classlabel"].ravel().astype(np.int64) - 1
    assert len(labels) == 288 and np.all(np.bincount(labels, minlength=4) == 72), f"{subject}{session}"
    return labels


def labelled_events(raw: mne.io.BaseRaw, subject: str, session: str):
    """(events, event_id) for all 288 cues of a T or E recording, labelled from the true-label files.

    T files contain the class in the cue code (769-772); E files only contain 783 ("unknown").
    For T files we check that the label file agrees with the cue codes trial by trial.
    """
    ann = raw.annotations
    cue_codes = set(CUE_CODES) | {"783"}
    onsets = np.sort(ann.onset[np.isin(ann.description, list(cue_codes))])
    assert len(onsets) == 288, f"{subject}{session}: found {len(onsets)} cues"
    labels = true_labels(subject, session)
    if session == "T":
        events_gdf, _ = cue_events(raw)
        assert np.array_equal(events_gdf[:, 2], labels), f"{subject}T: label file disagrees with cue codes"
    samples = raw.time_as_index(onsets, use_rounding=True)
    events = np.column_stack([samples, np.zeros(288, dtype=int), labels])
    return events, {name: i for i, name in enumerate(CLASS_NAMES)}


def trial_rejected_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """Boolean mask over trials (in '768' order) marking trials flagged with code 1023."""
    ann = raw.annotations
    starts = ann.onset[ann.description == "768"]
    rejected = ann.onset[ann.description == "1023"]
    return np.isin(np.round(starts, 3), np.round(rejected, 3))
