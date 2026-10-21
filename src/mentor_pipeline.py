"""Mentor's pipeline (Sreeja & Samanta, Neurocomputing 368 (2019) 133-145), adapted to BCI IV-2a.

Steps from the paper:
  1. (optional) channel selection around C3/C4
  2. filtering: Case 1 band-pass + CSP (10 filters), Case 2 band-pass only, Case 3 no filtering;
     band-pass with 1, 2, 3, 6 or 9 sub-bands (Chebyshev type II, order 6)
  3. segmentation of the 4 s imagery period into short segments (0.2 s or 0.5 s)
  4. wavelet energy feature: coif1, level-1 DWT, energy of approximation + detail per channel
  5. per-subject dictionary of training-segment feature vectors, grouped by class
  6. sparse representation with Orthogonal Matching Pursuit (OMP) and rules R1-R4

Additions needed for our protocols (not in the paper):
  - the paper classifies segments; a wheelchair command is issued per trial, so segment scores
    are combined per trial (summed residuals / scores over the trial's segments)
  - dictionary atoms and test vectors are L2-normalised (standard for sparse-representation
    classification; the paper does not state its normalisation)
"""
import numpy as np
import pywt
from scipy.signal import sosfiltfilt

from data import EEG_CHANNELS, labelled_events, load_raw
from preprocess import cheby2_sos

SFREQ = 250.0
TMIN, TMAX = 0.0, 4.0                 # imagery period after the cue (paper: 3-7 s with cue at 3 s)

# Channels surrounding C3 and C4 (paper: 18 of 60 electrodes neighbouring C3/C4)
MOTOR_CHANNELS = ["FC3", "FC1", "C5", "C3", "C1", "CP3", "CP1",
                  "FC2", "FC4", "C2", "C4", "C6", "CP2", "CP4"]
MOTOR_IDX = np.array([EEG_CHANNELS.index(c) for c in MOTOR_CHANNELS])

# Sub-band sets from the paper (Table 2/3)
BAND_SETS = {
    1: [(8, 30)],
    2: [(8, 13), (13, 30)],
    3: [(8, 12), (12, 16), (16, 30)],
    6: [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)],
    9: [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)],
}
CLASS_RULES = ("R1", "R2", "R3", "R4")


# --------------------------------------------------------------------------- signals
def load_band_epochs(subject, session, bands, channels="all"):
    """Epochs (288, n_ch, 1000) in microvolts for each band (None = no filtering), plus labels.

    Filtering is applied to the continuous recording (zero-phase), then 0-4 s after each cue
    is cut, exactly as in our main pipeline. Returns ({band: X}, y).
    """
    raw = load_raw(subject, session).pick("eeg")
    events, _ = labelled_events(raw, subject, session)
    data = raw.get_data() * 1e6                                   # (22, n_times) in microvolts
    if channels == "motor":
        data = data[MOTOR_IDX]
    start = events[:, 0] + int(round(TMIN * SFREQ))
    n = int(round((TMAX - TMIN) * SFREQ))
    out = {}
    for band in bands:
        sig = data if band is None else sosfiltfilt(cheby2_sos(*band, SFREQ), data, axis=-1)
        out[band] = np.stack([sig[:, s:s + n] for s in start]).astype(np.float32)
    return out, events[:, 2].astype(np.int64)


def segment(X, seg_seconds):
    """(trials, ch, T) -> (trials, n_seg, ch, seg_len); the 4 s period is split without overlap."""
    seg_len = int(round(seg_seconds * SFREQ))
    n_seg = X.shape[-1] // seg_len
    Xs = X[..., :n_seg * seg_len].reshape(X.shape[0], X.shape[1], n_seg, seg_len)
    return Xs.transpose(0, 2, 1, 3)


# --------------------------------------------------------------------------- features
def wavelet_energy(S):
    """Wavelet energy (coif1, level 1) of each segment and channel.

    S: (..., ch, seg_len) -> (..., 2*ch) = [energy of approximation coeffs | energy of detail coeffs].
    'periodization' mode makes the transform orthogonal, so energy(A) + energy(D) = energy(signal).
    """
    cA, cD = pywt.dwt(S, "coif1", mode="periodization", axis=-1)
    return np.concatenate([(cA ** 2).sum(-1), (cD ** 2).sum(-1)], axis=-1)


def wavelet_entropy(S):
    """Shannon entropy of the normalised squared coefficients of A and D, per channel -> (..., 2*ch)."""
    feats = []
    for c in pywt.dwt(S, "coif1", mode="periodization", axis=-1):
        p = c ** 2 / np.maximum((c ** 2).sum(-1, keepdims=True), 1e-12)
        feats.append(-(p * np.log(np.maximum(p, 1e-12))).sum(-1))
    return np.concatenate(feats, axis=-1)


def features(X_by_band, seg_seconds, feature="energy"):
    """{band: (trials, ch, T)} -> (trials, n_seg, n_features), bands concatenated (as in the paper)."""
    parts = []
    for band in sorted(X_by_band, key=lambda b: (-1, -1) if b is None else b):
        S = segment(X_by_band[band], seg_seconds)
        f = wavelet_energy(S)
        if feature == "energy+entropy":
            f = np.concatenate([f, wavelet_entropy(S)], axis=-1)
        parts.append(f)
    return np.concatenate(parts, axis=-1).astype(np.float64)


def fit_csp(X_by_band, y, n_components=10):
    """One multi-class CSP (MNE) per band, fitted on TRAINING trials only."""
    from mne.decoding import CSP
    csps = {}
    for band, X in X_by_band.items():
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=None,
                  transform_into="csp_space", norm_trace=False)
        csps[band] = csp.fit(X.astype(np.float64), y)
    return csps


def apply_csp(csps, X_by_band):
    return {band: csps[band].transform(X.astype(np.float64)).astype(np.float32)
            for band, X in X_by_band.items()}


# --------------------------------------------------------------------------- classifier
def _l2norm(F):
    return F / np.maximum(np.linalg.norm(F, axis=-1, keepdims=True), 1e-12)


class SparseRepresentationClassifier:
    """Dictionary of training segments + OMP + class rules R1-R4 (Sreeja & Samanta, 2019).

    R1: largest L2 norm of the class's coefficients      (argmax)
    R2: most non-zero coefficients in the class          (argmax)
    R3: largest variance of the class's coefficients     (argmax)
    R4: smallest class residual ||y - D_c a_c||          (argmin)
    OMP stops when ||r||^2 <= var(y) (paper: ||r|| <= ||eps||, eps ~ variance of y) or after
    max_atoms atoms.
    """

    def __init__(self, n_classes=4, max_atoms=64, batch=2048):
        self.n_classes = n_classes
        self.max_atoms = max_atoms
        self.batch = batch

    def fit(self, F, y):
        self.D = _l2norm(np.asarray(F, dtype=np.float64))              # (n_atoms, m), unit-norm rows
        self.atom_class = np.asarray(y)
        self.n_per_class = np.bincount(self.atom_class, minlength=self.n_classes)
        return self

    def sparse_codes(self, Y):
        """Batched OMP. Y: (N, m) unit-norm rows -> (indices (N, K), coefs (N, K), n_used (N,)).

        Same algorithm as standard OMP, run for all test vectors at once: pick the atom most
        correlated with the residual, re-fit the coefficients of all chosen atoms by least squares,
        stop a vector when ||r||^2 <= var(y) or when max_atoms (or the dimension m) is reached.
        """
        N, m = Y.shape
        K = min(self.max_atoms, m)
        tol = np.maximum(Y.var(axis=1), 1e-10)
        idx = np.zeros((N, K), dtype=np.int64)
        coef = np.zeros((N, K))
        n_used = np.zeros(N, dtype=np.int64)
        active = np.ones(N, dtype=bool)
        R = Y.copy()
        rows = np.arange(N)
        for t in range(K):
            a = np.flatnonzero(active)
            if len(a) == 0:
                break
            corr = np.abs(R[a] @ self.D.T)                                # (n_active, n_atoms)
            if t:
                corr[np.arange(len(a))[:, None], idx[a, :t]] = -1.0       # never pick an atom twice
            idx[a, t] = corr.argmax(1)
            Dsel = self.D[idx[a, :t + 1]]                                  # (n_active, t+1, m)
            A = np.einsum("nkm,nlm->nkl", Dsel, Dsel) + 1e-10 * np.eye(t + 1)
            b = np.einsum("nkm,nm->nk", Dsel, Y[a])
            x = np.linalg.solve(A, b[..., None])[..., 0]
            coef[a, :t + 1] = x
            R[a] = Y[a] - np.einsum("nk,nkm->nm", x, Dsel)
            n_used[a] = t + 1
            done = (R[a] ** 2).sum(1) <= tol[a]
            active[a[done]] = False
        coef[np.arange(K)[None, :] >= n_used[:, None]] = 0.0
        return idx, coef, n_used

    def scores(self, F):
        """F: (n, m) -> dict rule -> (n, n_classes) scores where HIGHER means more likely."""
        Y = _l2norm(np.asarray(F, dtype=np.float64))
        out = {r: np.zeros((len(Y), self.n_classes)) for r in CLASS_RULES}
        for s in range(0, len(Y), self.batch):
            Yb = Y[s:s + self.batch]
            idx, coef, _ = self.sparse_codes(Yb)
            cls = self.atom_class[idx]                                     # (n, K) class of each chosen atom
            for c in range(self.n_classes):
                xc = np.where(cls == c, coef, 0.0)                         # class-c coefficients only
                n_c = self.n_per_class[c]
                out["R1"][s:s + len(Yb), c] = np.linalg.norm(xc, axis=1)
                out["R2"][s:s + len(Yb), c] = np.count_nonzero(xc, axis=1)
                mean = xc.sum(1) / n_c                                     # variance over ALL class atoms
                out["R3"][s:s + len(Yb), c] = (xc ** 2).sum(1) / n_c - mean ** 2
                recon = np.einsum("nk,nkm->nm", xc, self.D[idx])
                out["R4"][s:s + len(Yb), c] = -np.linalg.norm(Yb - recon, axis=1)   # higher = better
        return out


def trial_scores(seg_scores, n_trials, n_seg):
    """Combine segment scores into trial scores by summing over each trial's segments."""
    return {r: s.reshape(n_trials, n_seg, -1).sum(1) for r, s in seg_scores.items()}
