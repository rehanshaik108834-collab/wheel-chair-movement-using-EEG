"""Correctness tests for src/mentor_pipeline.py. Run: python tests/test_mentor_pipeline.py"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mentor_pipeline as mp  # noqa: E402
from preprocess import load_config_data  # noqa: E402


def test_wavelet_energy_conserves_energy():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(5, 20, 22, 50))
    E = mp.wavelet_energy(S)
    assert E.shape == (5, 20, 44)
    total = (S ** 2).sum(-1)
    assert np.allclose(E[..., :22] + E[..., 22:], total, rtol=1e-10), "A+D energy must equal signal energy"
    print("ok  wavelet energy: shape (5,20,44); energy(A)+energy(D) == energy(signal)")


def test_segmentation_keeps_trials_separate():
    X = np.arange(2 * 3 * 1000, dtype=float).reshape(2, 3, 1000)
    S = mp.segment(X, 0.2)
    assert S.shape == (2, 20, 3, 50)
    assert np.array_equal(S[1, 0, 2], X[1, 2, :50]) and np.array_equal(S[1, 19, 0], X[1, 0, 950:1000])
    print("ok  segmentation: 4 s -> 20 x 0.2 s segments, each from its own trial and channel")


def test_batched_omp_matches_sklearn():
    """Our batched OMP must give the same sparse code as scikit-learn's reference OMP."""
    from sklearn.linear_model import orthogonal_mp
    rng = np.random.default_rng(5)
    clf = mp.SparseRepresentationClassifier(max_atoms=8).fit(rng.normal(size=(300, 44)), np.repeat(np.arange(4), 75))
    Y = mp._l2norm(rng.normal(size=(20, 44)))
    idx, coef, n_used = clf.sparse_codes(Y)
    worst = 0.0
    for i in range(len(Y)):
        ref = orthogonal_mp(clf.D.T, Y[i], n_nonzero_coefs=int(n_used[i]))
        ours = np.zeros(300); ours[idx[i, :n_used[i]]] = coef[i, :n_used[i]]
        worst = max(worst, np.abs(ref - ours).max())
    assert worst < 1e-6, worst
    print(f"ok  batched OMP == scikit-learn orthogonal_mp (max coefficient difference {worst:.1e})")


def test_omp_recovers_sparse_code():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(400, 44))
    labels = np.repeat(np.arange(4), 100)
    clf = mp.SparseRepresentationClassifier().fit(F, labels)
    D = clf.D
    y = 1.0 * D[250] - 0.6 * D[260]                           # two atoms of class 2
    s = clf.scores(y[None])
    for r in mp.CLASS_RULES:
        assert s[r][0].argmax() == 2, (r, s[r][0])
    print("ok  OMP: test vector built from 2 class-2 atoms is assigned to class 2 by R1-R4")


def test_classifier_on_separable_data():
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(4, 44)) * 3
    Ftr = np.concatenate([centers[c] + rng.normal(size=(200, 44)) for c in range(4)])
    ytr = np.repeat(np.arange(4), 200)
    Fte = np.concatenate([centers[c] + rng.normal(size=(50, 44)) for c in range(4)])
    yte = np.repeat(np.arange(4), 50)
    clf = mp.SparseRepresentationClassifier().fit(Ftr, ytr)
    t0 = time.time()
    s = clf.scores(Fte)
    acc = {r: (s[r].argmax(1) == yte).mean() for r in mp.CLASS_RULES}
    assert acc["R4"] > 0.9, acc
    print(f"ok  classifier on separable toy data: accuracy {acc}  ({(time.time() - t0) / 200 * 1000:.1f} ms/vector)")


def test_real_data_loading():
    Xb, y = mp.load_band_epochs("A01", "T", [None, (8, 30)], channels="motor")
    _, yA = load_config_data("A", "A01", "T")
    assert np.array_equal(y, yA), "labels must match the main pipeline"
    assert Xb[None].shape == (288, 14, 1000) and Xb[(8, 30)].shape == (288, 14, 1000)
    # unfiltered epochs must equal the raw signal: compare with our config A only through labels/shape,
    # and check the filter actually removed low frequencies
    def lowfreq_power(X):
        f = np.fft.rfftfreq(X.shape[-1], 1 / 250)
        P = np.abs(np.fft.rfft(X, axis=-1)) ** 2
        return P[..., f < 4].mean() / P[..., (f > 10) & (f < 25)].mean()
    r_raw, r_filt = lowfreq_power(Xb[None]), lowfreq_power(Xb[(8, 30)])
    assert r_filt < 0.05 * r_raw, (r_raw, r_filt)
    print(f"ok  real data: (288,14,1000), labels match main pipeline; <4 Hz power ratio raw {r_raw:.2f} -> filtered {r_filt:.4f}")


def test_csp_fit_on_train_only_shapes():
    Xb, y = mp.load_band_epochs("A01", "T", [(8, 30)])
    tr = np.arange(288) < 240
    csps = mp.fit_csp({b: X[tr] for b, X in Xb.items()}, y[tr])
    Z = mp.apply_csp(csps, {b: X[~tr] for b, X in Xb.items()})
    assert Z[(8, 30)].shape == (48, 10, 1000)
    print("ok  CSP: fitted on 240 training trials, 10 components applied to 48 held-out trials")


def test_speed_realistic():
    rng = np.random.default_rng(3)
    clf = mp.SparseRepresentationClassifier().fit(rng.random((4800, 44)), np.repeat(np.arange(4), 1200))
    t0 = time.time()
    clf.scores(rng.random((960, 44)))
    per = (time.time() - t0) / 960
    print(f"ok  speed: dictionary 4800 atoms x 44 dims -> {per * 1000:.1f} ms per test segment "
          f"(~{per * 960:.0f} s per fold of 960 segments)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL TESTS PASSED")
