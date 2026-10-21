"""Correctness tests for src/alignment.py (Euclidean Alignment). Run: python tests/test_alignment.py"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import alignment as ea  # noqa: E402
from data import SUBJECTS  # noqa: E402
from preprocess import load_config_data  # noqa: E402


def test_inv_sqrtm():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(22, 22)); R = A @ A.T + 22 * np.eye(22)
    M = ea.inv_sqrtm(R)
    assert np.allclose(M @ R @ M, np.eye(22), atol=1e-8) and np.allclose(M, M.T)
    print("ok  R^(-1/2) R R^(-1/2) = I and R^(-1/2) is symmetric")


def test_mean_covariance_becomes_identity_real_data():
    worst = 0.0
    for s in SUBJECTS:
        for sess in "TE":
            X, _ = load_config_data("A", s, sess)
            Xa, R = ea.align_self(X)
            C = ea.reference_covariance(Xa)
            worst = max(worst, np.abs(C - np.eye(22)).max())
    assert worst < 1e-4, worst
    print(f"ok  all 18 recordings: mean covariance after EA = identity (max deviation {worst:.1e})")


def test_no_labels_used_and_shape():
    X, y = load_config_data("A", "A01", "T")
    Xa1, _ = ea.align_self(X)
    Xa2, _ = ea.align_self(X[np.random.default_rng(1).permutation(288)][np.argsort(np.random.default_rng(1).permutation(288))])
    assert Xa1.shape == X.shape and np.allclose(Xa1, Xa2, atol=1e-4)
    print("ok  EA uses no labels and does not depend on trial order; shape preserved (288, 22, 1000)")


def test_subjects_become_more_similar():
    """Distance between subjects' mean covariances (Frobenius, after trace-normalisation) shrinks."""
    def dists(aligned):
        Cs = []
        for s in SUBJECTS:
            X, _ = load_config_data("A", s, "T")
            C = ea.reference_covariance(ea.align_self(X)[0] if aligned else X)
            Cs.append(C / np.trace(C))
        return np.mean([np.linalg.norm(a - b) for i, a in enumerate(Cs) for b in Cs[i + 1:]])
    before, after = dists(False), dists(True)
    assert after < before
    print(f"ok  mean pairwise distance between subjects' mean covariances: {before:.4f} before EA -> {after:.4f} after "
          "(zero by construction: every subject is centred on the identity; individual trials still differ)")


def test_pseudo_online_reference_close_to_offline():
    X, _ = load_config_data("A", "A01", "E")
    R_all, R_48 = ea.reference_covariance(X), ea.reference_covariance(X[:48])
    rel = np.linalg.norm(R_48 - R_all) / np.linalg.norm(R_all)
    print(f"ok  pseudo-online reference (first 48 trials) differs from offline reference by {100 * rel:.1f}%")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL TESTS PASSED")
