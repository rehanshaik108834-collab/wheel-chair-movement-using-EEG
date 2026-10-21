"""Euclidean Alignment (He & Wu, IEEE TBME 2020; evaluated with deep learning by Junqueira et al., 2024).

For each group of trials (one subject's session):
    R = (1/n) * sum_i X_i X_i^T / T          reference = mean spatial covariance of the trials
    X_i_aligned = R^(-1/2) X_i                after which the group's mean covariance is the identity
No labels are used, so the new user's reference can be computed from unlabelled EEG
(all of it = "offline", or only a first calibration block = "pseudo-online").
"""
import numpy as np


def trial_covariances(X):
    """X: (n, ch, T) -> (n, ch, ch) spatial covariance of each trial (normalised by T)."""
    X = np.asarray(X, dtype=np.float64)
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]


def reference_covariance(X):
    """Arithmetic mean of the trials' covariance matrices (the EA reference R)."""
    return trial_covariances(X).mean(axis=0)


def inv_sqrtm(R, eps=1e-10):
    """R^(-1/2) for a symmetric positive-definite matrix, via eigen-decomposition."""
    w, V = np.linalg.eigh((R + R.T) / 2)
    w = np.maximum(w, eps * w.max())
    return (V * (1.0 / np.sqrt(w))) @ V.T


def align(X, R):
    """Apply R^(-1/2) to every trial of X (n, ch, T)."""
    return np.einsum("cd,ndt->nct", inv_sqrtm(R), np.asarray(X, dtype=np.float64)).astype(np.float32)


def align_self(X):
    """Align a group of trials with its own reference (returns aligned X and R)."""
    R = reference_covariance(X)
    return align(X, R), R
