"""Calibration and uncertainty-aware command rejection (Stage B).

Probabilities are (n_trials, n_classes) arrays; a rejection score is "higher = more uncertain",
and a trial is ACCEPTED (becomes a wheelchair command) when score <= tau.

  temperature scaling   p = softmax(z / T), T fitted by minimising NLL on labelled calibration data
  calibration metrics   NLL, Brier score, ECE (top-label, equal-width bins), reliability bins
  uncertainty scores    predictive entropy, 1 - max probability, mutual information of an ensemble
  rejection             risk-coverage curve, AURC, misclassification AUROC, threshold selection,
                        wheelchair safety metrics (coverage, accuracy of accepted, false commands)
"""
import numpy as np
from scipy.optimize import minimize_scalar

COMMANDS = ["LEFT", "RIGHT", "FORWARD", "STOP"]   # left hand, right hand, feet, tongue


# --------------------------------------------------------------------------- probabilities
def softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def nll(p, y):
    """Mean negative log-likelihood of the true class."""
    p = np.asarray(p)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def brier(p, y):
    """Multi-class Brier score: mean over trials of sum_k (p_k - onehot_k)^2 (0 = perfect, 2 = worst)."""
    p = np.asarray(p)
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


def reliability(p, y, n_bins=15):
    """Per confidence bin: (mean confidence, accuracy, count); bins are equal-width on (0, 1]."""
    p = np.asarray(p)
    conf, correct = p.max(1), p.argmax(1) == np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((float(conf[m].mean()) if m.any() else np.nan,
                     float(correct[m].mean()) if m.any() else np.nan, int(m.sum())))
    return rows


def ece(p, y, n_bins=15):
    """Expected calibration error: sum over bins of |accuracy - confidence| weighted by bin size."""
    rows = reliability(p, y, n_bins)
    n = sum(r[2] for r in rows)
    return float(sum(r[2] / n * abs(r[1] - r[0]) for r in rows if r[2]))


def fit_temperature(logits, y, bounds=(0.05, 20.0)):
    """Single temperature T minimising the NLL of softmax(logits / T) on labelled calibration data."""
    z = np.asarray(logits, dtype=np.float64)
    res = minimize_scalar(lambda lt: nll(softmax(z / np.exp(lt)), y), bounds=np.log(bounds),
                          method="bounded", options={"xatol": 1e-5})
    return float(np.exp(res.x))


# --------------------------------------------------------------------------- uncertainty scores
def entropy(p):
    """Predictive entropy in nats (0 = certain, ln C = uniform)."""
    p = np.asarray(p)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def one_minus_maxprob(p):
    return 1.0 - np.asarray(p).max(-1)


def mutual_information(member_probs):
    """Disagreement of an ensemble: H(mean p) - mean H(p_m); member_probs (M, n, C). 0 if all agree."""
    P = np.asarray(member_probs)
    return entropy(P.mean(0)) - entropy(P).mean(0)


# --------------------------------------------------------------------------- rejection
def risk_coverage(score, correct):
    """Accept the k most certain trials (lowest score), k = 1..n -> (coverage k/n, selective risk)."""
    order = np.argsort(np.asarray(score), kind="stable")
    err = ~np.asarray(correct, dtype=bool)[order]
    k = np.arange(1, len(order) + 1)
    return k / len(order), np.cumsum(err) / k


def aurc(score, correct):
    """Area under the risk-coverage curve (lower = better ranking of errors)."""
    return float(risk_coverage(score, correct)[1].mean())


def auroc_error(score, correct):
    """How well the score separates wrong from correct predictions (0.5 = chance, 1 = perfect)."""
    from sklearn.metrics import roc_auc_score
    wrong = ~np.asarray(correct, dtype=bool)
    return float(roc_auc_score(wrong, score)) if 0 < wrong.sum() < len(wrong) else float("nan")


def threshold_for_coverage(score_cal, coverage):
    """tau such that a fraction `coverage` of the calibration trials has score <= tau."""
    return float(np.quantile(np.asarray(score_cal), coverage))


def threshold_for_accuracy(score_cal, correct_cal, target, min_accepted=20):
    """Largest tau (= highest coverage) whose accepted calibration trials are >= target correct.

    Returns None when the target is not reachable with at least `min_accepted` accepted trials.
    """
    order = np.argsort(np.asarray(score_cal), kind="stable")
    s = np.asarray(score_cal)[order]
    acc = np.cumsum(np.asarray(correct_cal, dtype=bool)[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(acc >= target)
    ok = ok[ok >= min_accepted - 1]
    return float(s[ok.max()]) if len(ok) else None


def selective_metrics(p, y, accept):
    """Wheelchair safety metrics for a boolean accept mask (rejected trials issue no command)."""
    p, y, accept = np.asarray(p), np.asarray(y), np.asarray(accept, dtype=bool)
    pred = p.argmax(1)
    wrong_accepted = accept & (pred != y)
    return {"coverage": float(accept.mean()),
            "accepted_accuracy": float((pred == y)[accept].mean()) if accept.any() else float("nan"),
            "false_per_100": float(100 * wrong_accepted.mean()),
            "false_by_command_per_100": {c: float(100 * (wrong_accepted & (pred == k)).mean())
                                         for k, c in enumerate(COMMANDS)},
            "n_trials": int(len(y))}


def robust_z(x, ref=None):
    """(x - median) / (1.4826 * MAD), statistics from `ref` (default: x itself)."""
    ref = np.asarray(x if ref is None else ref, dtype=np.float64)
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-12
    return (np.asarray(x, dtype=np.float64) - med) / mad
