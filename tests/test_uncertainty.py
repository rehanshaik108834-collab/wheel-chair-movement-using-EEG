"""Correctness tests for src/uncertainty.py (Stage B). Run: python tests/test_uncertainty.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import uncertainty as U  # noqa: E402

RNG = np.random.default_rng(0)


def sample_labels(p):
    """Draw one label per row from the categorical distribution p (so p is perfectly calibrated)."""
    u = RNG.random((len(p), 1))
    return (u > np.cumsum(p, 1)).sum(1)


def test_temperature_recovers_known_T():
    z = RNG.normal(scale=3.0, size=(20000, 4))
    y = sample_labels(U.softmax(z / 2.0))          # the true distribution is softmax(z / 2)
    T = U.fit_temperature(z, y)
    assert abs(T - 2.0) < 0.12, T
    print(f"ok  temperature scaling recovers a known temperature: T_true = 2.00, fitted {T:.3f}")


def test_ece_calibrated_vs_overconfident():
    z = RNG.normal(scale=2.0, size=(20000, 4))
    p = U.softmax(z)
    y = sample_labels(p)
    e_cal, e_over = U.ece(p, y), U.ece(U.softmax(3 * z), y)
    assert e_cal < 0.02 and e_over > 0.1, (e_cal, e_over)
    print(f"ok  ECE: calibrated probabilities {e_cal:.3f}, over-confident (logits x3) {e_over:.3f}")


def test_brier_nll_bounds():
    y = np.array([0, 1, 2, 3])
    perfect = np.eye(4)[y]
    assert U.brier(perfect, y) == 0 and U.nll(perfect, y) < 1e-9
    uniform = np.full((4, 4), 0.25)
    assert abs(U.nll(uniform, y) - np.log(4)) < 1e-12 and abs(U.brier(uniform, y) - 0.75) < 1e-12
    print("ok  Brier/NLL: perfect = 0, uniform = 0.75 / ln 4")


def test_entropy_and_mutual_information():
    assert abs(U.entropy(np.full(4, 0.25)) - np.log(4)) < 1e-12 and U.entropy(np.eye(4)[0]) < 1e-9
    p = U.softmax(RNG.normal(size=(50, 4)))
    assert np.allclose(U.mutual_information(np.stack([p, p, p])), 0, atol=1e-12)
    disagree = np.stack([np.eye(4)[np.zeros(50, int)], np.eye(4)[np.ones(50, int)]])
    assert np.allclose(U.mutual_information(disagree), np.log(2), atol=1e-6)
    print("ok  entropy: uniform = ln 4, one-hot = 0; mutual information: identical members = 0, "
          "two confident members that disagree = ln 2")


def test_risk_coverage_and_aurc():
    correct = RNG.random(1000) > 0.3                      # 30% errors
    oracle = (~correct).astype(float)                     # errors get the highest score
    cov, risk = U.risk_coverage(oracle, correct)
    assert risk[int(0.7 * 1000) - 50] == 0.0 and abs(risk[-1] - (~correct).mean()) < 1e-12
    a_oracle, a_random = U.aurc(oracle, correct), U.aurc(RNG.random(1000), correct)
    assert a_oracle < 0.06 < 0.25 < a_random < 0.35, (a_oracle, a_random)
    assert U.auroc_error(oracle, correct) == 1.0
    print(f"ok  risk-coverage: oracle score AURC {a_oracle:.3f} < random score AURC {a_random:.3f} "
          f"(~error rate 0.30); oracle AUROC = 1")


def test_thresholds():
    score = RNG.random(5000)
    correct = RNG.random(5000) > score                    # more uncertain -> more often wrong
    tau = U.threshold_for_coverage(score, 0.8)
    assert abs((score <= tau).mean() - 0.8) < 0.005
    tau_a = U.threshold_for_accuracy(score, correct, 0.85)
    acc = correct[score <= tau_a].mean()
    assert acc >= 0.85 and acc - 0.85 < 0.01, acc
    assert U.threshold_for_accuracy(score, correct, 1.01) is None
    print(f"ok  thresholds: 80% coverage target -> {(score <= tau).mean():.3f}; "
          f"accuracy target 85% -> accepted accuracy {acc:.3f} at coverage {(score <= tau_a).mean():.3f}; "
          f"unreachable target -> None")


def test_selective_metrics_counts():
    p = np.eye(4)[[0, 1, 2, 3, 2, 2]]
    y = np.array([0, 1, 2, 3, 0, 1])                      # the last two FORWARD predictions are wrong
    m = U.selective_metrics(p, y, np.array([1, 1, 1, 1, 1, 0], bool))
    assert m["coverage"] == 5 / 6 and m["accepted_accuracy"] == 0.8
    assert abs(m["false_per_100"] - 100 / 6) < 1e-9 and abs(m["false_by_command_per_100"]["FORWARD"] - 100 / 6) < 1e-9
    print("ok  safety metrics: coverage, accuracy of accepted commands and false commands per command")


def test_robust_z():
    x = np.r_[RNG.normal(size=1000), 50.0]
    z = U.robust_z(x)
    assert abs(np.median(z)) < 0.1 and z[-1] > 30
    print("ok  robust z-score: median ~0, an outlier stands out")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL TESTS PASSED")
