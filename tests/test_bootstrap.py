import numpy as np

from src.stats.bootstrap import bootstrap_ci, bootstrap_ci_indexed, paired_bootstrap_test


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=10, scale=2, size=200)
    result = bootstrap_ci(values, np.mean, n_resamples=500)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    assert abs(result["point_estimate"] - 10) < 1.0


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)
    small = rng.normal(size=30)
    large = rng.normal(size=3000)
    small_ci = bootstrap_ci(small, np.mean, n_resamples=500)
    large_ci = bootstrap_ci(large, np.mean, n_resamples=500)
    small_width = small_ci["ci_high"] - small_ci["ci_low"]
    large_width = large_ci["ci_high"] - large_ci["ci_low"]
    assert large_width < small_width


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(2)
    a = np.ones(500)  # always correct
    b = rng.binomial(1, 0.5, size=500).astype(float)  # coin flip
    result = paired_bootstrap_test(a, b, np.mean, n_resamples=1000)
    assert result["observed_diff"] > 0
    assert result["p_value"] < 0.05
    assert result["significant_at_0.05"] is True


def test_paired_bootstrap_no_difference_is_not_significant():
    rng = np.random.default_rng(3)
    a = rng.binomial(1, 0.5, size=500).astype(float)
    b = a.copy()  # identical -> zero difference, definitely not significant
    result = paired_bootstrap_test(a, b, np.mean, n_resamples=1000)
    assert result["observed_diff"] == 0
    assert result["significant_at_0.05"] is False


def test_bootstrap_ci_indexed_matches_manual_computation():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

    def accuracy_of_indices(idx):
        return float(np.mean(y_true[idx] == y_pred[idx]))

    result = bootstrap_ci_indexed(len(y_true), accuracy_of_indices, n_resamples=300)
    manual_accuracy = float(np.mean(y_true == y_pred))
    assert result["point_estimate"] == manual_accuracy
    assert result["ci_low"] <= result["ci_high"]
