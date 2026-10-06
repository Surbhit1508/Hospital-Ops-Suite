"""
Reusable bootstrap resampling utilities for honest uncertainty quantification.

Ported (same approach) from the RAGBench project: instead of eyeballing a
single point estimate, we resample the evaluation set with replacement many
times, recompute the metric each time, and report a confidence interval.
For paired comparisons (e.g. "is model A better than model B?") we resample
the SAME indices for both so the comparison is apples-to-apples.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, metric_fn, n_resamples: int = 10_000, ci: float = 0.95, seed: int = 42):
    """Bootstrap confidence interval for a metric computed over `values`.

    `metric_fn` receives a resampled 1-D array and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        estimates[i] = metric_fn(sample)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 - (1 - ci) / 2) * 100
    return {
        "point_estimate": float(metric_fn(values)),
        "ci_low": float(np.percentile(estimates, lo_pct)),
        "ci_high": float(np.percentile(estimates, hi_pct)),
        "n_resamples": n_resamples,
    }


def bootstrap_ci_indexed(n: int, metric_of_indices, n_resamples: int = 10_000, ci: float = 0.95, seed: int = 42):
    """Bootstrap CI for metrics that need more than one aligned array per sample
    (e.g. multi-class F1, which needs both y_true[idx] and y_pred[idx]).

    `metric_of_indices` receives an array of resampled row indices (0..n-1,
    with replacement) and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates[i] = metric_of_indices(idx)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 - (1 - ci) / 2) * 100
    return {
        "point_estimate": float(metric_of_indices(all_idx)),
        "ci_low": float(np.percentile(estimates, lo_pct)),
        "ci_high": float(np.percentile(estimates, hi_pct)),
        "n_resamples": n_resamples,
    }


def paired_bootstrap_test(values_a: np.ndarray, values_b: np.ndarray, metric_fn, n_resamples: int = 10_000, seed: int = 42):
    """Two-sided paired bootstrap significance test for metric_fn(a) vs metric_fn(b).

    Returns the observed difference (a - b) and a p-value: the fraction of
    resampled differences at least as extreme as zero would be under the null
    that there's no true difference, estimated via the shift method.
    """
    rng = np.random.default_rng(seed)
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    n = len(values_a)
    assert n == len(values_b), "paired arrays must be the same length"

    observed_diff = metric_fn(values_a) - metric_fn(values_b)

    # Null-shifted samples: remove the observed effect, then resample, to test
    # "how often would we see a difference this extreme if there were none?"
    diffs = values_a - values_b
    shifted = diffs - diffs.mean()

    resampled_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = shifted[rng.integers(0, n, size=n)]
        resampled_diffs[i] = sample.mean()

    p_value = float(np.mean(np.abs(resampled_diffs) >= np.abs(diffs.mean())))

    return {
        "observed_diff": float(observed_diff),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "n_resamples": n_resamples,
    }
