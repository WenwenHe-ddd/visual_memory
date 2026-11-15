"""Transient within-master ridge inverse-covariance scoring."""
import numpy as np


def fit(captions):
    x = np.asarray(captions, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected a finite, nonempty matrix of caption vectors')
    centered = x - x.mean(axis=0)
    if len(x) == 1:
        return np.empty((0, x.shape[1])), np.empty(0)
    _, sigma, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(len(x) - 1, x.shape[1])
    values = sigma[:count] ** 2 / (len(x) - 1)
    # Numerically zero variance directions contribute zero to the formula.
    tolerance = max(centered.shape) * np.finfo(float).eps * sigma[0]
    values[sigma[:count] <= tolerance] = 0
    return vt[:count], values


def score(query, captions, components, eigenvalues, ridge):
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be finite and strictly positive')
    x, q = np.asarray(captions), np.asarray(query)
    raw = x @ q
    weights = eigenvalues / (eigenvalues + ridge)
    return raw - (x @ components.T) @ (weights * (components @ q))


def standardize(scores):
    """Population z-score; singleton retains raw score; equal scores map to zero."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or not len(scores):
        raise ValueError('Expected finite, nonempty scores')
    if len(scores) == 1:
        return scores.copy(), 'singleton_raw'
    residual = scores - scores.mean()
    std = float(np.sqrt(np.mean(residual ** 2)))
    tolerance = 32 * np.finfo(float).eps * max(1., float(np.abs(scores).max()))
    if std <= tolerance:
        return np.zeros_like(scores), 'zero_variance'
    if len(scores) == 2:
        # Exact population z-scores avoid spurious floating-point tie breaks.
        return np.array([1., -1.]) * np.sign(scores[0] - scores[1]), 'two_leaf_exact'
    return residual / std, 'zscore'
