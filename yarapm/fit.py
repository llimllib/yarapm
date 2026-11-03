"""Small fitting helpers shared by the models."""

import numpy as np


def decay(ages: np.ndarray, half_life: float) -> np.ndarray:
    """Weights for games `ages` days old"""
    return 0.5 ** (np.asarray(ages, dtype=float) / half_life)


def weighted_ridge(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, penalty: np.ndarray
) -> np.ndarray:
    """
    The coefficients minimizing sum(w * (y - X @ b)**2) + sum(penalty * b**2).
    A penalty of 0 leaves a column (an intercept, say) unshrunk, though
    every column gets a tiny one, so a column of zeros (no playoff games in
    the data, say) gets a coefficient of 0
    """
    Xw = X * w[:, None]
    A = X.T @ Xw + np.diag(np.asarray(penalty, dtype=float) + 1e-9)
    return np.linalg.solve(A, Xw.T @ y)


def index(ids: list[str]) -> dict[str, int]:
    """A column for each distinct id, in sorted order"""
    return {t: i for i, t in enumerate(sorted(set(ids)))}
