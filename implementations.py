"""
Core functions for the project
"""

from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np


"""Utility helpers (internal)."""

def _ensure_1d_vector(w: np.ndarray) -> np.ndarray:
    """Ensure weight vector is 1D shaped (D,)."""
    if w.ndim == 2 and (w.shape[1] == 1 or w.shape[0] == 1):
        return w.reshape(-1)
    if w.ndim != 1:
        return w.reshape(-1)
    return w


def _compute_mse_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """MSE with factor 1/(2N)."""
    e = y - tx @ w
    n = y.shape[0]
    return float((e @ e) / (2.0 * n))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function defined piecewise (to avoid underflow/overflow)."""
    out = np.empty_like(z, dtype=np.float64)
    positive_mask = z >= 0
    negative_mask = ~positive_mask
    out[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
    exp_z = np.exp(z[negative_mask])
    out[negative_mask] = exp_z / (1.0 + exp_z)
    return out


def _logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """Data loss for logistic regression: sum of negative log-likelihood."""
    z = tx @ w
    # Use softplus for numerical stability: log(1 + exp(z)).
    abs_z = np.abs(z)
    softplus = np.maximum(0.0, z) + np.log1p(np.exp(-abs_z))
    nll = np.sum(softplus - y * z)
    return float(nll)


def _logistic_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of the logistic data loss (unregularized)."""
    p = _sigmoid(tx @ w)
    return tx.T @ (p - y)


def _minibatch_iterator(
    y: np.ndarray, tx: np.ndarray, batch_size: int = 1
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """ Initialize mini-batches of size `batch_size` (default 1 for the project)."""
    n = y.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield y[start:end], tx[start:end]

"""Table 1: algorithms implemented."""

def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """Linear regression with full-batch gradient descent. Returns (w, MSE)."""
    w = _ensure_1d_vector(np.asarray(initial_w, dtype=np.float64).copy())
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    tx = np.asarray(tx, dtype=np.float64)

    n = y.shape[0]
    for _ in range(max_iters):
        e = y - tx @ w
        grad = -(tx.T @ e) / n
        w -= gamma * grad

    loss = _compute_mse_loss(y, tx, w)
    return w, loss

def mean_squared_error_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """Linear regression with SGD. Uses batch size 1 as required. Returns (w, MSE)."""
    w = _ensure_1d_vector(np.asarray(initial_w, dtype=np.float64).copy())
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    tx = np.asarray(tx, dtype=np.float64)

    n = y.shape[0]

    steps_done = 0
    while steps_done < max_iters:
        for y_b, tx_b in _minibatch_iterator(y, tx):
            # For batch-size 1, shapes are (1,) and (1,D)
            e_b = y_b - tx_b @ w
            grad = -(tx_b.T @ e_b)  # already correct scale for size 1
            w -= gamma * grad
            steps_done += 1
            if steps_done >= max_iters:
                break

    loss = _compute_mse_loss(y, tx, w)
    return w, loss

def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least squares via normal equations. Falls back to pinv if matrix is singular."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    tx = np.asarray(tx, dtype=np.float64)

    xtx = tx.T @ tx
    xty = tx.T @ y
    try:
        w = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(xtx) @ xty

    loss = _compute_mse_loss(y, tx, w)
    return _ensure_1d_vector(w), loss

def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> Tuple[np.ndarray, float]:
    """Ridge regression using normal equations"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    tx = np.asarray(tx, dtype=np.float64)

    n, d = tx.shape
    xtx = tx.T @ tx
    a = xtx + (2.0 * n * lambda_) * np.eye(d)
    b = tx.T @ y

    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b

    loss = _compute_mse_loss(y, tx, w)
    return _ensure_1d_vector(w), loss

def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """Logistic regression with gradient descent. Returns (w, sum NLL)."""
    w = _ensure_1d_vector(np.asarray(initial_w, dtype=np.float64).copy())
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    tx = np.asarray(tx, dtype=np.float64)

    for _ in range(max_iters):
        grad = _logistic_gradient(y, tx, w)
        w -= gamma * grad

    loss = _logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """Regularized logistic regression (L2). Return only data loss; grad adds 2λw."""
    w = _ensure_1d_vector(np.asarray(initial_w, dtype=np.float64).copy())
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    tx = np.asarray(tx, dtype=np.float64)

    for _ in range(max_iters):
        grad = _logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w -= gamma * grad

    loss = _logistic_loss(y, tx, w)
    return w, loss
