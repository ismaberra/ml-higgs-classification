import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from implementations import (
    logistic_regression,
    reg_logistic_regression,
    _logistic_gradient,
    _logistic_loss,
)
from new_implementations import (
    RandomForestClassifier,
    logistic_regression_weighted,
    reg_logistic_regression_weighted,
)
from helpers import create_csv_submission, load_csv_data


def load_preprocessed(data_dir):
    """Load preprocessed matrices (no IDs) from a directory.

    Expects `x_train.csv`, `y_train.csv`, `x_test.csv` in `data_dir`.
    Returns `X, y, Xtest`.
    """
    x_train = np.loadtxt(
        os.path.join(data_dir, "x_train.csv"), delimiter=",", dtype=np.float64
    )
    y_train = np.loadtxt(
        os.path.join(data_dir, "y_train.csv"), delimiter=",", dtype=np.float64
    )
    x_test = np.loadtxt(
        os.path.join(data_dir, "x_test.csv"), delimiter=",", dtype=np.float64
    )
    if isinstance(y_train, np.ndarray) and y_train.ndim > 1:
        y_train = y_train.reshape(-1)
    return x_train, y_train, x_test


def to01(y):
    """Map labels from {-1, 1} to {0, 1}."""
    # map {-1,1} -> {0,1}
    return (y + 1) / 2


def to_sign(yhat_prob, threshold=0.5):
    """Convert probabilities to {-1, 1} labels using a threshold.

    Args:
        yhat_prob: Probability scores in [0, 1].
        threshold: Cutoff for the positive class.

    Returns:
        Numpy array of predictions in {-1, 1}.
    """
    # map prob-> {-1,1}
    return np.where(yhat_prob >= threshold, 1, -1)


def sigmoid(z):
    """Numerically stable sigmoid applied element-wise."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    mask = z >= 0
    out[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    ez = np.exp(z[~mask])
    out[~mask] = ez / (1.0 + ez)
    return out


def predict_scores(X, model_or_weights):
    """Predict positive-class probabilities for various models.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        model_or_weights: Either weight vector (for logistic models) or trained model object.

    Returns:
        Array of probabilities in [0, 1].
    """
    # Check if it's a weight vector (logistic models)
    if isinstance(model_or_weights, np.ndarray):
        # Logistic family models
        scores = X @ model_or_weights
        return sigmoid(scores)

    # Tree-based models
    elif hasattr(model_or_weights, "predict_proba"):
        proba = model_or_weights.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]  # Return probability of positive class
        else:
            # For binary classification, assume second column is positive class
            return proba[:, -1]

    # Fallback: use predict and convert to probabilities
    else:
        predictions = model_or_weights.predict(X)
        # Convert {0, 1} predictions to probabilities
        return predictions.astype(float)


def compute_metrics(y_true_sign, y_pred_sign):
    """Compute binary classification metrics for {-1, 1} labels.

    Returns a dict with counts and precision/recall/F1/accuracy.
    """
    tp = np.sum((y_true_sign == 1) & (y_pred_sign == 1))
    fp = np.sum((y_true_sign == -1) & (y_pred_sign == 1))
    fn = np.sum((y_true_sign == 1) & (y_pred_sign == -1))
    tn = np.sum((y_true_sign == -1) & (y_pred_sign == -1))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / max(1, y_true_sign.size)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


def compute_auc(scores, y_true_sign):
    """Compute ROC AUC and PR AUC from scores and {-1,1} labels."""
    order = np.argsort(-scores)
    y = (y_true_sign[order] == 1).astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    pos = y.sum()
    neg = y.size - pos
    tpr = tp / max(1.0, pos)
    fpr = fp / max(1.0, neg)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tpr
    roc_auc = float(np.trapz(tpr, fpr))
    pr_auc = float(np.trapz(precision, recall))
    return roc_auc, pr_auc


def kfold_indices(n_samples, k, seed=1, y=None):
    """Create stratified K-fold indices for {-1, 1} labels.

    If `y` is None, returns non-stratified folds.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    if y is None:
        rng.shuffle(indices)
        folds = np.array_split(indices, k)
        return folds
    # Stratified by sign of y
    pos = indices[y == 1]
    neg = indices[y == -1]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    folds = [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]
    for f in folds:
        rng.shuffle(f)
    return folds


def train_once_with_early_stopping(model, X, y_sign, args):
    """Train with early stopping by monitoring loss during iterations."""
    n_features = X.shape[1]
    init_w = np.zeros(n_features)
    y01 = to01(y_sign)

    # Early stopping parameters
    tol = getattr(args, "early_stopping_tol", 1e-6)
    patience = getattr(args, "early_stopping_patience", 10)

    w = init_w.copy()
    prev_loss = float("inf")
    patience_counter = 0

    for i in range(args.max_iters):
        # Manual gradient step (replicating implementations.py logic)
        if model == "logistic":
            grad = _logistic_gradient(y01, X, w)
        elif model == "reg_logistic":
            grad = _logistic_gradient(y01, X, w) + 2.0 * args.lambda_ * w
        elif model == "logistic_weighted":
            # For weighted models, we need to use the weighted implementations
            class_weight = {
                0: getattr(args, "class_weight_0", 1.0),
                1: getattr(args, "class_weight_1", 1.0),
            }
            # Calculate sample weights
            sample_weights = np.ones(len(y01))
            for class_label, weight in class_weight.items():
                sample_weights[y01 == class_label] = weight
            # Weighted gradient (VECTORIZED)
            z = X @ w
            z = np.clip(z, -500, 500)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (p - y01) * sample_weights
            grad = X.T @ grad / len(y01)
        elif model == "reg_logistic_weighted":
            # For weighted models, we need to use the weighted implementations
            class_weight = {
                0: getattr(args, "class_weight_0", 1.0),
                1: getattr(args, "class_weight_1", 1.0),
            }
            # Calculate sample weights
            sample_weights = np.ones(len(y01))
            for class_label, weight in class_weight.items():
                sample_weights[y01 == class_label] = weight
            # Weighted gradient + regularization (VECTORIZED)
            z = X @ w
            z = np.clip(z, -500, 500)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (p - y01) * sample_weights
            grad = X.T @ grad / len(y01) + 2.0 * args.lambda_ * w
        else:
            raise ValueError(f"Unknown model: {model}")

        w -= args.gamma * grad

        # Early stopping check every 10 iterations
        if i % 10 == 0:
            current_loss = _logistic_loss(y01, X, w)
            if abs(prev_loss - current_loss) < tol:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
            prev_loss = current_loss

    loss = _logistic_loss(y01, X, w)
    return w, float(loss)


def train_once(model, X, y_sign, args):
    """Train a single model on the provided data.

    Args:
        model: Model type string.
        X: Training features.
        y_sign: Labels in {-1, 1}.
        args: Namespace with hyperparameters.

    Returns:
        Tuple `(model_or_weights, loss)` with learned model/weights and data loss.
    """
    # Get random state for reproducibility
    random_state = getattr(args, "seed", None)

    # Use early stopping if enabled for logistic models
    if model in [
        "logistic",
        "reg_logistic",
        "logistic_weighted",
        "reg_logistic_weighted",
    ] and not getattr(args, "no_early_stopping", False):
        return train_once_with_early_stopping(model, X, y_sign, args)

    # Logistic family models
    if model == "logistic":
        n_features = X.shape[1]
        init_w = np.zeros(n_features)
        y01 = to01(y_sign)
        w, loss = logistic_regression(
            y01, X, init_w, max_iters=args.max_iters, gamma=args.gamma
        )
        return w, float(loss)

    elif model == "reg_logistic":
        n_features = X.shape[1]
        init_w = np.zeros(n_features)
        y01 = to01(y_sign)
        w, loss = reg_logistic_regression(
            y01,
            X,
            lambda_=args.lambda_,
            initial_w=init_w,
            max_iters=args.max_iters,
            gamma=args.gamma,
        )
        return w, float(loss)

    elif model == "logistic_weighted":
        n_features = X.shape[1]
        init_w = np.zeros(n_features)
        y01 = to01(y_sign)
        class_weight = {
            0: getattr(args, "class_weight_0", 1.0),
            1: getattr(args, "class_weight_1", 1.0),
        }
        w, loss = logistic_regression_weighted(
            y01,
            X,
            init_w,
            max_iters=args.max_iters,
            gamma=args.gamma,
            class_weight=class_weight,
            random_state=random_state,
        )
        return w, float(loss)

    elif model == "reg_logistic_weighted":
        n_features = X.shape[1]
        init_w = np.zeros(n_features)
        y01 = to01(y_sign)
        class_weight = {
            0: getattr(args, "class_weight_0", 1.0),
            1: getattr(args, "class_weight_1", 1.0),
        }
        w, loss = reg_logistic_regression_weighted(
            y01,
            X,
            lambda_=args.lambda_,
            initial_w=init_w,
            max_iters=args.max_iters,
            gamma=args.gamma,
            class_weight=class_weight,
            random_state=random_state,
        )
        return w, float(loss)

    # Random Forest
    elif model == "random_forest":
        y01 = to01(y_sign)
        rf = RandomForestClassifier(
            n_estimators=getattr(args, "n_estimators", 100),
            max_depth=getattr(args, "max_depth", 10),
            min_samples_split=getattr(args, "min_samples_split", 20),
            min_samples_leaf=getattr(args, "min_samples_leaf", 10),
            max_features=getattr(args, "max_features", "sqrt"),
            random_state=random_state,
            categorical_features=getattr(args, "categorical_features", []),
        )
        rf.fit(X, y01)
        # Calculate training loss (misclassification rate)
        train_pred = rf.predict(X)
        loss = np.mean(train_pred != y01)
        return rf, float(loss)

    else:
        raise ValueError(f"Unknown model: {model}")


def eval_fold_cv(i, folds, X, y_sign, model, args, k):
    """Evaluate a single CV fold - moved outside for ProcessPoolExecutor pickling."""
    val_idx = folds[i]
    train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
    Xtr, ytr = X[train_idx], y_sign[train_idx]
    Xva, yva = X[val_idx], y_sign[val_idx]

    w, loss = train_once(model, Xtr, ytr, args)
    scores = predict_scores(Xva, w)
    yhat = to_sign(scores, threshold=args.threshold)
    metrics = compute_metrics(yva, yhat)
    roc_auc, pr_auc = compute_auc(scores, yva)
    return loss, metrics, roc_auc, pr_auc


def eval_combo_grid(params, model, X, y_sign, k, seed, metric):
    """Evaluate a single grid search combination - moved outside for ProcessPoolExecutor pickling."""

    # clone args with overrides to avoid race conditions
    class A:
        pass

    a = A()
    a.progress = False
    a.cv_n_jobs = 1
    a.metric = metric

    # Set parameters based on model type
    if model in ["logistic", "reg_logistic"]:
        g, lmb, thr, iters = params
        a.gamma = g
        a.lambda_ = lmb
        a.threshold = thr
        a.max_iters = iters
    elif model in ["logistic_weighted", "reg_logistic_weighted"]:
        g, lmb, thr, iters, cw0, cw1 = params
        a.gamma = g
        a.lambda_ = lmb
        a.threshold = thr
        a.max_iters = iters
        a.class_weight_0 = cw0
        a.class_weight_1 = cw1
        a.cv_n_jobs = 1
    elif model == "random_forest":
        n_est, max_d, min_split, min_leaf, max_feat = params
        a.n_estimators = n_est
        a.max_depth = max_d
        a.min_samples_split = min_split
        a.min_samples_leaf = min_leaf
        a.max_features = max_feat
        a.categorical_features = []
        # Add threshold for Random Forest (not used in training but needed for evaluation)
        a.threshold = 0.5

    result = cross_validate(model, X, y_sign, k=k, seed=seed, args=a)[0]
    return params, result


def cross_validate(model, X, y_sign, k, seed, args):
    """Perform K-fold cross-validation for logistic-family models.

    Returns aggregate metrics (incl. AUCs), per-fold metrics, per-fold losses.
    """
    folds = kfold_indices(X.shape[0], k, seed=seed, y=y_sign)
    fold_metrics = []
    fold_losses = []
    fold_roc_aucs = []
    fold_pr_aucs = []

    cv_jobs = getattr(args, "cv_n_jobs", 1)
    if cv_jobs > 1:
        print(f"CV parallel using ProcessPoolExecutor with {cv_jobs} workers")
        with ProcessPoolExecutor(max_workers=cv_jobs) as ex:
            futures = [
                ex.submit(eval_fold_cv, i, folds, X, y_sign, model, args, k)
                for i in range(k)
            ]
            it = futures
            if getattr(args, "progress", True):
                it = tqdm(as_completed(futures), total=k, desc="CV folds", leave=False)
                for fut in it:
                    loss, metrics, roc_auc, pr_auc = fut.result()
                    fold_losses.append(loss)
                    fold_metrics.append(metrics)
                    fold_roc_aucs.append(roc_auc)
                    fold_pr_aucs.append(pr_auc)
            else:
                for fut in as_completed(futures):
                    loss, metrics, roc_auc, pr_auc = fut.result()
                    fold_losses.append(loss)
                    fold_metrics.append(metrics)
                    fold_roc_aucs.append(roc_auc)
                    fold_pr_aucs.append(pr_auc)
    else:
        fold_iter = range(k)
        if getattr(args, "progress", True):
            fold_iter = tqdm(fold_iter, desc="CV folds", leave=False)
        for i in fold_iter:
            loss, metrics, roc_auc, pr_auc = eval_fold_cv(
                i, folds, X, y_sign, model, args, k
            )
            fold_losses.append(loss)
            fold_metrics.append(metrics)
            fold_roc_aucs.append(roc_auc)
            fold_pr_aucs.append(pr_auc)

    # Aggregate
    agg = {
        k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()
    }
    agg["loss_mean"] = float(np.mean(fold_losses))
    agg["loss_std"] = float(np.std(fold_losses))
    agg["roc_auc"] = float(np.mean(fold_roc_aucs))
    agg["pr_auc"] = float(np.mean(fold_pr_aucs))
    return agg, fold_metrics, fold_losses


def plot_curves(scores, y_true_sign, out_dir, prefix="train"):
    """Plot ROC and PR curves and save PNGs.

    Args:
        scores: Scores or probabilities (higher means more positive).
        y_true_sign: Ground-truth labels in {-1, 1}.
        out_dir: Directory to save figures.
        prefix: Filename prefix for the plots.

    Returns:
        Dict with numeric AUCs for ROC and PR.
    """
    # ROC and PR from scratch (coarse) for binary sign labels
    # Sort by descending score
    order = np.argsort(-scores)
    y = (y_true_sign[order] == 1).astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    pos = y.sum()
    neg = y.size - pos
    tpr = tp / max(1.0, pos)
    fpr = fp / max(1.0, neg)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tpr

    # ROC AUC (trapezoid)
    roc_auc = float(np.trapz(tpr, fpr))
    # PR AUC
    pr_auc = float(np.trapz(precision, recall))

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"))
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png"))
    plt.close()

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def main():
    """CLI entrypoint to train and evaluate logistic models on preprocessed data."""
    parser = argparse.ArgumentParser(
        description="Train logistic models on preprocessed data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Folder with x_train.csv,y_train.csv,x_test.csv. Overrides config.json if set.",
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "logistic",
            "reg_logistic",
            "logistic_weighted",
            "reg_logistic_weighted",
            "random_forest",
        ],
        help="Model to train. Overrides config.json if set.",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="K folds; overrides config.json"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Learning rate for GD/SGD/logistic"
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=1e-3,
        help="Regularization strength for ridge/reg_logistic",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=1000,
        help="Max iterations for iterative methods",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for logistic models",
    )
    # Grid-search options
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "roc_auc", "pr_auc", "precision", "recall"],
        help="Selection metric for grid search",
    )
    parser.add_argument(
        "--gamma_grid",
        type=str,
        default=None,
        help="Comma-separated gamma values for search",
    )
    parser.add_argument(
        "--lambda_grid",
        type=str,
        default=None,
        help="Comma-separated lambda values for search",
    )
    parser.add_argument(
        "--threshold_grid",
        type=str,
        default=None,
        help="Comma-separated threshold values for search",
    )
    parser.add_argument(
        "--max_iters_grid",
        type=str,
        default=None,
        help="Comma-separated max_iters values for search",
    )
    parser.add_argument(
        "--search_max_iters_grid",
        type=str,
        default=None,
        help="Comma-separated max_iters for SEARCH (overrides max_iters_grid during CV)",
    )
    parser.add_argument(
        "--final_max_iters",
        type=int,
        default=None,
        help="Max iterations for FINAL training only (overrides best from search)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Output directory base; overrides config.json",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Run tag for naming the results folder"
    )
    parser.add_argument(
        "--make_submission",
        action="store_true",
        help="Write submission CSV using x_test.csv",
    )
    parser.add_argument(
        "--no_progress", action="store_true", help="Disable tqdm progress bars"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose prints during search/training"
    )
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="dataset",
        help="Path to raw dataset to fetch test_ids for submission",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel workers for grid search (1 = no parallelism)",
    )
    parser.add_argument(
        "--cv_n_jobs",
        type=int,
        default=1,
        help="Parallel workers for cross-validation folds (batch-CV)",
    )
    parser.add_argument(
        "--early_stopping_tol",
        type=float,
        default=1e-6,
        help="Tolerance for early stopping (loss change threshold)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping (iterations to wait)",
    )
    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Disable early stopping (use original implementations)",
    )
    # Random Forest parameters
    parser.add_argument(
        "--max_depth", type=int, default=10, help="Maximum depth for Random Forest"
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=20,
        help="Minimum samples to split for Random Forest",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=10,
        help="Minimum samples per leaf for Random Forest",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators for Random Forest",
    )
    parser.add_argument(
        "--max_features",
        type=str,
        default="sqrt",
        choices=["sqrt", "log2", "all"],
        help="Number of features to consider for splits in Random Forest",
    )

    # Random Forest grid search parameters
    parser.add_argument(
        "--n_estimators_grid",
        type=str,
        default=None,
        help="Comma-separated n_estimators values for Random Forest grid search",
    )
    parser.add_argument(
        "--max_depth_grid",
        type=str,
        default=None,
        help="Comma-separated max_depth values for Random Forest grid search",
    )
    parser.add_argument(
        "--min_samples_split_grid",
        type=str,
        default=None,
        help="Comma-separated min_samples_split values for Random Forest grid search",
    )
    parser.add_argument(
        "--min_samples_leaf_grid",
        type=str,
        default=None,
        help="Comma-separated min_samples_leaf values for Random Forest grid search",
    )
    parser.add_argument(
        "--max_features_grid",
        type=str,
        default=None,
        help="Comma-separated max_features values for Random Forest grid search",
    )

    # Categorical features (for Random Forest)
    parser.add_argument(
        "--categorical_features",
        type=str,
        default=None,
        help="Comma-separated list of categorical feature indices",
    )

    # Class weights (for weighted logistic regression)
    parser.add_argument(
        "--class_weight_0",
        type=float,
        default=1.0,
        help="Weight for class 0 (negative class) in weighted logistic regression",
    )
    parser.add_argument(
        "--class_weight_1",
        type=float,
        default=1.0,
        help="Weight for class 1 (positive class) in weighted logistic regression",
    )
    parser.add_argument(
        "--class_weight_0_grid",
        type=str,
        default=None,
        help="Comma-separated class_weight_0 values for grid search",
    )
    parser.add_argument(
        "--class_weight_1_grid",
        type=str,
        default=None,
        help="Comma-separated class_weight_1 values for grid search",
    )

    args = parser.parse_args()

    # Load config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)

    data_dir = args.data_dir or cfg.get("data_dir", "preprocessed/level0")
    model = args.model or cfg.get("model", "logistic")
    # support both k_folds and k in config
    k = args.k or int(cfg.get("k_folds", cfg.get("k", 5)))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 1))
    results_root = args.results_dir or cfg.get("results_dir", "results")
    tag = args.tag or cfg.get("tag", model)
    os.makedirs(results_root, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_root, f"{model}_{tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Configure progress flag
    args.progress = not args.no_progress

    # Store original cv_n_jobs for later restoration after grid search
    args.original_cv_n_jobs = getattr(args, "cv_n_jobs", 1)

    # Parse categorical features if specified
    if args.categorical_features is not None:
        try:
            args.categorical_features = [
                int(x.strip())
                for x in args.categorical_features.split(",")
                if x.strip()
            ]
        except ValueError:
            print("Warning: Invalid categorical_features format. Using empty list.")
            args.categorical_features = []
    else:
        args.categorical_features = []

    # Merge parallelism settings from config if provided
    if args.n_jobs == 1 and "n_jobs" in cfg:
        try:
            args.n_jobs = int(cfg["n_jobs"])
        except Exception:
            pass
    if getattr(args, "cv_n_jobs", 1) == 1 and "cv_n_jobs" in cfg:
        try:
            args.cv_n_jobs = int(cfg["cv_n_jobs"])
        except Exception:
            pass

    print("Started Loading")
    # Load data: always use preprocessed matrices; fetch test_ids from raw dataset
    X, y_sign, Xtest = load_preprocessed(data_dir)
    try:
        # prefer raw_dataset_dir from config if provided
        raw_dir = cfg.get("raw_dataset_dir", args.raw_dataset_dir)
        _, _, _, _, test_ids = load_csv_data(raw_dir)
    except Exception:
        # Safe fallback if raw dataset not available
        test_ids = np.arange(1, Xtest.shape[0] + 1)
    y_sign = y_sign.astype(int)
    print(f"Loaded {X.shape} train, {Xtest.shape} test from {data_dir}")

    def parse_grid(opt, cast=float):
        if opt is None:
            return None
        vals = [s.strip() for s in opt.split(",") if s.strip()]
        return [cast(v) for v in vals]

    # Allow metric and grids from config when CLI flags not provided
    if "metric" in cfg and (not hasattr(args, "metric") or args.metric == "f1"):
        args.metric = cfg["metric"]
    gamma_grid = parse_grid(
        args.gamma_grid if args.gamma_grid is not None else cfg.get("gamma_grid"), float
    )
    lambda_grid = parse_grid(
        args.lambda_grid if args.lambda_grid is not None else cfg.get("lambda_grid"),
        float,
    )
    threshold_grid = parse_grid(
        (
            args.threshold_grid
            if args.threshold_grid is not None
            else cfg.get("threshold_grid")
        ),
        float,
    )
    max_iters_grid = parse_grid(
        (
            args.max_iters_grid
            if args.max_iters_grid is not None
            else cfg.get("max_iters_grid")
        ),
        int,
    )
    search_max_iters_grid = parse_grid(
        (
            args.search_max_iters_grid
            if args.search_max_iters_grid is not None
            else cfg.get("search_max_iters_grid")
        ),
        int,
    )
    final_max_iters = (
        args.final_max_iters
        if args.final_max_iters is not None
        else cfg.get("final_max_iters")
    )

    # Class weight grids
    class_weight_0_grid = parse_grid(
        (
            args.class_weight_0_grid
            if args.class_weight_0_grid is not None
            else cfg.get("class_weight_0_grid")
        ),
        float,
    )
    class_weight_1_grid = parse_grid(
        (
            args.class_weight_1_grid
            if args.class_weight_1_grid is not None
            else cfg.get("class_weight_1_grid")
        ),
        float,
    )

    # Random Forest parameters from config
    if "n_estimators" in cfg and not hasattr(args, "n_estimators"):
        args.n_estimators = cfg["n_estimators"]
    if "max_depth" in cfg and not hasattr(args, "max_depth"):
        args.max_depth = cfg["max_depth"]
    if "min_samples_split" in cfg and not hasattr(args, "min_samples_split"):
        args.min_samples_split = cfg["min_samples_split"]
    if "min_samples_leaf" in cfg and not hasattr(args, "min_samples_leaf"):
        args.min_samples_leaf = cfg["min_samples_leaf"]
    if "max_features" in cfg and not hasattr(args, "max_features"):
        args.max_features = cfg["max_features"]
    if "categorical_features" in cfg and not hasattr(args, "categorical_features"):
        args.categorical_features = cfg["categorical_features"]
    else:
        args.categorical_features = []

    best = None
    searched = False
    # prefer dedicated search grid for iterations if provided
    iters_list = search_max_iters_grid or max_iters_grid

    # Check if we should do grid search based on model type
    if model == "random_forest":
        # Check for Random Forest grid parameters
        n_estimators_grid = parse_grid(
            (
                args.n_estimators_grid
                if hasattr(args, "n_estimators_grid") and args.n_estimators_grid
                else cfg.get("n_estimators_grid")
            ),
            int,
        )
        max_depth_grid = parse_grid(
            (
                args.max_depth_grid
                if hasattr(args, "max_depth_grid") and args.max_depth_grid
                else cfg.get("max_depth_grid")
            ),
            int,
        )
        min_samples_split_grid = parse_grid(
            (
                args.min_samples_split_grid
                if hasattr(args, "min_samples_split_grid")
                and args.min_samples_split_grid
                else cfg.get("min_samples_split_grid")
            ),
            int,
        )
        min_samples_leaf_grid = parse_grid(
            (
                args.min_samples_leaf_grid
                if hasattr(args, "min_samples_leaf_grid") and args.min_samples_leaf_grid
                else cfg.get("min_samples_leaf_grid")
            ),
            int,
        )
        max_features_grid = parse_grid(
            (
                args.max_features_grid
                if hasattr(args, "max_features_grid") and args.max_features_grid
                else cfg.get("max_features_grid")
            ),
            str,
        )

        if any(
            v is not None
            for v in [
                n_estimators_grid,
                max_depth_grid,
                min_samples_split_grid,
                min_samples_leaf_grid,
                max_features_grid,
            ]
        ):
            searched = True
            print("Starting Random Forest grid search...")
    else:
        # Check for logistic regression grid parameters (including weighted models)
        if any(
            v is not None
            for v in [
                gamma_grid,
                lambda_grid,
                threshold_grid,
                iters_list,
                class_weight_0_grid,
                class_weight_1_grid,
            ]
        ):
            searched = True
            if model in ["logistic_weighted", "reg_logistic_weighted"]:
                print("Starting weighted logistic regression grid search...")
            else:
                print("Starting logistic regression grid search...")

    if searched:
        gammas = gamma_grid or [args.gamma]
        lambdas = lambda_grid or [args.lambda_]
        thresholds = threshold_grid or [args.threshold]
        # iters_list already set above

        # Class weights for weighted models
        class_weights_0 = class_weight_0_grid or [getattr(args, "class_weight_0", 1.0)]
        class_weights_1 = class_weight_1_grid or [getattr(args, "class_weight_1", 1.0)]

        print("Starting grid search...")

        # Generate parameter combinations based on model type
        if model == "random_forest":
            # Parse Random Forest parameters
            n_estimators_list = n_estimators_grid
            max_depth_list = max_depth_grid
            min_samples_split_list = min_samples_split_grid
            min_samples_leaf_list = min_samples_leaf_grid
            max_features_list = max_features_grid

            if n_estimators_list is None:
                n_estimators_list = [getattr(args, "n_estimators", 100)]
            if max_depth_list is None:
                max_depth_list = [getattr(args, "max_depth", 10)]
            if min_samples_split_list is None:
                min_samples_split_list = [getattr(args, "min_samples_split", 20)]
            if min_samples_leaf_list is None:
                min_samples_leaf_list = [getattr(args, "min_samples_leaf", 10)]
            if max_features_list is None:
                max_features_list = [getattr(args, "max_features", "sqrt")]

            total = (
                len(n_estimators_list)
                * len(max_depth_list)
                * len(min_samples_split_list)
                * len(min_samples_leaf_list)
                * len(max_features_list)
            )
            combos = [
                (n_est, max_d, min_split, min_leaf, max_feat)
                for n_est in n_estimators_list
                for max_d in max_depth_list
                for min_split in min_samples_split_list
                for min_leaf in min_samples_leaf_list
                for max_feat in max_features_list
            ]
        else:
            # Logistic regression parameters (including weighted models)
            if model in ["logistic_weighted", "reg_logistic_weighted"]:
                total = (
                    len(gammas)
                    * len(lambdas)
                    * len(thresholds)
                    * len(iters_list)
                    * len(class_weights_0)
                    * len(class_weights_1)
                )
                combos = [
                    (g, lmb, thr, iters, cw0, cw1)
                    for g in gammas
                    for lmb in lambdas
                    for thr in thresholds
                    for iters in iters_list
                    for cw0 in class_weights_0
                    for cw1 in class_weights_1
                ]
            else:
                total = len(gammas) * len(lambdas) * len(thresholds) * len(iters_list)
                combos = [
                    (g, lmb, thr, iters)
                    for g in gammas
                    for lmb in lambdas
                    for thr in thresholds
                    for iters in iters_list
                ]

        if args.n_jobs and args.n_jobs > 1:
            print(
                f"Grid search parallel using ProcessPoolExecutor with {args.n_jobs} workers"
            )
            with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                futures = [
                    ex.submit(
                        eval_combo_grid, p, model, X, y_sign, k, seed, args.metric
                    )
                    for p in combos
                ]
                it = futures
                if args.progress:
                    it = tqdm(
                        as_completed(futures),
                        total=total,
                        desc="Grid search",
                        leave=False,
                    )
                for fut in it:
                    if args.progress:
                        params, agg_res = fut.result()
                    else:
                        params, agg_res = fut.result()

                    score = agg_res[args.metric]

                    # Create candidate based on model type
                    if model == "random_forest":
                        n_est, max_d, min_split, min_leaf, max_feat = params
                        cand = {
                            "n_estimators": n_est,
                            "max_depth": max_d,
                            "min_samples_split": min_split,
                            "min_samples_leaf": min_leaf,
                            "max_features": max_feat,
                            "metric": args.metric,
                            "score": float(score),
                            "cv": agg_res,
                        }
                    elif model in ["logistic_weighted", "reg_logistic_weighted"]:
                        g, lmb, thr, iters, cw0, cw1 = params
                        cand = {
                            "gamma": g,
                            "lambda_": lmb,
                            "threshold": thr,
                            "max_iters": iters,
                            "class_weight_0": cw0,
                            "class_weight_1": cw1,
                            "metric": args.metric,
                            "score": float(score),
                            "cv": agg_res,
                        }
                    else:
                        g, lmb, thr, iters = params
                        cand = {
                            "gamma": g,
                            "lambda_": lmb,
                            "threshold": thr,
                            "max_iters": iters,
                            "metric": args.metric,
                            "score": float(score),
                            "cv": agg_res,
                        }

                    if (best is None) or (score > best["score"]):
                        best = cand
                        if args.verbose:
                            print(f"New best {args.metric}={score:.4f} with {cand}")
        else:
            iterator = combos
            if args.progress:
                iterator = tqdm(combos, total=total, desc="Grid search", leave=False)
            for params in iterator:
                # Create args object based on model type
                if model == "random_forest":
                    n_est, max_d, min_split, min_leaf, max_feat = params
                    a = argparse.Namespace(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        max_features=max_feat,
                        categorical_features=[],
                        threshold=0.5,
                        progress=False,
                        metric=args.metric,
                        cv_n_jobs=args.cv_n_jobs,
                    )
                elif model in ["logistic_weighted", "reg_logistic_weighted"]:
                    g, lmb, thr, iters, cw0, cw1 = params
                    a = argparse.Namespace(
                        gamma=g,
                        lambda_=lmb,
                        threshold=thr,
                        max_iters=iters,
                        class_weight_0=cw0,
                        class_weight_1=cw1,
                        progress=False,
                        metric=args.metric,
                        cv_n_jobs=args.cv_n_jobs,
                    )
                else:
                    g, lmb, thr, iters = params
                    a = argparse.Namespace(
                        gamma=g,
                        lambda_=lmb,
                        threshold=thr,
                        max_iters=iters,
                        progress=False,
                        metric=args.metric,
                        cv_n_jobs=args.cv_n_jobs,
                    )

                agg_res, _, _ = cross_validate(model, X, y_sign, k=k, seed=seed, args=a)
                score = agg_res[args.metric]

                # Create candidate based on model type
                if model == "random_forest":
                    n_est, max_d, min_split, min_leaf, max_feat = params
                    cand = {
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "min_samples_split": min_split,
                        "min_samples_leaf": min_leaf,
                        "max_features": max_feat,
                        "metric": args.metric,
                        "score": float(score),
                        "cv": agg_res,
                    }
                elif model in ["logistic_weighted", "reg_logistic_weighted"]:
                    g, lmb, thr, iters, cw0, cw1 = params
                    cand = {
                        "gamma": g,
                        "lambda_": lmb,
                        "threshold": thr,
                        "max_iters": iters,
                        "class_weight_0": cw0,
                        "class_weight_1": cw1,
                        "metric": args.metric,
                        "score": float(score),
                        "cv": agg_res,
                    }
                else:
                    g, lmb, thr, iters = params
                    cand = {
                        "gamma": g,
                        "lambda_": lmb,
                        "threshold": thr,
                        "max_iters": iters,
                        "metric": args.metric,
                        "score": float(score),
                        "cv": agg_res,
                    }

                if (best is None) or (score > best["score"]):
                    best = cand
                    if args.verbose:
                        print(f"New best {args.metric}={score:.4f} with {cand}")
        # set best params back to args
        if model == "random_forest":
            args.n_estimators = best["n_estimators"]
            args.max_depth = best["max_depth"]
            args.min_samples_split = best["min_samples_split"]
            args.min_samples_leaf = best["min_samples_leaf"]
            args.max_features = best["max_features"]
        elif model in ["logistic_weighted", "reg_logistic_weighted"]:
            args.gamma = best["gamma"]
            args.lambda_ = best["lambda_"]
            args.threshold = best["threshold"]
            args.class_weight_0 = best["class_weight_0"]
            args.class_weight_1 = best["class_weight_1"]
            # set final training iterations: prefer explicit final_max_iters, else best from search
            args.max_iters = (
                int(final_max_iters)
                if final_max_iters is not None
                else best["max_iters"]
            )
        else:
            args.gamma = best["gamma"]
            args.lambda_ = best["lambda_"]
            args.threshold = best["threshold"]
            # set final training iterations: prefer explicit final_max_iters, else best from search
            args.max_iters = (
                int(final_max_iters)
                if final_max_iters is not None
                else best["max_iters"]
            )
        print(f"Best by {args.metric}: {best}")

    # CV with final params (either searched best or provided)
    # Restore original cv_n_jobs for parallel CV after grid search
    original_cv_n_jobs = getattr(args, "original_cv_n_jobs", 1)
    args.cv_n_jobs = original_cv_n_jobs
    agg, per_fold, losses = cross_validate(model, X, y_sign, k=k, seed=seed, args=args)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})

    # Train on full train set with final params
    w, train_loss = train_once(model, X, y_sign, args)

    # COMMENTED OUT: Post-training threshold selection to prevent overfitting
    # This section was causing overfitting because:
    # 1. Target encoding already uses target information during preprocessing
    # 2. Threshold selection uses target information again
    # 3. This creates double information leakage and inflated validation scores
    #
    # if args.metric in ["pr_auc", "roc_auc"]:
    #     print("Selecting optimal threshold on validation split...")
    #     from sklearn.model_selection import train_test_split
    #     X_train_final, X_val, y_train_final, y_val = train_test_split(
    #         X, y_sign, test_size=0.2, random_state=seed, stratify=y_sign
    #     )
    #
    #     # Retrain on train_final
    #     w_final, _ = train_once(model, X_train_final, y_train_final, args)
    #
    #     # Find best threshold on validation
    #     val_scores = predict_scores(X_val, w_final)
    #     best_threshold = 0.5
    #     best_f1 = 0
    #
    #     for thresh in np.arange(0.1, 0.9, 0.05):
    #         yhat = to_sign(val_scores, threshold=thresh)
    #         metrics = compute_metrics(y_val, yhat)
    #         if metrics["f1"] > best_f1:
    #             best_f1 = metrics["f1"]
    #             best_threshold = thresh
    #
    #     args.threshold = best_threshold
    #     print(f"Selected threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    #
    #     # Retrain on full dataset with selected threshold
    #     w, train_loss = train_once(model, X, y_sign, args)

    # Train-set curves (using train predictions)
    scores_train = predict_scores(X, w)
    aucs = plot_curves(scores_train, y_sign, out_dir, prefix=f"{model}_train")

    # Save summary
    if model == "random_forest":
        params_dict = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "categorical_features": args.categorical_features,
        }
    elif model in ["logistic_weighted", "reg_logistic_weighted"]:
        params_dict = {
            "gamma": args.gamma,
            "lambda_": args.lambda_,
            "max_iters": args.max_iters,
            "threshold": args.threshold,
            "class_weight_0": getattr(args, "class_weight_0", 1.0),
            "class_weight_1": getattr(args, "class_weight_1", 1.0),
        }
    else:
        params_dict = {
            "gamma": args.gamma,
            "lambda_": args.lambda_,
            "max_iters": args.max_iters,
            "threshold": args.threshold,
        }

    summary = {
        "data_dir": data_dir,
        "model": model,
        "k_folds": k,
        "seed": seed,
        "params": params_dict,
        "cv": {
            "aggregate": agg,
            "per_fold": per_fold,
            "losses": losses,
        },
        "train": {
            "loss": float(train_loss),
            "roc_auc": aucs.get("roc_auc"),
            "pr_auc": aucs.get("pr_auc"),
        },
    }
    if searched and best is not None:
        summary["best_search"] = best
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Optional submission
    if args.make_submission or cfg.get("make_submission", False):
        test_scores = predict_scores(Xtest, w)
        ypred_sign = to_sign(test_scores, threshold=args.threshold)
        sub_path = os.path.join(out_dir, "submission.csv")
        create_csv_submission(test_ids, ypred_sign, sub_path)
        print(f"Wrote submission: {sub_path}")

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
