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
from helpers import create_csv_submission, load_csv_data


def load_preprocessed(data_dir):
    """Load preprocessed matrices (no IDs) from a directory.

    Expects `x_train.csv`, `y_train.csv`, `x_test.csv` in `data_dir`.
    Returns `X, y, Xtest`.
    """
    x_train = np.loadtxt(os.path.join(data_dir, "x_train.csv"), delimiter=",", dtype=np.float64)
    y_train = np.loadtxt(os.path.join(data_dir, "y_train.csv"), delimiter=",", dtype=np.float64)
    x_test = np.loadtxt(os.path.join(data_dir, "x_test.csv"), delimiter=",", dtype=np.float64)
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


def predict_scores(X, w):
    """Predict positive-class probabilities for logistic models.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        w: Weight vector of shape (n_features,).

    Returns:
        Array of probabilities in [0, 1].
    """
    # probabilities for logistic family
    scores = X @ w
    return sigmoid(scores)


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
    tol = getattr(args, 'early_stopping_tol', 1e-6)
    patience = getattr(args, 'early_stopping_patience', 10)
    
    w = init_w.copy()
    prev_loss = float('inf')
    patience_counter = 0
    
    for i in range(args.max_iters):
        # Manual gradient step (replicating implementations.py logic)
        if model == "logistic":
            grad = _logistic_gradient(y01, X, w)
        elif model == "reg_logistic":
            grad = _logistic_gradient(y01, X, w) + 2.0 * args.lambda_ * w
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
    """Train a single logistic-family model on the provided data.

    Args:
        model: Either "logistic" or "reg_logostic".
        X: Training features.
        y_sign: Labels in {-1, 1}.
        args: Namespace with hyperparameters.

    Returns:
        Tuple `(w, loss)` with learned weights and data loss.
    """
    # Use early stopping if enabled
    if not getattr(args, 'no_early_stopping', False):
        return train_once_with_early_stopping(model, X, y_sign, args)
    
    # Fallback to original implementations
    n_features = X.shape[1]
    init_w = np.zeros(n_features)
    y01 = to01(y_sign)

    if model == "logistic":
        w, loss = logistic_regression(y01, X, init_w, max_iters=args.max_iters, gamma=args.gamma)
    elif model == "reg_logistic":
        w, loss = reg_logistic_regression(y01, X, lambda_=args.lambda_, initial_w=init_w, max_iters=args.max_iters, gamma=args.gamma)
    else:
        raise ValueError(f"Unknown model: {model}")
    return w, float(loss)


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
    g, lmb, thr, iters = params
    # clone args with overrides to avoid race conditions
    class A: pass
    a = A()
    a.gamma = g
    a.lambda_ = lmb
    a.threshold = thr
    a.max_iters = iters
    a.progress = False
    a.cv_n_jobs = 1
    # other needed fields
    a.metric = metric
    return params, cross_validate(model, X, y_sign, k=k, seed=seed, args=a)[0]


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
            futures = [ex.submit(eval_fold_cv, i, folds, X, y_sign, model, args, k) for i in range(k)]
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
            loss, metrics, roc_auc, pr_auc = eval_fold_cv(i, folds, X, y_sign, model, args, k)
            fold_losses.append(loss)
            fold_metrics.append(metrics)
            fold_roc_aucs.append(roc_auc)
            fold_pr_aucs.append(pr_auc)

    # Aggregate
    agg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
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
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
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
    parser = argparse.ArgumentParser(description="Train logistic models on preprocessed data")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder with x_train.csv,y_train.csv,x_test.csv. Overrides config.json if set.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--model", type=str, default=None, choices=["logistic","reg_logistic"], help="Model to train. Overrides config.json if set.")
    parser.add_argument("--k", type=int, default=None, help="K folds; overrides config.json")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate for GD/SGD/logistic")
    parser.add_argument("--lambda_", type=float, default=1e-3, help="Regularization strength for ridge/reg_logistic")
    parser.add_argument("--max_iters", type=int, default=1000, help="Max iterations for iterative methods")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for logistic models")
    # Grid-search options
    parser.add_argument("--metric", type=str, default="f1", choices=["f1","accuracy","roc_auc","pr_auc","precision","recall"], help="Selection metric for grid search")
    parser.add_argument("--gamma_grid", type=str, default=None, help="Comma-separated gamma values for search")
    parser.add_argument("--lambda_grid", type=str, default=None, help="Comma-separated lambda values for search")
    parser.add_argument("--threshold_grid", type=str, default=None, help="Comma-separated threshold values for search")
    parser.add_argument("--max_iters_grid", type=str, default=None, help="Comma-separated max_iters values for search")
    parser.add_argument("--search_max_iters_grid", type=str, default=None, help="Comma-separated max_iters for SEARCH (overrides max_iters_grid during CV)")
    parser.add_argument("--final_max_iters", type=int, default=None, help="Max iterations for FINAL training only (overrides best from search)")
    parser.add_argument("--results_dir", type=str, default=None, help="Output directory base; overrides config.json")
    parser.add_argument("--tag", type=str, default=None, help="Run tag for naming the results folder")
    parser.add_argument("--make_submission", action="store_true", help="Write submission CSV using x_test.csv")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--verbose", action="store_true", help="Verbose prints during search/training")
    parser.add_argument("--raw_dataset_dir", type=str, default="dataset", help="Path to raw dataset to fetch test_ids for submission")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers for grid search (1 = no parallelism)")
    parser.add_argument("--cv_n_jobs", type=int, default=1, help="Parallel workers for cross-validation folds (batch-CV)")
    parser.add_argument("--early_stopping_tol", type=float, default=1e-6, help="Tolerance for early stopping (loss change threshold)")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping (iterations to wait)")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping (use original implementations)")
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
    args.original_cv_n_jobs = getattr(args, 'cv_n_jobs', 1)

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
    gamma_grid = parse_grid(args.gamma_grid if args.gamma_grid is not None else cfg.get("gamma_grid"), float)
    lambda_grid = parse_grid(args.lambda_grid if args.lambda_grid is not None else cfg.get("lambda_grid"), float)
    threshold_grid = parse_grid(args.threshold_grid if args.threshold_grid is not None else cfg.get("threshold_grid"), float)
    max_iters_grid = parse_grid(args.max_iters_grid if args.max_iters_grid is not None else cfg.get("max_iters_grid"), int)
    search_max_iters_grid = parse_grid(args.search_max_iters_grid if args.search_max_iters_grid is not None else cfg.get("search_max_iters_grid"), int)
    final_max_iters = args.final_max_iters if args.final_max_iters is not None else cfg.get("final_max_iters")

    best = None
    searched = False
    # prefer dedicated search grid for iterations if provided
    iters_list = search_max_iters_grid or max_iters_grid 

    if any(v is not None for v in [gamma_grid, lambda_grid, threshold_grid, iters_list]):
        searched = True
        gammas = gamma_grid or [args.gamma]
        lambdas = lambda_grid or [args.lambda_]
        thresholds = threshold_grid or [args.threshold]
        # iters_list already set above

        print("Starting grid search...")
        total = len(gammas) * len(lambdas) * len(thresholds) * len(iters_list)
        # prepare list for stable ordering in tqdm
        combos = [(g, lmb, thr, iters) for g in gammas for lmb in lambdas for thr in thresholds for iters in iters_list]

        if args.n_jobs and args.n_jobs > 1:
            print(f"Grid search parallel using ProcessPoolExecutor with {args.n_jobs} workers")
            with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                futures = [ex.submit(eval_combo_grid, p, model, X, y_sign, k, seed, args.metric) for p in combos]
                it = futures
                if args.progress:
                    it = tqdm(as_completed(futures), total=total, desc="Grid search", leave=False)
                for fut in it:
                    if args.progress:
                        params, agg_res = fut.result()
                    else:
                        params, agg_res = fut.result()
                    g, lmb, thr, iters = params
                    score = agg_res[args.metric]
                    cand = {"gamma": g, "lambda_": lmb, "threshold": thr, "max_iters": iters, "metric": args.metric, "score": float(score), "cv": agg_res}
                    if (best is None) or (score > best["score"]):
                        best = cand
                        if args.verbose:
                            print(f"New best {args.metric}={score:.4f} with {cand}")
        else:
            iterator = combos
            if args.progress:
                iterator = tqdm(combos, total=total, desc="Grid search", leave=False)
            for g, lmb, thr, iters in iterator:
                a = argparse.Namespace(gamma=g, lambda_=lmb, threshold=thr, max_iters=iters, progress=False, metric=args.metric, cv_n_jobs=args.cv_n_jobs)
                agg_res, _, _ = cross_validate(model, X, y_sign, k=k, seed=seed, args=a)
                score = agg_res[args.metric]
                cand = {"gamma": g, "lambda_": lmb, "threshold": thr, "max_iters": iters, "metric": args.metric, "score": float(score), "cv": agg_res}
                if (best is None) or (score > best["score"]):
                    best = cand
                    if args.verbose:
                        print(f"New best {args.metric}={score:.4f} with {cand}")
        # set best params back to args
        args.gamma = best["gamma"]
        args.lambda_ = best["lambda_"]
        args.threshold = best["threshold"]
        # set final training iterations: prefer explicit final_max_iters, else best from search
        args.max_iters = int(final_max_iters) if final_max_iters is not None else best["max_iters"]
        print(f"Best by {args.metric}: {best}")

    # CV with final params (either searched best or provided)
    # Restore original cv_n_jobs for parallel CV after grid search
    original_cv_n_jobs = getattr(args, 'original_cv_n_jobs', 1)
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
    summary = {
        "data_dir": data_dir,
        "model": model,
        "k_folds": k,
        "seed": seed,
        "params": {
            "gamma": args.gamma,
            "lambda_": args.lambda_,
            "max_iters": args.max_iters,
            "threshold": args.threshold,
        },
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


