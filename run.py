"""
Experiment runner 

- Loads data via helpers.load_csv_data
- Optional preprocessing via preprocessing.preprocess_pipeline
- Supports raw or preprocessed features in one unified pipeline
- K-fold cross-validation and hyperparameter search
- Metrics: accuracy, precision, recall, F1, log loss (logistic), ROC-AUC, PR-AUC
- Plots: ROC, PR, and comparison bar charts
- Verbose terminal logs for each step

Only NumPy and Matplotlib are used, per project rules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from helpers import load_csv_data, create_csv_submission
import implementations as impl
import preprocessing as prep


# -------------------------------
# FS utils and RNG
# -------------------------------

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def round_floats(obj, ndigits: int = 5):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_floats(v, ndigits) for v in obj)
    return obj


# -------------------------------
# Metrics for binary classification
# y in {-1, 1}
# -------------------------------

def _to01(y_pm1: np.ndarray) -> np.ndarray:
    return ((y_pm1 + 1.0) / 2.0).astype(np.float64)


def accuracy(y_true_pm1: np.ndarray, y_pred_pm1: np.ndarray) -> float:
    return float((y_true_pm1 == y_pred_pm1).mean())


def precision_recall_f1(y_true_pm1: np.ndarray, y_pred_pm1: np.ndarray) -> Tuple[float, float, float]:
    tp = float(((y_true_pm1 == 1) & (y_pred_pm1 == 1)).sum())
    fp = float(((y_true_pm1 == -1) & (y_pred_pm1 == 1)).sum())
    fn = float(((y_true_pm1 == 1) & (y_pred_pm1 == -1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def log_loss_from_probs(y_true_pm1: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y01 = _to01(y_true_pm1)
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-(y01 * np.log(p) + (1 - y01) * np.log(1 - p)).mean())


def roc_curve(y_true_pm1: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y = (y_true_pm1[order] == 1).astype(np.int32)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P = float(y.sum())
    N = float((1 - y).sum())
    tpr = tp / P if P > 0 else np.zeros_like(tp, dtype=np.float64)
    fpr = fp / N if N > 0 else np.zeros_like(fp, dtype=np.float64)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def precision_recall_curve(y_true_pm1: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y = (y_true_pm1[order] == 1).astype(np.int32)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P = float(y.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P if P > 0 else np.zeros_like(tp, dtype=np.float64)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return recall, precision


def pr_auc(y_true_pm1: np.ndarray, y_score: np.ndarray) -> float:
    recall, precision = precision_recall_curve(y_true_pm1, y_score)
    return auc(recall, precision)


# -------------------------------
# Data loading / preprocessing (with optional caching)
# -------------------------------

@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    preprocess_report: dict | None


def load_data(
    data_dir: str,
    use_preprocessed: bool,
    seed: int | None,
    oversample_ratio: float | None,
    cont_features: List[int] | None,
    cache_data: bool = False,
    cache_dir: str | None = None,
) -> DataBundle:
    def file_mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def build_cache_key() -> str:
        xtr_p = os.path.join(data_dir, "x_train.csv")
        xte_p = os.path.join(data_dir, "x_test.csv")
        ytr_p = os.path.join(data_dir, "y_train.csv")
        payload = {
            "use_preprocessed": bool(use_preprocessed),
            "oversample": float(oversample_ratio) if oversample_ratio is not None else None,
            "cont_features": list(cont_features) if cont_features is not None else None,
            "mtimes": {
                "x_train.csv": file_mtime(xtr_p),
                "x_test.csv": file_mtime(xte_p),
                "y_train.csv": file_mtime(ytr_p),
            },
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    print(f"[Data] Loading CSVs from '{data_dir}'...")
    x_train_raw, x_test_raw, y_train_pm1, train_ids, test_ids = load_csv_data(data_dir)
    print(f"[Data] Raw shapes: X_train={x_train_raw.shape}, X_test={x_test_raw.shape}")

    cache_dir = cache_dir or os.path.join(data_dir, "..", "artifacts")
    ensure_dir(cache_dir)
    cache_key = build_cache_key()
    cache_npz = os.path.join(cache_dir, f"cache_{cache_key}.npz")

    if cache_data and os.path.exists(cache_npz):
        print(f"[Data] Cache hit: {cache_npz}")
        arrs = np.load(cache_npz)
        return DataBundle(arrs["X_train"], arrs["y_train"], arrs["X_test"], arrs["train_ids"], arrs["test_ids"], None)

    if not use_preprocessed:
        print("[Data] Using RAW features (no preprocessing).")
        Xtr, ytr = x_train_raw, y_train_pm1
        if oversample_ratio is not None:
            print(f"[Data] Oversampling positives to ratio {oversample_ratio:.2f}...")
            Xtr, ytr = prep.oversample_minority(Xtr, ytr, target_pos_ratio=oversample_ratio, rng=get_rng(seed))
            print(f"[Data] After oversample: X_train={Xtr.shape}, pos_ratio={(ytr==1).mean():.5f}")
        bundle = DataBundle(Xtr, ytr, x_test_raw, train_ids, test_ids, None)
    else:
        print("[Prep] Applying preprocessing pipeline...")
        Xtr, ytr, Xte, report = prep.preprocess_pipeline(
            x_train_raw,
            y_train_pm1,
            x_test_raw,
            cont_features=cont_features,
            strategy_cont="mean",
            target_pos_ratio=oversample_ratio,
            rng=get_rng(seed),
        )
        print(f"[Prep] Done. Shapes: X_train={Xtr.shape}, X_test={Xte.shape}")
        if oversample_ratio is not None:
            print(f"[Prep] Post-oversample pos_ratio={(ytr==1).mean():.5f}")
        bundle = DataBundle(Xtr, ytr, Xte, train_ids, test_ids, report)

    if cache_data:
        print(f"[Data] Writing cache: {cache_npz}")
        np.savez_compressed(cache_npz, X_train=bundle.X_train, y_train=bundle.y_train, X_test=bundle.X_test,
                             train_ids=bundle.train_ids, test_ids=bundle.test_ids)
    return bundle


# -------------------------------
# Model helpers
# -------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def predict_scores(model: str, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    z = X @ w
    if model in {"logistic", "reg_logistic"}:
        return sigmoid(z)
    return z


def fit_model(model: str, X: np.ndarray, y_pm1: np.ndarray, params: dict, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    y01 = _to01(y_pm1)
    if model == "mse_gd":
        w0 = params.get("w0", rng.normal(0, 0.01, size=X.shape[1]))
        w, loss = impl.mean_squared_error_gd(y_pm1, X, w0, int(params["max_iters"]), float(params["gamma"]))
        return w, float(loss)
    if model == "mse_sgd":
        w0 = params.get("w0", rng.normal(0, 0.01, size=X.shape[1]))
        w, loss = impl.mean_squared_error_sgd(y_pm1, X, w0, int(params["max_iters"]), float(params["gamma"]))
        return w, float(loss)
    if model == "least_squares":
        w, loss = impl.least_squares(y_pm1, X)
        return w, float(loss)
    if model == "ridge":
        w, loss = impl.ridge_regression(y_pm1, X, float(params["lambda"]))
        return w, float(loss)
    if model == "logistic":
        w0 = params.get("w0", rng.normal(0, 0.01, size=X.shape[1]))
        w, loss = impl.logistic_regression(y01, X, w0, int(params["max_iters"]), float(params["gamma"]))
        return w, float(loss)
    if model == "reg_logistic":
        w0 = params.get("w0", rng.normal(0, 0.01, size=X.shape[1]))
        w, loss = impl.reg_logistic_regression(y01, X, float(params["lambda"]), w0, int(params["max_iters"]), float(params["gamma"]))
        return w, float(loss)
    raise ValueError(f"Unknown model: {model}")


# -------------------------------
# CV, search, and evaluation
# -------------------------------

def k_fold_indices(n_samples: int, k: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]


def optimal_threshold_by_f1(y_true_pm1: np.ndarray, y_score: np.ndarray, is_prob: bool) -> float:
    if is_prob:
        thresholds = np.linspace(0.0, 1.0, 201)
    else:
        unique_scores = np.unique(y_score)
        thresholds = np.concatenate([[-math.inf], unique_scores, [math.inf]])
    best_f1, best_t = -1.0, 0.5 if is_prob else 0.0
    for t in thresholds:
        y_pred = np.where(y_score >= t, 1, -1)
        _, _, f1 = precision_recall_f1(y_true_pm1, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def evaluate_split(model: str, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, params: dict, rng: np.random.Generator) -> Dict[str, float]:
    w, _ = fit_model(model, X_tr, y_tr, params, rng)
    scores = predict_scores(model, X_va, w)
    is_prob = model in {"logistic", "reg_logistic"}
    thr = optimal_threshold_by_f1(y_va, scores, is_prob)
    y_pred = np.where(scores >= thr, 1, -1)
    acc = accuracy(y_va, y_pred)
    prec, rec, f1 = precision_recall_f1(y_va, y_pred)
    fpr, tpr = roc_curve(y_va, scores)
    roc_area = auc(fpr, tpr)
    pr_area = pr_auc(y_va, scores)
    ll = log_loss_from_probs(y_va, scores) if is_prob else float('nan')
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_area, "pr_auc": pr_area, "log_loss": ll, "threshold": thr}


def grid_search(model: str, X: np.ndarray, y: np.ndarray, param_grid: List[dict], k: int, seed: int | None, verbose: bool = True) -> Tuple[dict, Dict[str, float], List[dict]]:
    rng = get_rng(seed)
    splits = k_fold_indices(X.shape[0], k, rng)
    results = []
    if verbose:
        print(f"[Search] {model}: {len(param_grid)} configuration(s), {k}-fold CV")
    for i, params in enumerate(param_grid, start=1):
        if verbose:
            print(f"  [Search] {i}/{len(param_grid)} params={params}")
        fold_metrics = []
        for fi, (tr, va) in enumerate(splits, start=1):
            m = evaluate_split(model, X[tr], y[tr], X[va], y[va], params, rng)
            fold_metrics.append(m)
            if verbose:
                print("    [Fold {}] acc={:.5f} prec={:.5f} rec={:.5f} f1={:.5f} rocAUC={:.5f} prAUC={:.5f}".format(
                    fi, m["acc"], m["precision"], m["recall"], m["f1"], m["roc_auc"], m["pr_auc"]))
        avg = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
        if verbose:
            print("  [Search] avg -> acc={:.5f} prec={:.5f} rec={:.5f} f1={:.5f} rocAUC={:.5f} prAUC={:.5f}".format(
                avg["acc"], avg["precision"], avg["recall"], avg["f1"], avg["roc_auc"], avg["pr_auc"]))
        results.append({"params": params, "metrics": avg})

    def score_key(r):
        m = r["metrics"]
        return (m["f1"], m["pr_auc"], m["roc_auc"])  # higher is better

    best = max(results, key=score_key)
    if verbose:
        print(f"[Search] Best params: {best['params']}")
    return best["params"], best["metrics"], results


# -------------------------------
# Plotting
# -------------------------------

def plot_roc_pr(y_true_pm1: np.ndarray, scores: np.ndarray, out_prefix: str) -> Dict[str, float]:
    fpr, tpr = roc_curve(y_true_pm1, scores)
    roc_area = auc(fpr, tpr)
    recall, precision = precision_recall_curve(y_true_pm1, scores)
    pr_area = auc(recall, precision)

    plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_area:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5.2, 4.2))
    plt.plot(recall, precision, label=f"PR AUC={pr_area:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=150)
    plt.close()

    return {"roc_auc": float(roc_area), "pr_auc": float(pr_area)}


def plot_model_comparison(model_results: List[Tuple[str, Dict[str, float]]], out_path: str, metric: str = "f1") -> None:
    labels = [name for name, _ in model_results]
    values = [res[metric] for _, res in model_results]
    plt.figure(figsize=(6.0, 4.0))
    plt.bar(labels, values)
    plt.ylabel(metric.upper())
    plt.title(f"Model comparison ({metric.upper()})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------------
# CLI and main
# -------------------------------

def build_param_grid(args: argparse.Namespace, model: str) -> List[dict]:
    grid: List[dict] = []
    if model == "mse_gd":
        for g in args.mse_gammas:
            grid.append({"gamma": float(g), "max_iters": int(args.mse_iters)})
    elif model == "mse_sgd":
        for g in args.mse_gammas:
            grid.append({"gamma": float(g), "max_iters": int(args.mse_iters)})
    elif model == "least_squares":
        grid.append({})
    elif model == "ridge":
        for lam in args.ridge_lambdas:
            grid.append({"lambda": float(lam)})
    elif model == "logistic":
        for g in args.logit_gammas:
            grid.append({"gamma": float(g), "max_iters": int(args.logit_iters)})
    elif model == "reg_logistic":
        for lam in args.reg_lambdas:
            for g in args.logit_gammas:
                grid.append({"lambda": float(lam), "gamma": float(g), "max_iters": int(args.logit_iters)})
    else:
        raise ValueError(f"Unknown model: {model}")
    return grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EPFL ML Project 1 runner")
    # data
    p.add_argument("--data-dir", type=str, default="dataset")
    p.add_argument("--use-preprocessed", action="store_true")
    p.add_argument("--oversample", type=float, default=None, help="Target positive ratio (e.g., 0.35)")
    p.add_argument("--cont-features", type=int, nargs="*", default=[7, 8, 222, 226, 229, 253])
    p.add_argument("--cache-data", action="store_true")
    p.add_argument("--cache-dir", type=str, default=None)
    # experiment
    p.add_argument("--model", type=str, required=True, choices=["mse_gd", "mse_sgd", "least_squares", "ridge", "logistic", "reg_logistic"])
    p.add_argument("--compare-models", type=str, nargs="*", choices=["mse_gd", "mse_sgd", "least_squares", "ridge", "logistic", "reg_logistic"], help="Run multiple models and compare")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-plots", action="store_true")
    # hyperparams
    p.add_argument("--mse-gammas", type=float, nargs="*", default=[0.1, 0.05, 0.02, 0.01])
    p.add_argument("--mse-iters", type=int, default=2000)
    p.add_argument("--ridge-lambdas", type=float, nargs="*", default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    p.add_argument("--logit-gammas", type=float, nargs="*", default=[0.1, 0.05, 0.02, 0.01])
    p.add_argument("--logit-iters", type=int, default=2000)
    p.add_argument("--reg-lambdas", type=float, nargs="*", default=[1e-4, 1e-3, 1e-2, 1e-1])
    # submission (single-model mode)
    p.add_argument("--make-submission", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_root = "compare" if args.compare_models else args.model
    exp_name = f"{exp_root}_{args.tag}_{timestamp}" if args.tag else f"{exp_root}_{timestamp}"
    out_dir = os.path.join(args.results_dir, exp_name)
    ensure_dir(out_dir)

    print(f"[Run] Model={args.model} | Compare={args.compare_models is not None} | folds={args.k_folds} | seed={args.seed}")
    print(f"[Run] Preprocessed={'YES' if args.use_preprocessed else 'NO'} | Oversample={args.oversample}")

    data = load_data(
        data_dir=args.data_dir,
        use_preprocessed=args.use_preprocessed,
        seed=args.seed,
        oversample_ratio=args.oversample,
        cont_features=args.cont_features,
        cache_data=args.cache_data,
        cache_dir=args.cache_dir,
    )

    # Multi-model path
    if args.compare_models:
        models = args.compare_models
        comparison = []
        per_model = {}
        for model_name in models:
            print(f"[Compare] Running {model_name}...")
            grid = build_param_grid(args, model_name)
            best_params, best_metrics, _all = grid_search(model_name, data.X_train, data.y_train, grid, args.k_folds, args.seed)
            rng = get_rng(args.seed)
            w_best, _ = fit_model(model_name, data.X_train, data.y_train, best_params, rng)
            scores_tr = predict_scores(model_name, data.X_train, w_best)
            is_prob = model_name in {"logistic", "reg_logistic"}
            thr = optimal_threshold_by_f1(data.y_train, scores_tr, is_prob)
            y_pred_tr = np.where(scores_tr >= thr, 1, -1)
            final_metrics = {
                "acc": accuracy(data.y_train, y_pred_tr),
                "precision": precision_recall_f1(data.y_train, y_pred_tr)[0],
                "recall": precision_recall_f1(data.y_train, y_pred_tr)[1],
                "f1": precision_recall_f1(data.y_train, y_pred_tr)[2],
                "log_loss": log_loss_from_probs(data.y_train, scores_tr) if is_prob else float('nan'),
            }
            if not args.no_plots:
                aucs = plot_roc_pr(data.y_train, scores_tr, os.path.join(out_dir, f"{model_name}_train"))
                final_metrics.update(aucs)
            comparison.append((model_name, final_metrics))
            per_model[model_name] = {"best_params": best_params, "cv_best_metrics": best_metrics, "threshold": float(thr), "final_train_metrics": final_metrics}

        if not args.no_plots and len(comparison) > 1:
            plot_model_comparison(comparison, os.path.join(out_dir, "compare_f1.png"), metric="f1")
            plot_model_comparison(comparison, os.path.join(out_dir, "compare_roc_auc.png"), metric="roc_auc")
            plot_model_comparison(comparison, os.path.join(out_dir, "compare_pr_auc.png"), metric="pr_auc")

        summary = {"models": models, "use_preprocessed": bool(args.use_preprocessed), "oversample": args.oversample, "k_folds": int(args.k_folds), "seed": int(args.seed), "n_train": int(data.X_train.shape[0]), "n_features": int(data.X_train.shape[1]), "details": per_model}
        with open(os.path.join(out_dir, "comparison_summary.json"), "w") as f:
            json.dump(round_floats(summary, 5), f, indent=2)
        print("[Compare] F1:", {name: round(mets["f1"], 5) for name, mets in comparison})
        return

    # Single-model path
    grid = build_param_grid(args, args.model)
    best_params, best_metrics, all_results = grid_search(args.model, data.X_train, data.y_train, grid, args.k_folds, args.seed)
    rng = get_rng(args.seed)
    w_best, _ = fit_model(args.model, data.X_train, data.y_train, best_params, rng)
    scores_tr = predict_scores(args.model, data.X_train, w_best)
    is_prob = args.model in {"logistic", "reg_logistic"}
    thr = optimal_threshold_by_f1(data.y_train, scores_tr, is_prob)
    y_pred_tr = np.where(scores_tr >= thr, 1, -1)
    final_metrics = {
        "acc": accuracy(data.y_train, y_pred_tr),
        "precision": precision_recall_f1(data.y_train, y_pred_tr)[0],
        "recall": precision_recall_f1(data.y_train, y_pred_tr)[1],
        "f1": precision_recall_f1(data.y_train, y_pred_tr)[2],
        "log_loss": log_loss_from_probs(data.y_train, scores_tr) if is_prob else float('nan'),
    }
    if not args.no_plots:
        aucs = plot_roc_pr(data.y_train, scores_tr, os.path.join(out_dir, "train"))
        final_metrics.update(aucs)

    meta = {"model": args.model, "best_params": best_params, "cv_best_metrics": best_metrics, "threshold": float(thr), "final_train_metrics": final_metrics, "use_preprocessed": bool(args.use_preprocessed), "oversample": args.oversample, "k_folds": int(args.k_folds), "seed": int(args.seed), "cont_features": list(args.cont_features) if args.cont_features is not None else None, "preprocess_report_keys": list(data.preprocess_report.keys()) if data.preprocess_report is not None else None, "n_train": int(data.X_train.shape[0]), "n_features": int(data.X_train.shape[1])}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(round_floats(meta, 5), f, indent=2)
    with open(os.path.join(out_dir, "cv_results.json"), "w") as f:
        json.dump(round_floats(all_results, 5), f, indent=2)

    if args.make_submission:
        scores_te = predict_scores(args.model, data.X_test, w_best)
        y_pred_te = np.where(scores_te >= thr, 1, -1).astype(int)
        sub_path = os.path.join(out_dir, f"submission_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        create_csv_submission(data.test_ids, y_pred_te, sub_path)
        print("[Submit] Wrote:", sub_path)

    print("\n[Done] Best params:", best_params)
    print("[Done] CV (avg):", {k: ("{:.5f}".format(v) if isinstance(v, float) and not np.isnan(v) else v) for k, v in best_metrics.items()})
    print("[Done] Final train:", {k: ("{:.5f}".format(v) if isinstance(v, float) and not np.isnan(v) else v) for k, v in final_metrics.items()})


if __name__ == "__main__":
    main()


