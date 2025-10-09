import eda as eda
import numpy as np
import os
import json

# ============================================================
# Utilities
# ============================================================

def _remap_indices_through_keep(indices, keep):
    """
    Given original indices and a KEEP list (sorted indices kept),
    return indices remapped to the new (post-selection) coordinate system.
    Any index not in KEEP is dropped.
    """
    pos = {orig_j: new_j for new_j, orig_j in enumerate(keep)}
    return [pos[j] for j in indices if j in pos]


def _complement(n_cols, keep_list):
    """Return the complement (as a sorted list) of keep_list in range(n_cols)."""
    keep_set = set(keep_list)
    return [j for j in range(n_cols) if j not in keep_set]


# ============================================================
# Class imbalance (keep for per-fold use; NOT called in global pipeline)
# ============================================================

def oversample_minority(X, y, target_pos_ratio=0.5, rng=None):
    """
    Random oversampling by duplicating minority (y==1) examples until
    the positive ratio ~= target_pos_ratio. Works with y in {-1, 1}.
    Returns new X, y (shuffled).
    """
    assert X.shape[0] == y.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    pos_mask = (y == 1)
    neg_mask = (y == -1)
    n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())

    current_ratio = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
    if current_ratio >= target_pos_ratio or n_pos == 0:
        idx = rng.permutation(X.shape[0])
        return X[idx], y[idx]

    n_pos_needed = int(np.ceil(target_pos_ratio * n_neg / (1.0 - target_pos_ratio)))
    n_to_add = max(0, n_pos_needed - n_pos)

    pos_idx = np.where(pos_mask)[0]
    add_idx = rng.choice(pos_idx, size=n_to_add, replace=True)

    X_bal = np.concatenate([X, X[add_idx]], axis=0)
    y_bal = np.concatenate([y, y[add_idx]], axis=0)

    perm = rng.permutation(X_bal.shape[0])
    return X_bal[perm], y_bal[perm]


# ============================================================
# Feature dropping
# ============================================================

def drop_missing_features(x: np.ndarray, threshold: float = 0.5):
    """
    Drop features (columns) with more than threshold fraction of missing values.

    Returns:
        x_new, kept_features (orig indices), dropped_features (orig indices)
    """
    n_samples = x.shape[0]
    missing_fraction = np.sum(np.isnan(x), axis=0) / max(1, n_samples)

    kept_features = [j for j in range(x.shape[1]) if missing_fraction[j] <= threshold]
    dropped_features = [j for j in range(x.shape[1]) if missing_fraction[j] > threshold]
    x_new = x[:, kept_features]
    return x_new, kept_features, dropped_features


def drop_low_variance_features(x: np.ndarray, threshold: float = 0.995):
    """
    Drop constant and near-constant features using EDA utilities.
    Returns:
        x_new, keep_idx (relative to x), info dict
    """
    consts = eda.find_constant_features(x)
    near_consts = eda.find_near_constant_features(x, threshold)
    drop_idx = sorted(list(set(consts + near_consts)))
    keep_idx = [j for j in range(x.shape[1]) if j not in drop_idx]
    return x[:, keep_idx], keep_idx, {"constant": consts, "near_constant": near_consts}

def drop_duplicate_features(x: np.ndarray):
    n_features = x.shape[1]
    seen = {}
    keep, dup_map = [], {}
    for j in range(n_features):
        col = x[:, j]
        # treat NaNs as a fixed sentinel
        key_vals = np.nan_to_num(col, nan=1.23456789e308)
        # bytes key is robust and fast to hash
        key = key_vals.tobytes()
        if key in seen:
            dup_map[j] = seen[key]
        else:
            seen[key] = j
            keep.append(j)
    return x[:, keep], keep, dup_map


# ============================================================
# Imputation (split into fit/apply to avoid leakage)
# ============================================================

def fit_imputer(x: np.ndarray, cont_features: list, strategy_cont="mean", strategy_cat="mode"):
    """
    Learn per-column fill values.
    Returns:
        imp_cont: dict {j: fill_value} for continuous columns
        imp_cat : dict {j: fill_value} for categorical columns
    """
    n_features = x.shape[1]
    cont_set = set(cont_features)
    imp_cont, imp_cat = {}, {}

    for j in range(n_features):
        col = x[:, j]
        mask = np.isnan(col)

        if not np.any(mask):
            # record a consistent value anyway
            if j in cont_set:
                imp_cont[j] = float(np.mean(col)) if col.size else 0.0
            else:
                # categorical strategy (currently only 'mode')
                if strategy_cat == "mode":
                    vals, counts = np.unique(col, return_counts=True)
                    imp_cat[j] = float(vals[np.argmax(counts)]) if vals.size > 0 else 0.0
                else:
                    raise ValueError("strategy_cat must be 'mode'")
            continue

        if j in cont_set:
            if strategy_cont == "mean":
                fill_val = np.nanmean(col)
            elif strategy_cont == "median":
                fill_val = np.nanmedian(col)
            else:
                raise ValueError("strategy_cont must be 'mean' or 'median'")
            if np.isnan(fill_val):
                fill_val = 0.0
            imp_cont[j] = float(fill_val)
        else:
            if strategy_cat == "mode":
                values, counts = np.unique(col[~mask], return_counts=True)
                fill_val = float(values[np.argmax(counts)]) if values.size > 0 else 0.0
                imp_cat[j] = float(fill_val)
            else:
                raise ValueError("strategy_cat must be 'mode'")

    return imp_cont, imp_cat


def apply_imputer(x: np.ndarray, imp_cont: dict, imp_cat: dict):
    """
    Apply learned imputer values in-place, returns a copy.
    """
    x_imputed = x.copy()
    n_features = x.shape[1]
    for j in range(n_features):
        col = x_imputed[:, j]
        mask = np.isnan(col)
        if np.any(mask):
            if j in imp_cont:
                x_imputed[mask, j] = imp_cont[j]
            elif j in imp_cat:
                x_imputed[mask, j] = imp_cat[j]
            else:
                x_imputed[mask, j] = 0.0
    return x_imputed



# ============================================================
# Standardization (fit/apply)
# ============================================================

def fit_standardizer(x: np.ndarray, cont_features: list):
    """
    Compute mean and std for continuous features.
    Returns:
        means : dict {feature_idx: mean}
        stds  : dict {feature_idx: std>=1e-12}
    """
    means, stds = {}, {}
    for j in cont_features:
        col = x[:, j]
        m = float(np.mean(col)) if col.size else 0.0
        s = float(np.std(col))
        means[j] = m
        stds[j] = s if s > 1e-12 else 1.0  # avoid div/0
    return means, stds


def apply_standardizer(x: np.ndarray, cont_features: list, means: dict, stds: dict):
    """
    Apply standardization to continuous features.
    """
    x_scaled = x.copy()
    for j in cont_features:
        j_int = int(j)
        m = means.get(j_int, 0.0)   # default if a key is missing
        s = stds.get(j_int, 1.0)
        x_scaled[:, j_int] = (x_scaled[:, j_int] - m) / s
    return x_scaled


# ============================================================
# One-hot (fit/transform)
# ============================================================

def fit_one_hot(x: np.ndarray, cat_features: list, min_count: int = 50, max_categories: int = 50):
    """
    Learn one-hot specs for categorical features:
      - Keeps up to max_categories most frequent values
      - Groups rare categories (count < min_count) into 'OTHER' bucket

    Returns:
        specs : dict {feature_idx: {"values": np.array, "other": bool}}
    """
    specs = {}
    for j in cat_features:
        col = x[:, j]
        vals, counts = np.unique(col, return_counts=True)
        order = np.argsort(counts)[::-1]
        vals, counts = vals[order], counts[order]

        keep_mask = counts >= min_count
        kept_vals = vals[keep_mask]
        if kept_vals.size > max_categories:
            kept_vals = kept_vals[:max_categories]

        specs[j] = {
            "values": kept_vals.astype(float),
            "other": kept_vals.size < vals.size
        }
    return specs


def transform_one_hot(x: np.ndarray, cat_features: list, specs: dict):
    """
    Apply one-hot encoding using fitted specs.
    Returns:
        X_oh : numpy array with concatenated one-hot features
    """
    n = x.shape[0]
    encoded_blocks = []

    for j in cat_features:
        col = x[:, j]
        values = specs[j]["values"]
        other_flag = specs[j]["other"]
        width = values.size + (1 if other_flag else 0)
        oh = np.zeros((n, width))

        # Vectorized-ish fill
        if values.size > 0:
            # For each kept value, set the right column
            for k, v in enumerate(values):
                hits = (col == v)
                oh[hits, k] = 1.0
        if other_flag:
            # Any row not matching any kept value => OTHER
            if values.size > 0:
                any_match = np.zeros(n, dtype=bool)
                for v in values:
                    any_match |= (col == v)
                oh[~any_match, -1] = 1.0
            else:
                # No kept values -> everything goes to OTHER
                oh[:, -1] = 1.0

        encoded_blocks.append(oh)

    if encoded_blocks:
        return np.concatenate(encoded_blocks, axis=1)
    else:
        return np.empty((n, 0))


# ============================================================
# Option A: Functional + STATE
# ============================================================

def fit_preprocessor(
    X_train: np.ndarray,
    cont_features_orig: list,
    missing_threshold: float = 0.5,
    drop_low_var: bool = True,
    lowvar_threshold: float = 0.995,
    cont_strategy: str = "mean",
    cat_strategy: str = "mode",
    onehot_min_count: int = 50,
    onehot_max_categories: int = 50,
    drop_duplicates: bool = False, 
):
    """
    Fit the preprocessing state on TRAIN ONLY.
    Returns:
        state: dict containing everything needed to transform any split
    """
    state = {
        # config
        "missing_threshold": float(missing_threshold),
        "drop_low_var": bool(drop_low_var),
        "lowvar_threshold": float(lowvar_threshold),
        "cont_strategy": cont_strategy,
        "cat_strategy": cat_strategy,
        "onehot_min_count": int(onehot_min_count),
        "onehot_max_categories": int(onehot_max_categories),

        # learned selections
        "kept_after_missing": None,   # absolute indices w.r.t. original X_train
        "kept_after_lowvar_rel": None,  # indices relative to post-missing
        "kept_after_dups_rel": None,
        "final_kept_abs": None,       # absolute original indices of columns that survived both drops

        # remapped feature groups (relative to post-lowvar)
        "cont_features": None,
        "cat_features": None,

        # imputer
        "imp_cont": None,
        "imp_cat": None,

        # standardizer
        "std_means": None,
        "std_stds": None,

        # one-hot
        "onehot_specs": None,

        # bookkeeping
        "final_dim_before_oh": None,
    }

    # 1) Drop features > missing_threshold
    X_miss, kept_after_missing, _ = drop_missing_features(X_train, threshold=missing_threshold)

    # 2) Drop low variance (optional)
    if drop_low_var:
        X_lv, kept_lowvar_rel, _ = drop_low_variance_features(X_miss, threshold=lowvar_threshold)
    else:
        X_lv, kept_lowvar_rel = X_miss, list(range(X_miss.shape[1]))

    # Compose final absolute kept indices (original coordinates)
    final_kept_abs = [kept_after_missing[i] for i in kept_lowvar_rel]

    # 3) Remap continuous feature indices through selections
    cont_after_missing = _remap_indices_through_keep(cont_features_orig, kept_after_missing)
    cont_after_lowvar = _remap_indices_through_keep(cont_after_missing, kept_lowvar_rel)

    # 4) Define categorical as complement at this stage
    n_cols_lv = X_lv.shape[1]
    cat_after_lowvar = _complement(n_cols_lv, cont_after_lowvar)

    # 5) Fit imputer on X_lv (train only)
    imp_cont, imp_cat = fit_imputer(X_lv, cont_after_lowvar, strategy_cont=cont_strategy, strategy_cat=cat_strategy)

    # 6) Impute train to compute scaler safely
    X_imp = apply_imputer(X_lv, imp_cont, imp_cat)

    # 6) Drop duplicate features (optional, Level 3)
    if drop_duplicates:
        X_dd, kept_dups_rel, dup_map = drop_duplicate_features(X_imp)
    else:
        X_dd, kept_dups_rel, dup_map = X_imp, list(range(X_imp.shape[1])), {}

    # Remap continuous indices through the duplicate keep
    cont_after_dups = _remap_indices_through_keep(cont_after_lowvar, kept_dups_rel)
    n_cols_dd = X_dd.shape[1]
    cat_after_dups = _complement(n_cols_dd, cont_after_dups)

    # 7) Fit standardizer on post-dup imputed matrix
    std_means, std_stds = fit_standardizer(X_dd, cont_after_dups)

    # 8) Fit one-hot specs on post-dup imputed matrix
    oh_specs = fit_one_hot(
        X_dd, cat_after_dups,
        min_count=onehot_min_count,
        max_categories=onehot_max_categories
    )
    # ensure specs keys are plain ints
    oh_specs = {int(k): {"values": v["values"], "other": v["other"]}
                for k, v in oh_specs.items()}
    state["onehot_specs"] = oh_specs

    # Populate state
    state["kept_after_missing"] = kept_after_missing
    state["kept_after_lowvar_rel"] = kept_lowvar_rel
    state["final_kept_abs"] = final_kept_abs

    state["cont_features"] = cont_after_lowvar
    state["cat_features"] = cat_after_lowvar

    state["imp_cont"] = imp_cont
    state["imp_cat"] = imp_cat

    state["std_means"] = std_means
    state["std_stds"] = std_stds

    state["onehot_specs"] = oh_specs
    state["final_dim_before_oh"] = int(n_cols_lv)

    return state


def _to_int_list(seq):
    return [int(x) for x in seq]

def transform_with_state(X: np.ndarray, state: dict):
    """
    Apply the learned state to any split (train, val, test).
    Returns:
        X_proc: standardized continuous || one-hot categorical (concatenated)
    """
    # 1) Missing-drop
    kept_after_missing = _to_int_list(state["kept_after_missing"])
    X1 = X[:, kept_after_missing]

    # 2) Low-variance drop (relative to X1)
    kept_lowvar_rel = _to_int_list(state["kept_after_lowvar_rel"])
    X2 = X1[:, kept_lowvar_rel]

    # 3) Impute
    X2_imp = apply_imputer(X2, state["imp_cont"], state["imp_cat"])

    # 4) Duplicate-drop (relative to X2_imp)
    kept_dups_rel = state.get("kept_after_dups_rel")
    if kept_dups_rel is None:
        kept_dups_rel = list(range(X2_imp.shape[1]))   # keep all if no dup-drop used
    else:
        kept_dups_rel = _to_int_list(kept_dups_rel)
    X3 = X2_imp[:, kept_dups_rel]

    # 5) Standardize continuous (indices are relative to X3)
    cont_feats = _to_int_list(state["cont_features"])
    # normalize scaler dict keys to plain ints too (in case they were np.int64)
    std_means = {int(k): float(v) for k, v in state["std_means"].items()}
    std_stds  = {int(k): float(v) for k, v in state["std_stds"].items()}

    X3_scaled = apply_standardizer(X3, cont_feats, std_means, std_stds)

    # 6) One-hot categorical (relative to X3)
    cat_feats = [int(j) for j in state["cat_features"]]

    # normalize specs keys to int (paranoia + robustness)
    specs = {int(k): v for k, v in state["onehot_specs"].items()}

    # assert mapping consistency; if any cat index is missing, drop it with a warning
    missing = [j for j in cat_feats if j not in specs]
    if missing:
        # You can print or log; here we prune them to avoid a hard crash
        print(f"[warn] onehot specs missing for categorical indices {missing}; skipping those columns")
        cat_feats = [j for j in cat_feats if j in specs]
        
    X_cat = transform_one_hot(X3_scaled, cat_feats, state["onehot_specs"])

    # 7) Concatenate: continuous (scaled) + categorical (one-hot)
    if len(cont_feats) > 0:
        X_cont = X3_scaled[:, cont_feats]
    else:
        X_cont = np.empty((X3_scaled.shape[0], 0))

    return X_cont if X_cat.size == 0 else np.concatenate([X_cont, X_cat], axis=1)


def preprocess_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cont_features: list,
    strategy_cont: str = "mean",
    strategy_cat: str = "mode",
    missing_threshold: float = 0.5,
    drop_low_var: bool = True,
    lowvar_threshold: float = 0.995,
    onehot_min_count: int = 50,
    onehot_max_categories: int = 50,
    drop_duplicates: bool = False,
):
    """
    Convenience wrapper:
      - FIT on train only -> state
      - TRANSFORM train & test with that state
    Returns:
      X_train_proc, y_train, X_test_proc, state
    """
    state = fit_preprocessor(
        X_train,
        cont_features_orig=cont_features,
        missing_threshold=missing_threshold,
        drop_low_var=drop_low_var,
        lowvar_threshold=lowvar_threshold,
        cont_strategy=strategy_cont,
        cat_strategy=strategy_cat,
        onehot_min_count=onehot_min_count,
        onehot_max_categories=onehot_max_categories,
        drop_duplicates=drop_duplicates,
    )

    Xtr = transform_with_state(X_train, state)
    Xte = transform_with_state(X_test, state)
    return Xtr, y_train, Xte, state


# ============================================================
# RUN ALL LEVELS OF PREPRO
# ============================================================
LEVELS = {
    "level0": dict(
        missing_threshold=0.99, drop_low_var=False,
        strategy_cont="mean", strategy_cat="mode",
        onehot_min_count=0, onehot_max_categories=0,
        drop_duplicates=False,
    ),
    "level1": dict(
        missing_threshold=0.50, drop_low_var=True, lowvar_threshold=0.999,
        strategy_cont="mean", strategy_cat="mode",
        onehot_min_count=0, onehot_max_categories=0,
        drop_duplicates=False,
    ),
    "level2": dict(
        missing_threshold=0.50, drop_low_var=True, lowvar_threshold=0.995,
        strategy_cont="mean", strategy_cat="mode",
        onehot_min_count=50, onehot_max_categories=50,
        drop_duplicates=False,
    ),
    "level3": dict(
        missing_threshold=0.40, drop_low_var=True, lowvar_threshold=0.990,
        strategy_cont="median", strategy_cat="mode",     # cat impute via mode
        onehot_min_count=10, onehot_max_categories=100,
        drop_duplicates=True,                            # drop duplicate features
    ),
}


def save_csvs_and_state(Xtr, ytr, Xte, state, outdir):
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(f"{outdir}/x_train.csv", Xtr, delimiter=",", fmt="%.6f")
    np.savetxt(f"{outdir}/x_test.csv",  Xte, delimiter=",", fmt="%.6f")
    np.savetxt(f"{outdir}/y_train.csv", ytr, delimiter=",", fmt="%.0f")
    json_state = {
        k: state[k] for k in [
            "missing_threshold","drop_low_var","lowvar_threshold","cont_strategy",
            "onehot_min_count","onehot_max_categories","kept_after_missing",
            "kept_after_lowvar_rel","final_kept_abs","cont_features","cat_features",
            "imp_cont","imp_cat","std_means","std_stds"
        ]
    }
    json_state["onehot_specs"] = {
        str(k): {"values": state["onehot_specs"][k]["values"].tolist(),
                 "other": state["onehot_specs"][k]["other"]}
        for k in state["onehot_specs"]
    }
    with open(f"{outdir}/preproc_state.json", "w") as f:
        json.dump(json_state, f, indent=2)

def run_all_levels(x_train, x_test, y_train, cont_features_orig, only = None):

    levels = LEVELS if only is None else {k: LEVELS[k] for k in only}
    #Run preprocessing pipelines for all levels
    for name, cfg in levels.items():
        Xtr, ytr, Xte, state = preprocess_pipeline(
            x_train, y_train, x_test,
            cont_features=cont_features_orig,
            strategy_cont=cfg["strategy_cont"],
            strategy_cat=cfg.get("strategy_cat", "mode"),
            missing_threshold=cfg["missing_threshold"],
            drop_low_var=cfg["drop_low_var"],
            lowvar_threshold=cfg.get("lowvar_threshold", 0.995),
            onehot_min_count=cfg["onehot_min_count"],
            onehot_max_categories=cfg["onehot_max_categories"],
            drop_duplicates=cfg.get("drop_duplicates", False),
        )

        # OVERSAMPLING for balancing
        if name in ["level2", "level3", "level4"]:
            print(f"  -> Oversampling positives to reach 50% ratio...")
            Xtr, ytr = oversample_minority(Xtr, ytr, target_pos_ratio=0.35)

        save_csvs_and_state(Xtr, ytr, Xte, state, f"preprocessed/{name}")
        print(f"[OK] {name}: train {Xtr.shape}, test {Xte.shape}")
# ============================================================
# main
# ============================================================

def main():
    from helpers import load_csv_data  # stdlib + numpy only

    # 1) Load raw data
    x_train, x_test, y_train, _, _ = load_csv_data("dataset")
    print(f"Raw: train {x_train.shape}, test {x_test.shape}")

    # 2) Detect feature types; remove suspicious 'continuous' (dates/IDs) from continuous set
    categorical, continuous, binary = eda.detect_feature_types_refined(x_train)
    suspect_features = eda.detect_date_or_id_features(x_train[:, continuous])

    removed_from_continuous = []
    for idx in suspect_features:
        if idx in continuous:
            removed_from_continuous.append(idx)
            continuous.remove(idx)

    cont_features_orig = continuous  # original-coordinate indices

    # 3) Run all levels and export each to its own folder
    os.makedirs("preprocessed", exist_ok=True)
    run_all_levels(x_train, x_test, y_train, cont_features_orig, only=["level3"])

    print("\n[Done] All levels exported under ./preprocessed/")
    print("Levels:", ", ".join(LEVELS.keys()))

if __name__ == "__main__":
    main()
