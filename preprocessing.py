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

    pos_mask = y == 1
    neg_mask = y == -1
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


def fit_imputer(
    x: np.ndarray, cont_features: list, strategy_cont="mean", strategy_cat="mode"
):
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
                    imp_cat[j] = (
                        float(vals[np.argmax(counts)]) if vals.size > 0 else 0.0
                    )
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
        m = means.get(j_int, 0.0)  # default if a key is missing
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

def fit_target_encoding(
    x: np.ndarray,
    y: np.ndarray,
    cat_features: list,
    min_count: int = 50,
    max_categories: int = 50,
    smoothing: float = 1.0,
):
    """
    Learn target encoding for categorical features:
      - For each category, compute mean target value
      - Apply smoothing to handle rare categories
      - Groups rare categories (count < min_count) into 'OTHER' bucket
      - Keeps up to max_categories most frequent values

    Returns:
        specs : dict {feature_idx: {"encodings": dict, "other_encoding": float, "global_mean": float}}
    """
    specs = {}
    global_mean = np.mean(y)

    for j in cat_features:
        col = x[:, j]
        vals, counts = np.unique(col, return_counts=True)
        order = np.argsort(counts)[::-1]
        vals, counts = vals[order], counts[order]

        # Keep frequent categories
        keep_mask = counts >= min_count
        kept_vals = vals[keep_mask]
        if kept_vals.size > max_categories:
            kept_vals = kept_vals[:max_categories]

        # Compute target encodings for kept categories
        encodings = {}
        for val in kept_vals:
            mask = col == val
            if mask.sum() > 0:
                # Target encoding with smoothing
                category_mean = np.mean(y[mask])
                n_samples = mask.sum()
                # Smoothing: blend category mean with global mean
                encoding = (category_mean * n_samples + global_mean * smoothing) / (
                    n_samples + smoothing
                )
                encodings[val] = encoding

        # Compute encoding for "OTHER" category (rare values)
        other_mask = ~np.isin(col, kept_vals)
        if other_mask.sum() > 0:
            other_mean = np.mean(y[other_mask])
            other_n = other_mask.sum()
            other_encoding = (other_mean * other_n + global_mean * smoothing) / (
                other_n + smoothing
            )
        else:
            other_encoding = global_mean

        specs[j] = {
            "encodings": encodings,
            "other_encoding": other_encoding,
            "global_mean": global_mean,
            "kept_vals": kept_vals,
            "has_other": kept_vals.size < vals.size,
        }
    return specs


def transform_target_encoding(x: np.ndarray, cat_features: list, specs: dict):
    """
    Apply target encoding using fitted specs.
    Returns:
        X_te : numpy array with target-encoded features (one column per categorical feature)
    """
    n = x.shape[0]
    encoded_features = []

    for j in cat_features:
        col = x[:, j]
        spec = specs[j]
        encodings = spec["encodings"]
        other_encoding = spec["other_encoding"]
        global_mean = spec["global_mean"]
        kept_vals = spec["kept_vals"]
        has_other = spec["has_other"]

        # Initialize with global mean (fallback for unseen categories)
        encoded_col = np.full(n, global_mean)

        # Apply encodings for kept categories
        for val, encoding in encodings.items():
            mask = col == val
            encoded_col[mask] = encoding

        # Apply encoding for "OTHER" category
        if has_other:
            other_mask = ~np.isin(col, kept_vals)
            encoded_col[other_mask] = other_encoding

        encoded_features.append(encoded_col.reshape(-1, 1))

    if encoded_features:
        return np.concatenate(encoded_features, axis=1)
    else:
        return np.empty((n, 0))


# ============================================================
# Option A: Functional + STATE
# ============================================================


def fit_preprocessor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cont_features_orig: list,
    missing_threshold: float = 0.5,
    drop_low_var: bool = True,
    lowvar_threshold: float = 0.995,
    cont_strategy: str = "mean",
    cat_strategy: str = "mode",
    target_encoding_min_count: int = 50,
    target_encoding_max_categories: int = 50,
    target_encoding_smoothing: float = 1.0,
    drop_duplicates: bool = False,
):
    """
    Fit the preprocessing state on TRAIN ONLY.
    Returns a dict containing everything needed to transform any split.
    """
    state = {
        # --- config ---
        "missing_threshold": float(missing_threshold),
        "drop_low_var": bool(drop_low_var),
        "lowvar_threshold": float(lowvar_threshold),
        "cont_strategy": cont_strategy,
        "cat_strategy": cat_strategy,
        "target_encoding_min_count": int(target_encoding_min_count),
        "target_encoding_max_categories": int(target_encoding_max_categories),
        "target_encoding_smoothing": float(target_encoding_smoothing),
        # --- learned selections ---
        "kept_after_missing": None,
        "kept_after_lowvar_rel": None,
        "kept_after_dups_rel": None,
        "final_kept_abs": None,
        # --- groups ---
        "cont_features": None,
        "cat_features": None,
        # --- learned parameters ---
        "imp_cont": None,
        "imp_cat": None,
        "std_means": None,
        "std_stds": None,
        "onehot_specs": None,
        # --- bookkeeping ---
        "final_dim_before_oh": None,
    }

    # 1️⃣ Drop features with too many missing values
    X_miss, kept_after_missing, _ = drop_missing_features(
        X_train, threshold=missing_threshold
    )

    # 2️⃣ Drop low-variance features (optional)
    if drop_low_var:
        X_lv, kept_lowvar_rel, _ = drop_low_variance_features(
            X_miss, threshold=lowvar_threshold
        )
    else:
        X_lv, kept_lowvar_rel = X_miss, list(range(X_miss.shape[1]))

    # 3️⃣ (NEW POSITION) Drop duplicate features right after low variance
    if drop_duplicates:
        X_dd, kept_dups_rel, dup_map = drop_duplicate_features(X_lv)
    else:
        X_dd, kept_dups_rel, dup_map = X_lv, list(range(X_lv.shape[1])), {}

    # --- Compose absolute kept indices relative to original columns ---
    final_kept_abs = [kept_after_missing[i] for i in kept_lowvar_rel]
    final_kept_abs = [final_kept_abs[i] for i in kept_dups_rel]

    # 4️⃣ Remap continuous indices through all previous selections
    cont_after_missing = _remap_indices_through_keep(
        cont_features_orig, kept_after_missing
    )
    cont_after_lowvar = _remap_indices_through_keep(cont_after_missing, kept_lowvar_rel)
    cont_after_dups = _remap_indices_through_keep(cont_after_lowvar, kept_dups_rel)

    n_cols_dd = X_dd.shape[1]
    cat_after_dups = _complement(n_cols_dd, cont_after_dups)

    # 5️⃣ Fit imputers on de-duplicated matrix
    imp_cont, imp_cat = fit_imputer(
        X_dd, cont_after_dups, strategy_cont=cont_strategy, strategy_cat=cat_strategy
    )

    # 6️⃣ Apply imputation once to compute standardization safely
    X_imp = apply_imputer(X_dd, imp_cont, imp_cat)

    # 7️⃣ Fit standardizer on continuous features
    std_means, std_stds = fit_standardizer(X_imp, cont_after_dups)

    # 8️⃣ Fit target encoding specs on categorical features
    te_specs = fit_target_encoding(
        X_imp,
        y_train,
        cat_after_dups,
        min_count=target_encoding_min_count,
        max_categories=target_encoding_max_categories,
        smoothing=target_encoding_smoothing,
    )
    te_specs = {int(k): v for k, v in te_specs.items()}

    # --- Store everything ---
    state.update(
        {
            "kept_after_missing": kept_after_missing,
            "kept_after_lowvar_rel": kept_lowvar_rel,
            "kept_after_dups_rel": kept_dups_rel,
            "final_kept_abs": final_kept_abs,
            "cont_features": cont_after_dups,
            "cat_features": cat_after_dups,
            "imp_cont": imp_cont,
            "imp_cat": imp_cat,
            "std_means": std_means,
            "std_stds": std_stds,
            "target_encoding_specs": te_specs,
            "final_dim_before_oh": int(n_cols_dd),
        }
    )
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
        kept_dups_rel = list(range(X2_imp.shape[1]))  # keep all if no dup-drop used
    else:
        kept_dups_rel = _to_int_list(kept_dups_rel)
    X3 = X2_imp[:, kept_dups_rel]

    # 5) Standardize continuous (indices are relative to X3)
    cont_feats = _to_int_list(state["cont_features"])
    # normalize scaler dict keys to plain ints too (in case they were np.int64)
    std_means = {int(k): float(v) for k, v in state["std_means"].items()}
    std_stds = {int(k): float(v) for k, v in state["std_stds"].items()}

    X3_scaled = apply_standardizer(X3, cont_feats, std_means, std_stds)

    # 6) Target encoding categorical (relative to X3)
    cat_feats = [int(j) for j in state["cat_features"]]

    # normalize specs keys to int (paranoia + robustness)
    specs = {int(k): v for k, v in state["target_encoding_specs"].items()}

    # assert mapping consistency; if any cat index is missing, drop it with a warning
    missing = [j for j in cat_feats if j not in specs]
    if missing:
        # You can print or log; here we prune them to avoid a hard crash
        print(
            f"[warn] target encoding specs missing for categorical indices {missing}; skipping those columns"
        )
        cat_feats = [j for j in cat_feats if j in specs]

    X_cat = transform_target_encoding(
        X3_scaled, cat_feats, state["target_encoding_specs"]
    )

    # 7) Concatenate: continuous (scaled) + categorical (target-encoded)
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
    target_encoding_min_count: int = 50,
    target_encoding_max_categories: int = 50,
    target_encoding_smoothing: float = 1.0,
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
        y_train,
        cont_features_orig=cont_features,
        missing_threshold=missing_threshold,
        drop_low_var=drop_low_var,
        lowvar_threshold=lowvar_threshold,
        cont_strategy=strategy_cont,
        cat_strategy=strategy_cat,
        target_encoding_min_count=target_encoding_min_count,
        target_encoding_max_categories=target_encoding_max_categories,
        target_encoding_smoothing=target_encoding_smoothing,
        drop_duplicates=drop_duplicates,
    )

    Xtr = transform_with_state(X_train, state)
    Xte = transform_with_state(X_test, state)
    return Xtr, y_train, Xte, state

def preprocess_level5_simple_1(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    """
    Level 5 simple_1 pipeline (minimal changes from level2):
      1) Drop features with > 50% missing values (like level2)
      2) NO meta feature removal
      3) NO survey code cleaning
      4) NO outlier removal
      5) Identify categories (standard thresholds)
      6) Force-add given indices to continuous
      7) Impute missing values (like level2):
         - categorical: mode
         - continuous: mean
         - binary: mean
         - pseudo_binary: 0
      8) One-hot encode categorical (like level2: min_count=50, max_categories=50)
      9) Standardize continuous
      10) Oversample positives to 35% (like level2)
    """
    # 1) Drop features with > 50% missing (like level2)
    Xtr, kept1, _ = drop_missing_features(X_train, threshold=0.50)
    Xte = X_test[:, kept1]

    # Compose absolute kept indices (no meta drop)
    final_kept_abs = kept1

    # 2-4) Skip meta removal, survey cleaning, and outlier removal

    # 5) Identify feature types (standard)
    categorical, continuous, binary, pseudo_binary = eda.detect_feature_types_refined(Xtr)

    # 6) Force-add indices to continuous
    force_cont_abs = [63, 64, 146, 251, 252, 253, 254, 288, 289]
    abs_to_rel = {abs_j: rel_j for rel_j, abs_j in enumerate(final_kept_abs)}
    force_cont = [abs_to_rel[abs_j] for abs_j in force_cont_abs if abs_j in abs_to_rel]

    cont_set = set(continuous)
    cat_set = set(categorical)
    bin_set = set(binary)
    pbin_set = set(pseudo_binary)
    for idx in force_cont:
        cont_set.add(idx)
        cat_set.discard(idx)
        bin_set.discard(idx)
        pbin_set.discard(idx)
    continuous = sorted(list(cont_set))
    categorical = sorted(list(cat_set))
    binary = sorted(list(bin_set))
    pseudo_binary = sorted(list(pbin_set))

    # 7) Impute missing values (like level2)
    # 7a) Continuous: mean (like level2)
    imp_cont, _ = fit_imputer(Xtr, continuous, strategy_cont="mean", strategy_cat="mode")
    Xtr_imp = apply_imputer(Xtr, imp_cont, {})
    Xte_imp = apply_imputer(Xte, imp_cont, {})

    # 7b) Binary: mean (like level2)
    for j in binary:
        col = Xtr_imp[:, j]
        m = np.nanmean(col) if np.any(np.isnan(col)) else np.mean(col)
        Xtr_imp[np.isnan(Xtr_imp[:, j]), j] = m
        Xte_imp[np.isnan(Xte_imp[:, j]), j] = m

    # 7c) Pseudo-binary: 0
    for j in pseudo_binary:
        Xtr_imp[np.isnan(Xtr_imp[:, j]), j] = 0.0
        Xte_imp[np.isnan(Xte_imp[:, j]), j] = 0.0

    # 7d) Categorical: mode (like level2)
    imp_cat = {}
    for j in categorical:
        col = Xtr[:, j]
        vals, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        imp_cat[j] = float(vals[np.argmax(counts)]) if vals.size > 0 else 0.0
        Xtr_imp[np.isnan(Xtr_imp[:, j]), j] = imp_cat[j]
        Xte_imp[np.isnan(Xte_imp[:, j]), j] = imp_cat[j]

    # 8) One-hot encoding (like level2)
    specs = fit_one_hot(Xtr_imp, categorical, min_count=50, max_categories=50)
    Xtr_cat = transform_one_hot(Xtr_imp, categorical, specs)
    Xte_cat = transform_one_hot(Xte_imp, categorical, specs)

    # 9) Standardize continuous
    means, stds = fit_standardizer(Xtr_imp, continuous)
    Xtr_scaled = apply_standardizer(Xtr_imp, continuous, means, stds)
    Xte_scaled = apply_standardizer(Xte_imp, continuous, means, stds)

    # Build final matrices
    Xtr_cont = Xtr_scaled[:, continuous] if len(continuous) > 0 else np.empty((Xtr_scaled.shape[0], 0))
    Xte_cont = Xte_scaled[:, continuous] if len(continuous) > 0 else np.empty((Xte_scaled.shape[0], 0))
    Xtr_final = Xtr_cont if Xtr_cat.size == 0 else np.concatenate([Xtr_cont, Xtr_cat], axis=1)
    Xte_final = Xte_cont if Xte_cat.size == 0 else np.concatenate([Xte_cont, Xte_cat], axis=1)

    # Create a minimal state for consistency with other levels
    state = {
        "missing_threshold": 0.50,
        "drop_low_var": False,
        "lowvar_threshold": 0.995,
        "cont_strategy": "mean",
        "cat_strategy": "mode",
        "target_encoding_min_count": 50,
        "target_encoding_max_categories": 50,
        "target_encoding_smoothing": 1.0,
        "kept_after_missing": final_kept_abs,
        "kept_after_lowvar_rel": list(range(len(final_kept_abs))),
        "final_kept_abs": final_kept_abs,
        "cont_features": continuous,
        "cat_features": categorical,
        "imp_cont": imp_cont,
        "imp_cat": imp_cat,
        "std_means": means,
        "std_stds": stds,
        "onehot_specs": specs,
        "final_dim_before_oh": len(final_kept_abs),
    }
    
    # 10) Oversample positives to 35% (like level2)
    Xtr_bal, ytr_bal = oversample_minority(Xtr_final, y_train, target_pos_ratio=0.35)
    return Xtr_bal, ytr_bal, Xte_final, state

# ============================================================
# RUN ALL LEVELS OF PREPRO
# ============================================================
LEVELS = {
    "level0": dict(
        missing_threshold=0.99,
        drop_low_var=False,
        strategy_cont="mean",
        strategy_cat="mode",
        target_encoding_min_count=0,
        target_encoding_max_categories=0,
        target_encoding_smoothing=1.0,
        drop_duplicates=False,
    ),
    "level1": dict(
        missing_threshold=0.80,
        drop_low_var=True,
        lowvar_threshold=0.999,
        strategy_cont="mean",
        strategy_cat="mode",
        target_encoding_min_count=100,
        target_encoding_max_categories=50,
        target_encoding_smoothing=1.0,
        drop_duplicates=False,
    ),
    "level2_unbalanced": dict(
        missing_threshold=0.60,
        drop_low_var=True,
        lowvar_threshold=0.995,
        strategy_cont="mean",
        strategy_cat="mode",
        target_encoding_min_count=50,
        target_encoding_max_categories=20,
        target_encoding_smoothing=1.0,
        drop_duplicates=False,
    ),
    "level2": dict(
        missing_threshold=0.60,
        drop_low_var=True,
        lowvar_threshold=0.995,
        strategy_cont="mean",
        strategy_cat="mode",
        target_encoding_min_count=50,
        target_encoding_max_categories=20,
        target_encoding_smoothing=1.0,
        drop_duplicates=False,
    ),
    "level3": dict(
        missing_threshold=0.40,
        drop_low_var=True,
        lowvar_threshold=0.990,
        strategy_cont="median",
        strategy_cat="mode",  # cat impute via mode
        target_encoding_min_count=20,
        target_encoding_max_categories=10,
        target_encoding_smoothing=0.5,
        drop_duplicates=True,  # drop duplicate features
    ),
    "level4": dict(
        missing_threshold=0.60,
        drop_low_var=True,
        lowvar_threshold=0.995,
        strategy_cont="mean",
        strategy_cat="mode",  # cat impute via mode
        target_encoding_min_count=50,
        target_encoding_max_categories=20,
        target_encoding_smoothing=0.5,
        drop_duplicates=True,  # drop duplicate features
    ),
    "level5": dict(
        tag="level5_simple_1",
    ),
}


def save_csvs_and_state(Xtr, ytr, Xte, state, outdir):
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(f"{outdir}/x_train.csv", Xtr, delimiter=",", fmt="%.6f")
    np.savetxt(f"{outdir}/x_test.csv", Xte, delimiter=",", fmt="%.6f")
    np.savetxt(f"{outdir}/y_train.csv", ytr, delimiter=",", fmt="%.0f")
    json_state = {
        k: state[k]
        for k in [
            "missing_threshold",
            "drop_low_var",
            "lowvar_threshold",
            "cont_strategy",
            "target_encoding_min_count",
            "target_encoding_max_categories",
            "target_encoding_smoothing",
            "kept_after_missing",
            "kept_after_lowvar_rel",
            "final_kept_abs",
            "cont_features",
            "cat_features",
            "imp_cont",
            "imp_cat",
            "std_means",
            "std_stds",
        ]
    }
    
    # Handle target encoding specs (for target encoding approach)
    if "target_encoding_specs" in state:
        json_state["target_encoding_specs"] = {
            str(k): {
                "encodings": {
                    str(val): float(encoding) for val, encoding in v["encodings"].items()
                },
                "other_encoding": float(v["other_encoding"]),
                "global_mean": float(v["global_mean"]),
                "kept_vals": (
                    v["kept_vals"].tolist()
                    if hasattr(v["kept_vals"], "tolist")
                    else list(v["kept_vals"])
                ),
                "has_other": bool(v["has_other"]),
            }
            for k, v in state["target_encoding_specs"].items()
        }
    
    # Handle onehot specs (for one-hot encoding approach)
    if "onehot_specs" in state:
        json_state["onehot_specs"] = {
            str(k): {
                "values": state["onehot_specs"][k]["values"].tolist(),
                "other": state["onehot_specs"][k]["other"]
            }
            for k in state["onehot_specs"]
        }
    
    with open(f"{outdir}/preproc_state.json", "w") as f:
        json.dump(json_state, f, indent=2)


def run_all_levels(x_train, x_test, y_train, cont_features_orig, only=None):

    levels = LEVELS if only is None else {k: LEVELS[k] for k in only}
    # Run preprocessing pipelines for all levels
    for name, cfg in levels.items():
        if name in ["level5"]:
            Xtr, ytr, Xte, state = preprocess_level5_simple_1(x_train, y_train, x_test)
        else :            
            Xtr, ytr, Xte, state = preprocess_pipeline(
                x_train,
                y_train,
                x_test,
                cont_features=cont_features_orig,
                strategy_cont=cfg["strategy_cont"],
                strategy_cat=cfg.get("strategy_cat", "mode"),
                missing_threshold=cfg["missing_threshold"],
                drop_low_var=cfg["drop_low_var"],
                lowvar_threshold=cfg.get("lowvar_threshold", 0.995),
                target_encoding_min_count=cfg["target_encoding_min_count"],
                target_encoding_max_categories=cfg["target_encoding_max_categories"],
                target_encoding_smoothing=cfg["target_encoding_smoothing"],
                drop_duplicates=cfg.get("drop_duplicates", False),
            )

        # OVERSAMPLING for balancing (level5_simple_1 already does this internally)
        if name in ["level2", "level3", "level4"]:
            print(f"  -> Oversampling positives to reach 35% ratio...")
            Xtr, ytr = oversample_minority(Xtr, ytr, target_pos_ratio=0.35)

        save_csvs_and_state(Xtr, ytr, Xte, state, f"preprocessed/{name}")
        print(f"[OK] {name}: train {Xtr.shape}, test {Xte.shape}")


# ============================================================
# main
# ============================================================


def main():
    from helpers import load_csv_data  # stdlib + numpy only

    # Load raw data
    x_train, x_test, y_train, _, _ = load_csv_data("dataset")
    print(f"Raw: train {x_train.shape}, test {x_test.shape}")

    # Detect feature types; remove suspicious 'continuous' (dates/IDs) from continuous set
    _, continuous, _ , _ = eda.detect_feature_types_refined(x_train)
    suspect_features = eda.detect_date_or_id_features(x_train[:, continuous])

    removed_from_continuous = []
    for idx in suspect_features:
        if idx in continuous:
            removed_from_continuous.append(idx)
            continuous.remove(idx)

    cont_features_orig = continuous  # original-coordinate indices

    # Run all levels and export each to its own folder
    os.makedirs("preprocessed", exist_ok=True)
    run_all_levels(
        x_train, x_test, y_train, cont_features_orig, only=["level5"]
    )

    print("\n[Done] All levels exported under ./preprocessed/")
    print("Levels:", ", ".join(LEVELS.keys()))


if __name__ == "__main__":
    main()
