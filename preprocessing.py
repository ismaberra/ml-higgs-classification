import eda as eda
import numpy as np


############ HANDLE TARGET IMBALANCE
def oversample_minority(X, y, target_pos_ratio=0.5, rng=None):
    """
    Random oversampling by duplicating minority (y==1) examples until
    the positive ratio ~= target_pos_ratio. Works with y in {-1, 1}.
    Returns new X, y (shuffled).
    """
    assert X.shape[0] == y.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    # current counts
    pos_mask = (y == 1)
    neg_mask = (y == -1)
    n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())

    # already at or above target? then do nothing
    current_ratio = n_pos / (n_pos + n_neg)
    if current_ratio >= target_pos_ratio or n_pos == 0:
        # nothing to do or no positives at all
        # (if no positives, we can't oversample)
        # return shuffled copy for good measure
        idx = rng.permutation(X.shape[0])
        return X[idx], y[idx]

    # how many positives do we need to reach target?
    # Let N be final total; require n_pos_needed / N = target_pos_ratio
    # with N = n_neg + n_pos_needed  ->  n_pos_needed = target_pos_ratio * (n_neg + n_pos_needed)
    # => n_pos_needed * (1 - target_pos_ratio) = target_pos_ratio * n_neg
    # => n_pos_needed = target_pos_ratio * n_neg / (1 - target_pos_ratio)
    n_pos_needed = int(np.ceil(target_pos_ratio * n_neg / (1.0 - target_pos_ratio)))
    n_to_add = max(0, n_pos_needed - n_pos)

    pos_idx = np.where(pos_mask)[0]
    add_idx = rng.choice(pos_idx, size=n_to_add, replace=True)

    X_bal = np.concatenate([X, X[add_idx]], axis=0)
    y_bal = np.concatenate([y, y[add_idx]], axis=0)

    # shuffle
    perm = rng.permutation(X_bal.shape[0])
    return X_bal[perm], y_bal[perm]

############## DROP FEATURES WITH >50% MISSING VALUES

def drop_missing_features(x: np.ndarray, threshold: float = 0.5):
    """
    Drop features (columns) with more than threshold fraction of missing values.

    Args:
        x : numpy array of shape (n_samples, n_features)
        threshold : float, e.g. 0.5 means drop features with >50% missing

    Returns:
        x_new : numpy array with selected features
        kept_features : list of indices of kept columns
        dropped_features : list of indices of dropped columns
    """
    n_samples = x.shape[0]
    missing_fraction = np.sum(np.isnan(x), axis=0) / n_samples

    kept_features = [j for j in range(x.shape[1]) if missing_fraction[j] <= threshold]
    dropped_features = [j for j in range(x.shape[1]) if missing_fraction[j] > threshold]

    x_new = x[:, kept_features]

    return x_new, kept_features, dropped_features

################## DROP CONSTANT AND NEAR CONSTANT FEATURES

def drop_low_variance_features(x: np.ndarray, threshold: float = 0.995):
    """
    Drop constant and near-constant features using EDA utilities.
    """
    consts = eda.find_constant_features(x)
    near_consts = eda.find_near_constant_features(x, threshold)

    drop_idx = sorted(list(set(consts + near_consts)))
    keep_idx = [j for j in range(x.shape[1]) if j not in drop_idx]

    return x[:, keep_idx], keep_idx, {"constant": consts, "near_constant": near_consts}

################# IMPUTE REMAINING FEATURES WITH <=50% MISSING VALUES

def impute_missing_values(x: np.ndarray, cont_features: list, strategy_cont="mean"):
    """
    Impute missing values:
      - Continuous features -> mean or median
      - Categorical features -> mode (most frequent value)

    Args:
        x : numpy array (n_samples, n_features)
        cont_features : list of indices for continuous features
        strategy_cont : "mean" or "median" for continuous

    Returns:
        x_imputed : numpy array with imputed values
    """
    x_imputed = x.copy()
    n_features = x.shape[1]

    for j in range(n_features):
        col = x[:, j]
        mask = np.isnan(col)

        if np.any(mask):
            if j in cont_features:
                # continuous -> mean or median
                if strategy_cont == "mean":
                    fill_val = np.nanmean(col)
                elif strategy_cont == "median":
                    fill_val = np.nanmedian(col)
                else:
                    raise ValueError("strategy_cont must be 'mean' or 'median'")
            else:
                # categorical -> mode
                values, counts = np.unique(col[~mask], return_counts=True)
                fill_val = values[np.argmax(counts)] if values.size > 0 else 0

            x_imputed[mask, j] = fill_val

    return x_imputed

################### FIT-TRANSFORM FOR TRAIN AND TEST DATA FOR CONTINUOUS FEATURES

def fit_standardizer(x: np.ndarray, cont_features: list):
    """
    Compute mean and std for continuous features.

    Args:
        x : numpy array (n_samples, n_features), assumed already imputed
        cont_features : list of indices of continuous features

    Returns:
        means : dict {feature_idx: mean}
        stds  : dict {feature_idx: std}
    """
    means, stds = {}, {}
    for j in cont_features:
        col = x[:, j]
        means[j] = np.mean(col)
        stds[j] = np.std(col) if np.std(col) > 1e-12 else 1.0  # avoid div/0
    return means, stds


def apply_standardizer(x: np.ndarray, cont_features: list, means: dict, stds: dict):
    """
    Apply standardization to continuous features.

    Args:
        x : numpy array (n_samples, n_features)
        cont_features : list of indices of continuous features
        means, stds : dicts from fit_standardizer

    Returns:
        x_scaled : numpy array with standardized continuous features
    """
    x_scaled = x.copy()
    for j in cont_features:
        x_scaled[:, j] = (x_scaled[:, j] - means[j]) / stds[j]
    return x_scaled

################# FIT TRANSFORM FOR TRAIN AND TEST DATA FOR CATEGORICAL FEATURES (ENCODING)

def fit_one_hot(x: np.ndarray, cat_features: list, min_count: int = 50, max_categories: int = 50):
    """
    Learn one-hot specs for categorical features:
      - Keeps up to max_categories most frequent values
      - Groups rare categories (count < min_count) into 'OTHER' bucket

    Args:
        x : numpy array (n_samples, n_features), imputed
        cat_features : list of indices for categorical features
        min_count : minimum frequency to keep category explicitly
        max_categories : maximum number of categories to keep per feature

    Returns:
        specs : dict {feature_idx: {"values": kept_values, "other": bool}}
    """
    specs = {}
    for j in cat_features:
        col = x[:, j]
        vals, counts = np.unique(col, return_counts=True)

        # sort by frequency descending
        order = np.argsort(counts)[::-1]
        vals, counts = vals[order], counts[order]

        # keep frequent categories
        keep_mask = counts >= min_count
        kept_vals = vals[keep_mask]

        if kept_vals.size > max_categories:
            kept_vals = kept_vals[:max_categories]

        specs[j] = {
            "values": kept_vals.astype(float),
            "other": kept_vals.size < vals.size  # if we dropped some
        }
    return specs


def transform_one_hot(x: np.ndarray, cat_features: list, specs: dict):
    """
    Apply one-hot encoding using fitted specs.

    Args:
        x : numpy array (n_samples, n_features)
        cat_features : list of indices for categorical features
        specs : dict from fit_one_hot

    Returns:
        X_oh : numpy array with concatenated one-hot features
    """
    n = x.shape[0]
    encoded_blocks = []

    for j in cat_features:
        col = x[:, j]
        values = specs[j]["values"]
        other_flag = specs[j]["other"]

        # allocate output
        width = values.size + (1 if other_flag else 0)
        oh = np.zeros((n, width))

        for i, v in enumerate(col):
            if v in values:
                k = np.where(values == v)[0][0]
                oh[i, k] = 1.0
            else:
                if other_flag:
                    oh[i, -1] = 1.0
                # else: unseen rare value -> leave row zeros
        encoded_blocks.append(oh)

    # concatenate all categorical encodings
    if encoded_blocks:
        return np.concatenate(encoded_blocks, axis=1)
    else:
        return np.empty((n, 0))


# -------------------------------
# Full Pipeline
# -------------------------------

def preprocess_pipeline(x_train, y_train, x_test, cont_features, strategy_cont="mean"):
    """
    Full preprocessing pipeline:
      - Drop high-missing features
      - Drop constant/near-constant features
      - Impute
      - Standardize continuous
      - One-hot encode categorical
    """
    # Drop features >50% missing
    x_train, kept, dropped = drop_missing_features(x_train, threshold=0.5)
    x_test = x_test[:, kept]

    # Drop constant / near-constant features (EDA helpers)
    x_train, kept2, dropped_lowvar = drop_low_variance_features(x_train, threshold=0.995)
    x_test = x_test[:, kept2]

    # A FAIRE : DROP DUPLICATE FEATURES (7 AND 8 ARE THE SAME)

    # Impute
    x_train = impute_missing_values(x_train, cont_features, strategy_cont)
    x_test = impute_missing_values(x_test, cont_features, strategy_cont)

    # Standardize continuous
    means, stds = fit_standardizer(x_train, cont_features)
    x_train = apply_standardizer(x_train, cont_features, means, stds)
    x_test = apply_standardizer(x_test, cont_features, means, stds)

    # One-hot encode categorical
    cat_features = [j for j in range(x_train.shape[1]) if j not in cont_features]
    specs = fit_one_hot(x_train, cat_features, min_count=50, max_categories=50)
    X_train_cat = transform_one_hot(x_train, cat_features, specs)
    X_test_cat = transform_one_hot(x_test, cat_features, specs)

    # Combine continuous + categorical
    X_train_proc = np.concatenate([x_train[:, cont_features], X_train_cat], axis=1)
    X_test_proc = np.concatenate([x_test[:, cont_features], X_test_cat], axis=1)

    return X_train_proc, y_train, X_test_proc, kept, dropped


def main():
    from helpers import load_csv_data, create_csv_submission

    # load raw data
    x_train, x_test, y_train, _, _ = load_csv_data("dataset")
    print(f"Raw: train {x_train.shape}, test {x_test.shape}")

    # continuous features from EDA
    #A FAIRE : BIZZARE DE METTRE CETTE LISTE NOUS MEME 
    cont_features = [7, 8, 222, 226, 229, 253]

    # run pipeline
    Xtr, ytr, Xte, kept, dropped = preprocess_pipeline(x_train, y_train, x_test, cont_features)

    # save arrays
    np.save("X_train_proc.npy", Xtr)
    np.save("y_train.npy", ytr)
    np.save("X_test_proc.npy", Xte)

    print("\n[Report]")
    print(f"  Dropped {len(dropped)} features with >50% missing")
    print(f"  Kept {len(kept)} features → processed train {Xtr.shape}, test {Xte.shape}")

if __name__ == "__main__":
    main()