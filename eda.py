import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as combinations
from collections import defaultdict



def dataset_overview(x, y):
    n_samples, n_features = x.shape
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Target distribution: {np.unique(y, return_counts=True)}")

def summarize_features(x):
    stats = []
    for j in range(x.shape[1]):
        col = x[:, j]
        # ignore NaNs when computing stats
        col_min = np.nanmin(col)
        col_max = np.nanmax(col)
        col_mean = np.nanmean(col)
        col_median = np.nanmedian(col)
        col_std = np.nanstd(col)
        missing = np.sum(np.isnan(col))
        unique_vals = len(np.unique(col[~np.isnan(col)]))
        
        stats.append((j, col_min, col_max, col_mean, col_median, col_std, missing, unique_vals))
    
    return stats

def print_feature_stats(stats, n=5):
    print("feat | min | max | mean | median | std | missing | unique")
    print("---------------------------------------------------------")
    for row in stats[:n]:  # print first n features
        print(f"{row[0]:4d} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.2f} | {row[4]:.2f} | {row[5]:.2f} | {row[6]} | {row[7]}")


def analyze_missingness(x, bins=[0, 0.05, 0.2, 0.5, 1.0]):
    """
    Analyze missing values per feature and show histogram of missingness.
    
    Args:
        x : numpy array (n_samples, n_features)
        bins : list of thresholds for grouping features by missing proportion
    """
    n_samples = x.shape[0]
    # compute missing proportion per feature
    perc_missing = np.sum(np.isnan(x), axis=0) / n_samples

    # print global stats
    print(f"Features with no missing: {(perc_missing == 0).sum()}")
    print(f"Features with some missing: {(perc_missing > 0).sum()}")
    print(f"Max missing rate: {perc_missing.max():.2%}")

    # plot histogram of missing proportions
    plt.hist(perc_missing, bins=50, color="indianred", edgecolor="black")
    plt.xlabel("Proportion of missing values per feature")
    plt.ylabel("Number of features")
    plt.title("Distribution of missingness across features")
    plt.show()

    # also group counts by bins
    bin_counts = np.histogram(perc_missing, bins=bins)[0]
    print("\nMissingness categories:")
    for i in range(len(bin_counts)):
        low, high = bins[i], bins[i+1]
        print(f"  {low:.0%}–{high:.0%}: {bin_counts[i]} features")

    return perc_missing



def plot_target_distribution(y):
    values, counts = np.unique(y, return_counts=True)
    plt.bar(values.astype(str), counts, color=["steelblue", "indianred"])
    plt.xlabel("Class label")
    plt.ylabel("Number of samples")
    plt.title("Target distribution (y_train)")
    for i, c in enumerate(counts):
        plt.text(i, c, str(c), ha='center', va='bottom')
    plt.show()

def find_constant_features(x):
    constant_features = []
    for j in range(x.shape[1]):
        unique_vals = np.unique(x[:, j][~np.isnan(x[:, j])])  # ignore NaNs
        if len(unique_vals) == 1:
            constant_features.append(j)
    return constant_features

def find_near_constant_features(x, threshold=0.99):
    near_constant = []
    n_samples = x.shape[0]
    for j in range(x.shape[1]):
        col = x[:, j]
        # ignore NaNs
        values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        if counts.max() / n_samples >= threshold:
            near_constant.append(j)
    return near_constant


def analyze_near_constant_features(x, y, near_const_feats, top_n=5):
    """
    For each near-constant feature, print the distribution of values
    and the proportion of target=1 within each value.
    
    Args:
        x: (n_samples, n_features) numpy array
        y: (n_samples,) numpy array with values in {-1, 1}
        near_const_feats: list of feature indices to check
        top_n: max number of unique values to display (saves space)
    """
    for feat in near_const_feats:
        col = x[:, feat]
        values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        
        print(f"\nFeature {feat} (unique={len(values)}, most frequent={counts.max()/len(col):.2%})")
        for v, c in zip(values[:top_n], counts[:top_n]):
            mask = (col == v)
            # convert y from {-1,1} to {0,1} for positive rate
            positives = np.sum(y[mask] == 1)
            rate = positives / c if c > 0 else 0
            print(f"  value {v}: count={c}, positives={positives} ({rate:.2%})")
        
        if len(values) > top_n:
            print(f"  ... {len(values) - top_n} more values not shown ...")

def feature_scale_summary(x):
    mins = np.nanmin(x, axis=0)
    maxs = np.nanmax(x, axis=0)
    stds = np.nanstd(x, axis=0)

    print(f"Global min across features: {mins.min():.2f}")
    print(f"Global max across features: {maxs.max():.2f}")
    print(f"Median std across features: {np.median(stds):.2f}")
    print(f"Features with std < 1e-6: {(stds < 1e-6).sum()}")
    print(f"Features with std > 1e3: {(stds > 1e3).sum()}")

    return mins, maxs, stds


def plot_feature_stds(stds, bins=50):
    plt.hist(stds, bins=bins, color="steelblue", edgecolor="black")
    plt.xlabel("Standard deviation (scale of feature)")
    plt.ylabel("Number of features")
    plt.title("Distribution of feature scales (std per feature)")
    plt.show()

def extreme_scale_features(stds, low_thresh=1e-6, high_thresh=1e3):
    small = np.where(stds < low_thresh)[0]
    large = np.where(stds > high_thresh)[0]
    return small, large

def plot_feature_distribution(x, feat_idx, bins=30):
    col = x[:, feat_idx - 1]
    plt.hist(col[~np.isnan(col)], bins=bins, color="steelblue", edgecolor="black")
    plt.title(f"Feature {feat_idx} distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def detect_feature_types(x, threshold=20):
    categorical, continuous = [], []
    for j in range(x.shape[1]):
        n_unique = len(np.unique(x[:, j][~np.isnan(x[:, j])]))
        if n_unique <= threshold or n_unique / len(x) < 0.01 :
            categorical.append(j)
        else:
            continuous.append(j)
    return categorical, continuous


def detect_feature_types_refined(x, threshold_cat=15, nan_ratio_limit=0.95):
    """
    Detect feature types (categorical, continuous, binary) in survey-style datasets.

    Rules:
    - binary: exactly two unique values (e.g., [0,1] or [1,2]) or one unique + many NaNs.
    - categorical: small integer-coded ranges (e.g., 1–5, 1–9, 0–10), representing discrete choices.
    - continuous: many unique values or large numerical range, possibly with decimals.

    Parameters
    ----------
    x : np.ndarray
        2D array of shape (n_samples, n_features) containing the dataset.
    threshold_cat : int
        Maximum number of unique values for a feature to be considered categorical.
    nan_ratio_limit : float
        If more than this ratio of NaNs, the feature is ignored.

    Returns
    -------
    categorical, continuous, binary : list[int]
        Lists of feature indices for each detected type.
    """

    categorical, continuous, binary, ignored = [], [], [], []
    n_features = x.shape[1]

    for j in range(n_features):
        col = x[:, j]
        col_nonan = col[~np.isnan(col)]
        n_unique = len(np.unique(col_nonan))
        nan_ratio = np.isnan(col).mean()

        # Skip features that are almost entirely missing
        if n_unique == 0 or nan_ratio > nan_ratio_limit:
            ignored.append(j)
            continue

        unique_vals = np.unique(col_nonan)
        max_val, min_val = np.max(unique_vals), np.min(unique_vals)

        # True binary (exactly 2 values, e.g. [0,1], [1,2])
        if n_unique == 2:
            binary.append(j)
            continue

        # Pseudo-binary (only one value + many NaN)
        if n_unique == 1 and nan_ratio > (0.5 + 0.1 * np.log1p(len(col))):
            binary.append(j)
            continue

        # Categorical: few unique integer values (1–15), small range
        if (
            np.all(col_nonan % 1 == 0) and          # integer-coded
            n_unique <= threshold_cat and           # few unique
            max_val <= 20 and                       # small range
            (max_val - min_val) <= 20               # tightly grouped
        ):
            categorical.append(j)
            continue

        # Continuous: large value diversity, wide range, or decimals
        if (
            n_unique > threshold_cat or
            max_val > 20 or
            not np.all(col_nonan % 1 == 0)
        ):
            continuous.append(j)
            continue

        # Default fallback (should rarely happen)
        categorical.append(j)

    return categorical, continuous, binary





def is_survey_code(val):
        """Check if the number is made only of digits {7,0} or {9,0}."""
        s = str(int(abs(val)))  # remove sign, convert to string
        return all(ch in "70" for ch in s) or all(ch in "90" for ch in s)
    
def replace_survey_codes_by_pattern(x, feature_names=None, gap_ratio=2.0):
    """
    Detect and replace 'survey missing codes' such as 7, 9, 77, 99, 700, 900, etc.
    based on numeric isolation (large gap or variance) AND digit composition ({7,0} or {9,0}).

    Parameters
    ----------
    x : np.ndarray
        2D array of shape (n_samples, n_features).
    feature_names : list[str] or None
        Optional feature names used for printing.
    gap_ratio : float
        Threshold ratio for detecting a large jump compared to the mean previous gap.
        Example: 2.0 means “the last gap must be at least twice larger than the mean of the others”.

    Returns
    -------
    x_clean : np.ndarray
        Copy of the input array with detected survey codes replaced by NaN.
    """

    x_clean = np.copy(x)
    n_features = x.shape[1]

    for j in range(n_features):
        col = x_clean[:, j]
        col_nonan = col[~np.isnan(col)]
        if col_nonan.size < 4:  # too few values to analyze
            continue

        unique_vals = np.unique(col_nonan)
        if unique_vals.size < 4:
            continue

        diffs = np.diff(unique_vals)
        mean_gap = np.mean(diffs[:-2]) if diffs.size > 2 else np.mean(diffs)
        last_gap = diffs[-1]
        second_last_gap = diffs[-2] if diffs.size >= 2 else 0.0

        # Detect if last or last two values are structurally far
        isolated_two = last_gap >= gap_ratio * mean_gap and second_last_gap >= gap_ratio * mean_gap
        isolated_one = last_gap >= gap_ratio * mean_gap

        # Last two values in sorted unique list
        last_val = unique_vals[-1]
        prev_val = unique_vals[-2]

        # Check pattern condition
        last_two_are_codes = is_survey_code(last_val) and is_survey_code(prev_val)
        last_one_is_code = is_survey_code(last_val)

        replaced = False
        name = feature_names[j] if feature_names is not None else f"col_{j}"

        # Case 1: both last values look like survey codes
        if isolated_two and last_two_are_codes:
            mask = np.isin(col, [prev_val, last_val])
            col[mask] = np.nan
            replaced = True
            print(f"[{name}] → replaced values {prev_val}, {last_val} (2-code block) with NaN")

        # Case 2: only last value is a survey code
        elif isolated_one and last_one_is_code:
            mask = col == last_val
            col[mask] = np.nan
            replaced = True
            print(f"[{name}] → replaced value {last_val} (single code) with NaN")

        if replaced:
            x_clean[:, j] = col

    return x_clean




def detect_dependencies(x, parent_idx, candidate_idxs, threshold=0.95, treat_nan_as_zero=True):
    """
    Detect features that are likely dependent on a parent feature.

    Parameters
    ----------
    x : np.ndarray
        The feature matrix.
    parent_idx : int
        Index of the potential parent feature.
    candidate_idxs : list or range
        Indices of features to test as potential children.
    threshold : float, optional
        Minimum proportion of missing values among child features when parent=0.
        Default is 0.95 (95% missing = strong dependency).
    treat_nan_as_zero : bool, optional
        If True, treat NaN values in the parent as equivalent to parent=0 (non-applicable).

    Returns
    -------
    dependent : list
        List of indices of features likely dependent on the parent feature.
    """

    parent = x[:, parent_idx]
    dependent = []

    # Build mask for parent == 0
    if treat_nan_as_zero:
        mask_parent0 = np.isnan(parent) | (parent == 0)
    else:
        mask_parent0 = (parent == 0)

    # Handle edge case: no examples where parent == 0
    if np.sum(mask_parent0) == 0:
        print(f"⚠️ No examples where parent {parent_idx} == 0 — skipping dependency test.")
        return []

    for j in candidate_idxs:
        if j == parent_idx:
            continue  # skip self
        child = x[:, j]

        # Skip if all values are NaN
        if np.all(np.isnan(child)):
            continue

        # Compute ratio of missing children where parent=0
        missing_when_parent0 = np.isnan(child[mask_parent0]).mean()

        if missing_when_parent0 > threshold:
            dependent.append(j)

    return dependent



def detect_date_or_id_features(x, feature_names=None):
    suspect_indices = []
    n_samples = x.shape[0]

    for j in range(x.shape[1]):
        col = x[:, j]
        col_nonan = col[~np.isnan(col)]
        if len(col_nonan) == 0:
            continue

        n_unique = len(np.unique(col_nonan))
        min_val, max_val = np.min(col_nonan), np.max(col_nonan)
        std_val = np.std(col_nonan)
        ratio_unique = n_unique / n_samples

        # 1️⃣ Probable années
        if 1900 <= min_val <= 2100 and max_val <= 2100:
            suspect_indices.append(j)
            continue

        # 2️⃣ Probable jours/mois
        if (
            1 <= min_val <= 31
            and n_unique <= 31
            and std_val < 10
            and ratio_unique < 0.05
            and np.median(col_nonan) <= 12  # typique pour des mois
        ):
            suspect_indices.append(j)
            continue

        # 3️⃣ Probable IDs / dates concaténées
        if (
            np.all(col_nonan % 1 == 0)
            and max_val > 9999
            and (ratio_unique < 0.1 or n_unique < 0.1 * n_samples)
        ):
            suspect_indices.append(j)
            continue

    if feature_names is not None:
        return [(j, feature_names[j]) for j in suspect_indices]
    return suspect_indices





    


def compute_feature_correlation(x, features_idx):
    """
    Compute correlation matrix for a subset of features.
    
    Args:
        x : numpy array (n_samples, n_features)
        features_idx : list of feature indices to compute correlation
    
    Returns:
        corr_matrix : len(features_idx) x len(features_idx) correlation matrix
    """
    # extract only the columns of interest
    sub_x = x[:, features_idx]
    # replace NaNs with column means
    sub_x = np.where(np.isnan(sub_x), np.nanmean(sub_x, axis=0), sub_x)
    # compute correlation matrix
    corr_matrix = np.corrcoef(sub_x, rowvar=False)
    return corr_matrix


def find_highly_correlated(corr_matrix, threshold=0.95):
    """
    Find pairs of features with correlation above threshold.
    """
    pairs = []
    n = corr_matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_matrix[i, j]) > threshold:
                pairs.append((i, j, corr_matrix[i, j]))
    return pairs


def compare_y_distribution_from_report(x, y, report, feature_ids_idx, label):
    """
    Compares the distribution of y between clean and outlier individuals
    for a given list of features (numeric or continuous).
    """
    # Building an outlier mask
    outlier_mask = np.zeros(x.shape[0], dtype=bool)
    for f, _, _, q01, q99 in report:
        values = x[:, f]
        outlier_mask |= (values < q01) | (values > q99)

    clean_idx = ~outlier_mask
    outlier_idx = outlier_mask

    print(f"\n[{label}] Individuals without outlier: {clean_idx.sum()}, with outlier: {outlier_idx.sum()}")

    # Prepare data for seaborn
    y_clean = y[clean_idx]
    y_outlier = y[outlier_idx]
    data = {
        "y": np.concatenate([y_clean, y_outlier]),
        "group": [f"{label}-clean"] * len(y_clean) + [f"{label}-outlier"] * len(y_outlier)
    }

    # Plot
    plt.figure(figsize=(8,5))
    sns.countplot(x="y", hue="group", data=data, palette="Set1")
    plt.title(f"Distribution of y for {label} features")
    plt.xlabel("Class label (y)")
    plt.ylabel("Number of samples")
    plt.legend(title="Group")
    plt.show()

# Standardization (z-score)
def standardize(x_train, x_test):
    """
    Standardize features: mean=0, std=1.
    Important: use train statistics for both train and test.
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0] = 1  # avoid division by zero
    
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std
    
    return x_train_std, x_test_std



def one_hot_encode_numpy(x, feature_names=None, max_categories=30):
    """
    One-Hot Encode categorical features using pure NumPy.

    Parameters
    ----------
    x : np.ndarray
        2D array (n_samples, n_features) containing categorical features.
    feature_names : list[str] or None
        Optional list of feature names for labeling the encoded columns.
    max_categories : int
        Maximum number of unique values to encode per feature.
        Features exceeding this limit are skipped (to prevent explosion of dimensions).

    Returns
    -------
    x_encoded : np.ndarray
        Encoded matrix with 0/1 values.
    new_feature_names : list[str]
        Names of the new encoded columns.
    skipped_features : list[int]
        Indices of features that were skipped due to too many categories.
    """

    n_samples, n_features = x.shape
    encoded_blocks = []
    new_feature_names = []
    skipped_features = []

    for j in range(n_features):
        col = x[:, j]
        col_nonan = col[~np.isnan(col)]
        unique_vals = np.unique(col_nonan)

        # Skip features that are empty
        if unique_vals.size == 0:
            continue

        # Skip overly complex features (too many categories)
        if unique_vals.size > max_categories:
            skipped_features.append(j)
            continue

        # Create the encoded block
        encoded = np.zeros((n_samples, unique_vals.size))
        for i, val in enumerate(unique_vals):
            mask = (col == val)
            encoded[mask, i] = 1.0

            # Column name format: featureName_value
            if feature_names is not None:
                new_feature_names.append(f"{feature_names[j]}={int(val)}")
            else:
                new_feature_names.append(f"col{j}={int(val)}")

        encoded_blocks.append(encoded)

    # Concatenate all encoded columns horizontally
    if len(encoded_blocks) > 0:
        x_encoded = np.concatenate(encoded_blocks, axis=1)
    else:
        x_encoded = np.zeros((n_samples, 0))

    return x_encoded, new_feature_names, skipped_features



def find_suspicious_features(x_train, y_train, corr_threshold=0.3):
    """
    Identify suspicious features that may act as data leaks or non-informative features.
    Criteria:
    - High absolute correlation with y (above corr_threshold).
    - Very large numeric ranges that may indicate dates or IDs.
    """
    n_features = x_train.shape[1]
    suspicious = []

    for j in range(n_features):
        col = x_train[:, j]

        # Compute correlation (skip if constant)
        if np.std(col) == 0:
            continue
        corr = np.corrcoef(col, y_train)[0, 1]

        # Range check
        col_min, col_max = np.min(col), np.max(col)
        value_range = col_max - col_min

        # Flag suspicious features
        if abs(corr) > corr_threshold:
            suspicious.append((j, "High correlation", corr))
        elif value_range > 1e6:  # arbitrary threshold, e.g. dates
            suspicious.append((j, "Large range (maybe date/ID)", value_range))

    return suspicious





def clean_survey_codes(x, feature_names=None, verbose=True):
    """
    Remove survey codes using the following logic:
      1. If the value has more than 3 digits and consists only of 7, 8, or 9 → NaN
      2. For suspected codes (7, 8, 9, 77, 88, 99), compute distance to the closest
         normal value. If this distance is >= 2 × the mean distance between normal values → NaN
    """

    x_clean = np.copy(x)
    if x_clean.ndim == 1:
        x_clean = x_clean.reshape(-1, 1)

    n_samples, n_features = x_clean.shape
    replaced_log = {}
    suspect_vals = np.array([7, 8, 9, 77, 88, 99])

    for j in range(n_features):
        col = x_clean[:, j]
        valid_mask = ~np.isnan(col)
        unique_vals = np.unique(col[valid_mask])
        to_remove = []

        # --- Rule 1: remove repeated-digit codes with >=3 digits ---
        for v in unique_vals:
            s = str(int(v))
            if len(s) >= 3 and len(set(s)) == 1 and s[0] in {"7", "8", "9"}:
                to_remove.append(v)

        # --- Rule 2: detect isolated suspected codes adaptively and sequentially ---
        is_suspect = np.isin(unique_vals, suspect_vals)
        suspect_in_data = np.sort(unique_vals[is_suspect])
        normal_vals = np.sort(unique_vals[~is_suspect])

        if normal_vals.size >= 2 and suspect_in_data.size > 0:
            mean_dist = np.mean(np.diff(normal_vals))
            remove_rest = False  # once a code is detected, all following are codes

            for v in suspect_in_data:
                if remove_rest:
                    # once one suspect is marked as code, all following are codes
                    to_remove.append(v)
                    continue

                # compute distance to closest normal value
                d = np.min(np.abs(normal_vals - v))

                if d < 2 * mean_dist:
                    # close → treat as normal, include and re-compute
                    normal_vals = np.sort(np.append(normal_vals, v))
                    mean_dist = np.mean(np.diff(normal_vals))
                else:
                    # far → mark as code, and mark all next suspects as codes
                    to_remove.append(v)
                    remove_rest = True

        # --- Apply replacements ---
        if to_remove:
            mask = np.isin(col, to_remove)
            col[mask] = np.nan
            x_clean[:, j] = col
            replaced_log[j if feature_names is None else feature_names[j]] = sorted(set(to_remove))

    # --- Log summary ---
    if verbose and replaced_log:
        print("Removed survey codes:")
        for k, v in replaced_log.items():
            print(f"  {k}: {v}")

    return x_clean



def detect_hierarchical_dependencies(x, feature_names, threshold=0.995, verbose=True):
    """
    Detect hierarchical dependencies between survey features.

    Logic:
    - Compare all feature pairs (A, B) with A appearing before B.
    - If >97% of non-NaN entries in B occur only when A is non-NaN (presence-based),
      OR only when A takes a specific value v (value-based),
      then B is said to depend on A.
    - Build hierarchical groups (A → B → C) from dependencies.

    Parameters
    ----------
    x : np.ndarray
        Data matrix (rows = samples, cols = features)
    feature_names : list of str
        Feature names corresponding to the columns of x
    threshold : float
        Dependency ratio threshold (default 0.97)
    verbose : bool
        If True, prints dependency summary

    Returns
    -------
    groups : list[list[str]]
        Hierarchical dependency groups
    dependencies : list[tuple[str, str, str, str, float]]
        Detailed dependencies as (A, type, value, B, ratio)
    """
    n_features = x.shape[1]
    dependencies = []
    dep_map = defaultdict(set)

    for i in range(n_features):
        A = x[:, i]
        mask_A = ~np.isnan(A)
        if np.sum(mask_A) == 0:
            continue
        unique_A = np.unique(A[mask_A])

        for j in range(i + 1, n_features):  # only forward direction
            B = x[:, j]
            mask_B = ~np.isnan(B)
            if np.sum(mask_B) == 0:
                continue

            # --- Presence-based dependency ---
            p_B_given_A = np.mean(mask_B[mask_A]) if np.sum(mask_A) > 0 else 0
            p_B_given_notA = np.mean(mask_B[~mask_A]) if np.sum(~mask_A) > 0 else 0

            if p_B_given_A >= threshold and p_B_given_notA < (1 - threshold):
                dependencies.append((feature_names[i], "presence", None,
                                     feature_names[j], round(p_B_given_A, 3)))
                dep_map[feature_names[i]].add(feature_names[j])
                continue  # no need to test value-based if presence suffices

            # --- Value-based dependency ---
            for val in unique_A:
                mask_A_val = mask_A & (A == val)
                if np.sum(mask_A_val) == 0:
                    continue

                p_B_given_Aval = np.mean(mask_B[mask_A_val]) if np.sum(mask_A_val) > 0 else 0
                p_B_given_other = np.mean(mask_B[mask_A & (A != val)]) if np.sum(mask_A & (A != val)) > 0 else 0

                if p_B_given_Aval >= threshold and p_B_given_other < (1 - threshold):
                    dependencies.append((feature_names[i], "value", val,
                                         feature_names[j], round(p_B_given_Aval, 3)))
                    dep_map[feature_names[i]].add(feature_names[j])
                    break

    # --- Build hierarchical groups safely ---
    visited = set()
    groups = []

    def dfs(node, chain):
        visited.add(node)
        chain.append(node)
        for child in dep_map.get(node, []):
            if child not in visited:
                dfs(child, chain)

    # iterate over a fixed list of keys to avoid runtime error
    for root in list(dep_map.keys()):
        if root not in visited:
            chain = []
            dfs(root, chain)
            if len(chain) > 1:
                groups.append(chain)


    # --- Verbose summary ---
    if verbose:
        print(f"Detected {len(dependencies)} dependencies.")
        for A, dtype, val, B, ratio in dependencies:
            if dtype == "presence":
                print(f"{B} depends on {A} (presence-based, ratio={ratio})")
            else:
                print(f"{B} depends on {A}=={val} (value-based, ratio={ratio})")
        print(f"\nIdentified {len(groups)} hierarchical groups:")
        for g in groups:
            print("  → ".join(g))

    return groups, dependencies



import numpy as np

def process_dependency_groups(x, feature_names, groups, verbose=True):
    """
    Process dependency groups to prepare for model training and one-hot encoding.

    Steps:
      1. Remove constant features within groups (1 unique non-NaN value)
      2. Merge connected groups (avoid duplicate dependencies)
      3. Build encoding plan: root → dependent features
      4. Optionally print details if verbose=True

    Parameters
    ----------
    x : np.ndarray
        Dataset as (samples × features)
    feature_names : list of str
        Names of all features in the same column order as x
    groups : list[list[str]]
        Dependency groups (each list is ordered root → child → ...)
    verbose : bool
        If True, print progress and results

    Returns
    -------
    cleaned_groups : list[list[str]]
        Optimized dependency groups
    to_remove : list[str]
        Constant or useless features to drop
    encoding_plan : dict
        Mapping of root features → dependent features
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    to_remove = []
    cleaned_groups = []
    seen = set()

    # --- Step 1: remove constant features within groups ---
    for g in groups:
        valid_feats = []
        for feat in g:
            idx = name_to_idx.get(feat)
            if idx is None:
                continue
            col = x[:, idx]
            vals = col[~np.isnan(col)]
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                to_remove.append(feat)
            else:
                valid_feats.append(feat)
        if len(valid_feats) > 1:
            cleaned_groups.append(valid_feats)

    # --- Step 2: merge connected groups ---
    merged = []
    for g in cleaned_groups:
        if any(feat in seen for feat in g):
            continue
        connected = set(g)
        changed = True
        while changed:
            changed = False
            for h in cleaned_groups:
                if connected.intersection(h):
                    new_size = len(connected)
                    connected.update(h)
                    if len(connected) != new_size:
                        changed = True
        merged.append(sorted(connected))
        seen.update(connected)
    cleaned_groups = merged

    # --- Step 3: build encoding plan (root → dependents) ---
    encoding_plan = {}
    for g in cleaned_groups:
        root = g[0]
        children = g[1:]
        encoding_plan[root] = children

    # --- Step 4: verbose logging ---
    if verbose:
        print(f"\n=== Dependency Group Processing Summary ===")
        print(f"Removed {len(to_remove)} constant features:")
        if to_remove:
            print("  " + ", ".join(to_remove))
        print(f"\nOptimized to {len(cleaned_groups)} dependency groups:")
        for g in cleaned_groups:
            print("  " + " → ".join(g))
        print(f"\nEncoding plan created for {len(encoding_plan)} root features.\n")

    return cleaned_groups, to_remove, encoding_plan
