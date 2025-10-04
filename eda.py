import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    col = x[:, feat_idx]
    plt.hist(col[~np.isnan(col)], bins=bins, color="steelblue", edgecolor="black")
    plt.title(f"Feature {feat_idx} distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

"""
def detect_feature_types(x, threshold=20):
    categorical, continuous = [], []
    for j in range(x.shape[1]):
        n_unique = len(np.unique(x[:, j][~np.isnan(x[:, j])]))
        if n_unique <= threshold or n_unique / len(x) < 0.01 :
            categorical.append(j)
        else:
            continuous.append(j)
    return categorical, continuous
"""

def detect_feature_types(x, threshold=20, binary_tolerance=0.95):
    """
    Detect categorical, continuous, and binary features.
    - categorical: few unique values (<= threshold) or very rare values (<1%)
    - binary: exactly 2 unique values (0 and 1)
    - continuous: many unique values
    """
    categorical, continuous, binary = [], [], []
    
    for j in range(x.shape[1]):
        col = x[:, j]
        unique_vals, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        n_unique = len(unique_vals)
        
        # Proportion of 0/1 values
        mask_binary = np.isin(unique_vals, [0, 1])
        ratio_binary = counts[mask_binary].sum() / counts.sum()
        
        if ratio_binary >= binary_tolerance:
            binary.append(j)
        elif n_unique <= threshold or n_unique / len(x) < 0.01:
            categorical.append(j)
        else:
            continuous.append(j)
    
    return categorical, continuous, binary
    """{
        "categorical": categorical,
        "continuous": continuous,
        "binary": binary
    }"""


def detect_dependencies(x, parent_idx, candidate_idxs, threshold=0.95):
    """
    Detect child features that depend on a parent feature.
    threshold = % of missing children when parent=0
    """
    parent = x[:, parent_idx]
    dependent = []
    for j in candidate_idxs:
        child = x[:, j]
        missing_when_parent0 = np.isnan(child[parent == 0]).mean()
        if missing_when_parent0 > threshold:
            dependent.append(j)
    return dependent

    


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
