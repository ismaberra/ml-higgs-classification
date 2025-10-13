[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)


# EDA Conclusion :

### Ismael :

## Goal :

The preprocessing stage aimed to transform the raw dataset from Project 1 – ML212 into a clean, model-ready format by:
- Correctly separating continuous, categorical, and binary features
- Standardizing continuous variables
- Efficiently one-hot-encoding categorical features
- Reducing false positives in feature classification (e.g., year, date, or ID columns)

## Feature-Type Detection :

We implemented an improved detection function:
categorical, continuous, binary = detect_feature_types_refined(x_train)
This version uses:

- Number of unique values
- Dispersion (standard deviation of value gaps)
- Value range

to classify:

- Binary: exactly two values (e.g., 0/1 or 1/2)
- Categorical: few unique, irregularly spaced values
- Continuous: many unique or regularly spaced values

✅ Fixed issue:
Features like IDAY, IMONTH, or IYEAR were wrongly treated as continuous.
The refined logic now keeps them correctly separated.

## Handling Survey Codes (7, 9, 77, 99…) (part of outliers handle) :

Telephone survey data often encodes missing answers as 7, 9, 77, or 99.
We created a function that detects and replaces such codes with NaN, but only when they are isolated outliers.

x_clean = replace_survey_codes_by_pattern(x)


Detection logic:
- Large numerical gap compared to previous values
- Value composed only of 7/9/0 digits
- Replacement occurs only when consistent with the pattern

✅ Benefit: keeps valid data while safely removing coded missing values.

## Removing Date- or ID-like Features

Columns such as IDAY, IMONTH, IYEAR, IDATE, or NUMADULT can distort normalization.
We used:

suspect_features = detect_date_or_id_features(x_train[:, continuous])


and removed only those from the continuous set:

for idx in suspect_features:
    if idx in continuous:
        continuous.remove(idx)


✅ Result: removed 5–15 irrelevant columns while preserving valid variables.

## Standardization of Continuous Features :

Applied only on continuous feature indices:

x_train_cont_std, x_test_cont_std = standardize(
    x_train[:, continuous_train],
    x_test[:, continuous_train]
)


Each variable is centered and scaled​

Verified results:

Mean ≈ 0, Std ≈ 1 for training data

No leakage between train and test sets.

## One-Hot Encoding of Categorical Features :

Implemented a NumPy-only encoder:

x_train_cat_encoded, cat_feature_names, skipped = one_hot_encode_numpy(
    x_train[:, categorical_train],
    feature_names=[feature_names[i] for i in categorical_train],
    max_categories=30
)


Settings:

max_categories=30 prevents feature explosion

Features with too many unique values are automatically skipped

Handles NaNs safely

✅ Result:

~101 categorical features → ~500 encoded columns

<5 % skipped due to high cardinality

Final structure:

[ standardized continuous | one-hot categorical | binary ]


## Final Concatenation

All cleaned blocks are merged together:

x_train_final = np.concatenate(
    [x_train_cont_std, x_train_cat_encoded, x_train_bin],
    axis=1
)


The same logic is applied to the test set once categorical encoding is aligned.


### Leila

Missingness categories:
  0%–5%: 115 features
  5%–20%: 28 features
  20%–50%: 31 features
  50%–100%: 147 features

--> DROP 50-100%

y_train unbalanced : use duplication to balance

---- y_train

- Can either use random oversampling (sample rows with replacement of the neg class) until reach a balanche of choise (30/70 for ex)

- Or use class weights for some functions : logit and reg logit

6 constant features to drop : [9, 11, 12, 18, 19, 22]
2 near constant features to drop : [280, 281]

Dist of features scales rather good, standardize/normalize later. 
Features with really big std : [  2   7   8  62  63 101 105 219 226 264]
--> [7,226] power laws, try log transform 

Features 7,8 are the same : drop one.

Only 6 continuous : [7, 8, 222, 226, 229, 253]

Features 222 and 229 are highly correlated but not enough to drop one for now. Could try to explore interaction?

# Conclusions from run.py on RAW dataset 

The models on raw features default to predicting the majority class, so they almost never flag positives.
F1 (harmonic mean of precision and recall) ≈ 0 shows that, at any chosen threshold, either precision or recall collapses—i.e., the classifier is not retrieving positives usefully.
ROC curve (TPR vs FPR) with ROC-AUC ≈ 0.50 indicates the model’s scores rank positives no better than random across all thresholds.
PR curve (precision vs recall) with PR-AUC ≈ base positive rate (~0.089) confirms near-random behavior in the highly imbalanced regime, where PR is the more informative curve.
High accuracy (~0.91) is misleading: it just reflects class imbalance, not true detection.
Different λ/γ give the same degenerate outcome because unprocessed scaling/missingness and imbalance prevent learning meaningful decision boundaries.
Net: Without preprocessing and imbalance handling, both thresholded performance (F1) and ranking ability (ROC/PR curves) show no signal.

# How to use preprocessing.py

Run "python preprocessing.py"

- Dans le main() "run_all_levels(args, only = None)" mettre only = None pour run tous les levels de preprocess sinon only = ['level3'] par ex.
- Voir le dict LEVELS qui definit chaque level de preprocessing. Ajouter la un nouveau level si besoin.
- Mettre le dossier 'preprocessed' dans GITIGNORE !!

PROBLEME : level3 encore des pbs a cause de drop_features --> deplacer le drop dans transform_with_state au debut de la fonction (avant impute etc..)

# 🧠 Preprocessing Pipeline Levels

This project explores multiple levels of preprocessing — from minimal cleaning to heavy feature engineering — to analyze how each stage affects model performance.

PHASE 1 : Fix model (M1 and M2) and vary preprocessing
M1 = logistic reg
M2 = regularized logit 

PHASE 2 : Identify best compo (Mi + Level i of prepro) and fine-tune the model hyperparams.
---

## 🧩 Level 0 — Baseline (Almost Raw)

**Goal:** Evaluate model performance with minimal preprocessing.

**Steps**
1. Drop only features that are **100% missing**.  
2. Impute missing values:  
   - Continuous → **mean**  
   - Categorical → **mode**  
3. **No scaling** (keep raw units).  
4. **No encoding** of categorical variables.  
5. **No balancing** (use class weights in model instead).

---

## ⚙️ Level 1 — Light Preprocessing

**Goal:** Perform minimal cleaning while preserving most features.

**Steps**
1. Drop features with **>50% missing values**.  
2. Drop **constant or near-constant** features.  
3. Impute missing values (mean or median).  
4. **Standardize** continuous features.  
5. Keep categorical features numeric (no one-hot).  
6. Skip oversampling for now.

---

## 🧱 Level 2 — Medium Preprocessing

**Goal:** Make features more uniform and interpretable for ML models.

**Steps**
1. Drop features with **>50% missing values**.  
2. Drop low-variance features.  
3. Impute missing values (mean/median).  
4. **Standardize** continuous features.  
5. **Add one-hot encoding** for categorical features:  
   - `min_count = 50`, `max_categories = 50`  
6. **Balance classes** via oversampling before training.

---

## 🧠 Level 3 — Heavy Preprocessing

**Goal:** Maximize data quality and expressivity for more complex models.

**Steps**
1. Drop features with **>30–40% missing values**.  
2. Drop low-variance and duplicate columns.  
3. Impute missing values:  
   - Continuous → **median**  
   - Categorical → **mode**  
4. **Clip outliers** (e.g., to 5th–95th percentile).  
5. **Standardize** continuous features.  
6. **One-hot encode** categoricals with more detail:  
   - `min_count = 10`, `max_categories = 100`  
7. Optional: feature selection by correlation to target.  
8. Apply **oversampling** or use class weights.

---

## 🧪 Level 4 — Experimental / Feature-Engineered

**Goal:** Explore feature creation and transformations beyond simple cleaning.

**Ideas**
- Add **interaction terms** (feature1 × feature2).  
- Compute **ratios or differences** between related features.  
- Add **polynomial terms** (e.g., squares).  
- Apply **log transform** to skewed continuous features.  
- Use **binned features** (quantile bins).  
- Try **target encoding** for high-cardinality categorical variables.
