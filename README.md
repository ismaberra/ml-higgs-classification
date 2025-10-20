[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)

## Project Overview

This repository contains a complete ML pipeline for a binary, highly imbalanced classification task. It includes:

- Preprocessing with several predefined levels in `preprocessing.py`
- Model training, cross-validation, and grid search in `run.py`
- Configuration-driven runs via `config.json`
- Parallelization using `ProcessPoolExecutor` for grid search and CV


## Setup

1) Python 3.10+ recommended.
2) Install dependencies (minimal):

```bash
pip install numpy matplotlib tqdm
```

3) Data layout:
- Raw CSVs in `dataset/` with `x_train.csv`, `y_train.csv`, `x_test.csv`, `sample_submission.csv`.
- Preprocessed matrices written to `preprocessed/<level>/` by `preprocessing.py`.


## Preprocessing

All levels are defined in `preprocessing.py` under the `LEVELS` dictionary. Each level specifies:

- `missing_threshold`, `drop_low_var`, `lowvar_threshold`
- imputers: `strategy_cont`, `strategy_cat`
- target encoding knobs: `target_encoding_min_count`, `target_encoding_max_categories`, `target_encoding_smoothing`
- duplicate handling: `drop_duplicates`

Run preprocessing across all levels (or a subset):

```bash
python preprocessing.py
```

Output matrices:
- `preprocessed/<level>/x_train.csv`
- `preprocessed/<level>/y_train.csv`
- `preprocessed/<level>/x_test.csv`


## Configuration (`config.json`)

Primary keys:

- `data_dir`: path to a preprocessed level (e.g., `preprocessed/level2`)
- `model`: one of `logistic`, `reg_logistic`, `logistic_weighted`, `reg_logistic_weighted`, `random_forest`
- `k`: folds for CV, `seed`
- `gamma`, `lambda_`, `max_iters`, `threshold`
- Search grids: `gamma_grid`, `lambda_grid`, `max_iters_grid`, `search_max_iters_grid` (preferred for CV), `threshold_grid`
- Parallelism: `n_jobs` (grid search), `cv_n_jobs` (folds)
- Output: `results_dir`, `tag`, `make_submission`

Notes:
- `threshold` is used for converting probabilities to labels. We default to 0.5 for this imbalanced task.
- If `final_max_iters` is set, it overrides the best `max_iters` found during grid search for the final training.


## Running Training and Evaluation

Basic run (uses `config.json`):

```bash
python run.py --config config.json
```

Override model or data dir:

```bash
python run.py --model reg_logistic --data_dir preprocessed/level2
```

Control CV folds and seed:

```bash
python run.py --k 5 --seed 1
```

Make a submission CSV for the test set:

```bash
python run.py --make_submission
```

All results are saved to `results/<model>_<tag>_<timestamp>/` with:
- `summary.json` (params, CV metrics, train curves)
- `*_train_roc.png`, `*_train_pr.png`
- optional `submission.csv`


## Grid Search

Enable by setting grids in `config.json` or CLI:

```bash
python run.py --model reg_logistic \
  --gamma_grid "0.05,0.08,0.1,0.15" \
  --lambda_grid "0.0003,0.0005,0.001,0.002,0.005" \
  --search_max_iters_grid "600" \
  --cv_n_jobs 2 --n_jobs 3
```

Best parameters are written into the run summary and used for final training. If `final_max_iters` is set in `config.json` or CLI, it takes precedence for the final fit.


## Parallelization

- Grid search parallelism: `n_jobs` (number of parameter combinations evaluated in parallel)
- CV parallelism: `cv_n_jobs` (folds evaluated in parallel)
- Uses `ProcessPoolExecutor` to bypass Python’s GIL for CPU-bound tasks

Tips for Apple Silicon (M1/M2):
- Set `n_jobs` + `cv_n_jobs` conservatively to avoid oversubscription (e.g., `n_jobs=3`, `cv_n_jobs=2` on 8-core CPUs)


## Thresholding and Metrics

- For imbalanced data, we use PR AUC for selection during grid search.
- We fix the classification threshold to 0.45, leveraging domain knowledge and avoiding leakage from re-optimizing it on training data after target encoding.


## Reproducibility

- Set `seed` in `config.json` or via CLI.
- Folds are stratified by label.

## Repository Structure

- `preprocessing.py`: defines levels, fits state, and writes preprocessed matrices
- `run.py`: training, CV, grid search, plotting, submission
- `implementations.py`: baseline algorithms
- `new_implementations.py`: optimized Random Forest and weighted logistic variants
- `helpers.py`: IO utilities
- `config.json`: main configuration for runs
- `eda.py`: helper functions for the exploratory data analysis
- `eda.ipynb`: Exploratory Data Analysis
- `report.pdf`: final report of the project
