[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)


EDA Conclusion :

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