[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)


EDA Conclusion :

Missingness categories:
  0%–5%: 115 features
  5%–20%: 28 features
  20%–50%: 31 features
  50%–100%: 147 features

--> DROP 50-100%

y_train unbalanced : use duplication to balance

6 constant features to drop : [9, 11, 12, 18, 19, 22]
2 near constant features to drop : [280, 281]

Dist of features scales rather good, standardize/normalize later. 
Features with really big std : [  2   7   8  62  63 101 105 219 226 264]
--> [7,226] power laws, try log transform 

Features 7,8 are the same : drop one.

Only 6 continuous : [7, 8, 222, 226, 229, 253]

Features 222 and 229 are highly correlated but not enough to drop one for now. Could try to explore interaction?