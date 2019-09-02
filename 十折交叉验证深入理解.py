from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kf = KFold(n_splits=10, shuffle=False, random_state=None)


for train_index, val_index in kf.split(X):
    print("TRAIN_index:", train_index, "VAL_index:", val_index)
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]


from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
skf = StratifiedKFold(n_splits=2)
