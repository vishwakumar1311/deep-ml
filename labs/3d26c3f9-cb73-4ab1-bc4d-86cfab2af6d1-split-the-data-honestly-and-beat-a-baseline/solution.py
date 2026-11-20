# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    N = len(X)
    n_train = int(round(N * train_frac))
    n_val = int(round(N * val_frac))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    if len(test_idx) == 0:
        return np.array([], dtype=np.float64), train_idx, val_idx, test_idx

    if len(train_idx) == 0:
        return np.zeros(len(test_idx), dtype=np.float64), train_idx, val_idx, test_idx

    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])

    weights, _, _, _ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    test_predictions = (X_test_aug @ weights).reshape(-1)

    return test_predictions, train_idx, val_idx, test_idx