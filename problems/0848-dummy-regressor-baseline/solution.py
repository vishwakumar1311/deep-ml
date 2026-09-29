import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    y_np = np.array(y_train)
    match strategy:
        case 'mean':
            return [y_np.mean(axis=0)]*n_test
        case 'median':
            return [np.median(y_np)]*n_test
        case 'quantile':
            return [np.quantile(y_np,quantile)]*n_test
        case 'constant':
            return [constant]*n_test
    pass
