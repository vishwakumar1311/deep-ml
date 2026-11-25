import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    X_clean = X.copy()
    nan_X = np.isnan(X_clean)
    med = np.nanmedian(X_clean, axis=0)
    X_clean[nan_X] = np.take(med, np.where(nan_X)[1])
    return X_clean
    
    
    # TODO: Fill in NaN values
    
    return X_clean
