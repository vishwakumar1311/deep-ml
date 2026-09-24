import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here

    rng = np.random.default_rng(seed)
    
    n_samples = len(data)
    shuffled_indices = rng.permutation(n_samples)
    
    train_end = int(train_frac * n_samples)
    val_end = train_end + int(np.floor(validation_frac * n_samples))
    
    train_idx = shuffled_indices[:train_end]
    val_idx = shuffled_indices[train_end:val_end]
    test_idx = shuffled_indices[val_end:]
    
    return [data[train_idx], data[val_idx], data[test_idx]]
    pass