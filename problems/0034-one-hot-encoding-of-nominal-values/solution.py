import torch
from typing import Optional

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    # Hint: You can use torch.nn.functional.one_hot
    data_t = torch.tensor(x,dtype=torch.long)
    if n_col :
        return torch.nn.functional.one_hot(data_t,num_classes=n_col).float()
    else:
        return torch.nn.functional.one_hot(data_t).float()
    pass
