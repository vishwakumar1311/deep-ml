import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    # Your implementation here
    data_mean = data_t.mean(dim=0,keepdim=True)
    data_var = data_t.var(dim=0,keepdim=True)
    data_std = data_t.std(dim=0,correction=0,keepdim =True)
    z_score = ((data_t-data_mean)/data_std).round(decimals=4)
    
    data_max,indices = data_t.max(dim=0,keepdim=True)
    data_min,indices = data_t.min(dim=0,keepdim=True)
    minmaxnorm = (data_t-data_min)/(data_max-data_min)
    return (z_score,minmaxnorm)
    pass
