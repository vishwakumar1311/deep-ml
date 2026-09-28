import torch

def min_max(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A tensor of numerical values
    
    Returns:
        A new tensor with values normalized to [0, 1]
    """
    # Your code here
    x_result = torch.tensor(x,dtype = torch.float32)
    Minimum,indices = x_result.min(dim=0,keepdim=True)
    Maximum,indices = x_result.max(dim=0,keepdim=True)
    norm = (x_result-Minimum)/(Maximum-Minimum)
    return norm.round(decimals=4)

    pass