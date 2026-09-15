import torch
import torch.nn.functional as F

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # Implement your code here
    if not(v1.size()== v2.size()):
        return 0.0
    
    dotprod = v1 @ v2
    v1_norm = torch.sqrt(torch.sum(v1**2))
    v2_norm = torch.sqrt(torch.sum(v2**2))
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    denominator = v1_norm*v2_norm
    result = dotprod/denominator
    return result.item()
    pass