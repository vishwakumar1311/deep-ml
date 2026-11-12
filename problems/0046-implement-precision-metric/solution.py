import torch

def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the precision metric for binary classification.
    
    Precision is defined as the ratio of true positives to the sum of 
    true positives and false positives.
    
    Args:
        y_true: True binary labels (1D tensor)
        y_pred: Predicted binary labels (1D tensor)
    
    Returns:
        Precision value as a scalar tensor
    """
    # Your implementation here
    true_pos = ((y_true == 1) & (y_pred == 1)).sum().item()
    false_pos = ((y_true == 0) & (y_pred == 1)).sum().item()
    value = [true_pos/(true_pos + false_pos) if (true_pos + false_pos) !=0 else 0.0].pop()
    return value
    pass
