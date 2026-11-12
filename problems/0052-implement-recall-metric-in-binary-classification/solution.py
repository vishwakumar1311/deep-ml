import torch

def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Tensor of true binary labels (0 or 1)
        y_pred: Tensor of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    true_pos = ((y_true ==1) & (y_pred ==1)).sum().item()
    false_neg = ((y_true ==1) & (y_pred ==0)).sum().item()
    value = [true_pos/(true_pos + false_neg) if (true_pos + false_neg) != 0 else 0.0].pop()
    return value
    pass