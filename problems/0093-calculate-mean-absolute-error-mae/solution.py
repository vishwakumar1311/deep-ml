import torch

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error between two tensors.

    Parameters:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: Mean Absolute Error
    """
    # Your code here
    y_true = y_true.float()
    y_pred = y_pred.float()
    MAE_value = torch.mean(torch.abs(y_true-y_pred))
    return float(MAE_value)
    pass