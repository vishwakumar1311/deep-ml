import torch
import torch.nn.functional as F

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between actual and predicted values.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        RMSE value rounded to three decimal places.
    """
    # Write your code here
    if y_true.shape != y_pred.shape or y_true.numel() == 0 or y_pred.numel() == 0:
        raise ValueError("Shapes of y_true and y_pred must match and cannot be empty")
    RMSE_value = (torch.mean((y_true-y_pred)**2))**0.5
    return RMSE_value
    pass