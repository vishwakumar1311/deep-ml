import torch

def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the R-squared (R²) coefficient of determination using PyTorch.

    Args:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    SSR = torch.sum((y_true - y_pred)**2)
    SST = torch.sum((y_true - torch.mean(y_true))**2)
    return float(1 - SSR / SST)
    pass