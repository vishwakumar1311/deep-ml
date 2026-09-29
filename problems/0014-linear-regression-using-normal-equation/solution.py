import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    # Your implementation here
    X_trans = X_t.transpose(0,1)
    x_inverse = (X_trans @ X_t).inverse()
    beta = (x_inverse @ X_trans @ y_t).round(decimals=4)
    return beta

    pass
