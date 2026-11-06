import torch

def train_logreg(X: torch.Tensor, y: torch.Tensor, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    """
    Train logistic regression using gradient descent with BCE loss.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features) as Tensor
        y: Binary labels of shape (n_samples,) as Tensor
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations
    
    Returns:
        Tuple of (coefficients, losses) where:
        - coefficients: List of learned weights (bias first, then feature weights), rounded to 4 decimals
        - losses: List of BCE loss values at each iteration, rounded to 4 decimals
    
    Notes:
        - Initialize all coefficients to zero
        - Add bias column as FIRST column of X
        - Use sum-based BCE loss: -sum(y*log(p) + (1-y)*log(1-p))
    """
    # Your code here
    n_samples, n_features = X.shape
    ones_tensor = torch.ones(X.shape[0], dtype=X.dtype)
    X = torch.cat([ones_tensor.unsqueeze(1),X], dim=1)
    weights = torch.zeros(n_features + 1, dtype=X.dtype)
    losses = []
    for _ in range(iterations):
        linear = X @ weights
        predictions = torch.sigmoid(linear)
        loss = -torch.sum(y*torch.log(predictions) + (1-y)*torch.log(1-predictions)).item()
        losses.append(loss)
        gradient = X.T @ (predictions - y)
        weights -= learning_rate * gradient
        rounded_weights = [round(w, 4) for w in weights.tolist()]
        rounded_losses = [round(l, 4) for l in losses]
    return rounded_weights, rounded_losses
    pass