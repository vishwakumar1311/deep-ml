import torch

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll using PyTorch.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    
    var_value = (n**2 -1)/12
    Exp_val = (n+1)/2
    return (Exp_val,var_value)
    pass