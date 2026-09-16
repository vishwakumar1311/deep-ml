import torch

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    if not torch.is_tensor(data):
        t =torch.as_tensor(data, dtype = torch.float32)
    else:
        t = data.clone().detach().to(dtype = torch.float32)

    t = t.flatten()
    mean = torch.mean(t).item()
    median = torch.quantile(t,q=0.5).item()
    mode = int(torch.mode(t).values.item())
    var = torch.var(t, unbiased=False).item()
    std_dn = torch.std(t, unbiased=False).item()
    points = torch.tensor([0.25,0.50,0.75], dtype = torch.float32)
    quantiles= torch.quantile(t, points)
    p25= quantiles[0].item()
    p50 = quantiles[1].item()
    p75 = quantiles[2].item()
    iqr = p75-p25

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": var,
        "standard_deviation": std_dn,
        "25th_percentile": p25,
        "50th_percentile":p50,
        "75th_percentile":p75,
        "interquartile_range": iqr
    }
    pass