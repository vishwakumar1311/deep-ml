import torch

def confusion_matrix(data: list) -> torch.Tensor:
    """
    Generate a 2x2 confusion matrix for binary classification.

    Args:
        data: A list of [y_true, y_pred] pairs for binary labels (0 or 1)

    Returns:
        A 2x2 torch.Tensor confusion matrix arranged as [[TP, FN], [FP, TN]]
    """
    y_true = torch.tensor([row[0] for row in data])
    y_pred = torch.tensor([row[1] for row in data])
    true_pos = ((y_true == 1) & (y_pred == 1)).sum().item()
    false_neg = ((y_true ==1) & (y_pred ==0)).sum().item()
    false_pos = ((y_true == 0) & (y_pred == 1)).sum().item()
    true_neg = ((y_true ==0) & (y_pred ==0)).sum().item()
    return torch.tensor([[true_pos,false_neg],[false_pos,true_neg]])
    pass
