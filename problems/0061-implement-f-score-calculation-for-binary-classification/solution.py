import torch

def f_score(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float) -> float:
    """
    Calculate F-Score for a binary classification task.

    :param y_true: torch.Tensor of true labels (binary)
    :param y_pred: torch.Tensor of predicted labels (binary)
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """

    true_pos = ((y_true == 1) & (y_pred == 1)).sum().item()
    false_pos = ((y_true == 0) & (y_pred == 1)).sum().item()
    false_neg = ((y_true ==1) & (y_pred ==0)).sum().item()
    precision = [true_pos/(true_pos + false_pos) if (true_pos + false_pos) != 0 else 0.0].pop()
    Recall = [true_pos/(true_pos + false_neg) if (true_pos + false_neg) != 0 else 0.0].pop()
    value = (1+ beta**2) * (precision * Recall) / (beta**2 * precision + Recall) if (beta**2 * precision + Recall) != 0 else 0.0
    return value
    pass
