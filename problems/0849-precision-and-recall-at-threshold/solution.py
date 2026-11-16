import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    y_pred = [1 if score >= threshold else 0 for score in y_scores]
    true_pos = sum(1 for yt,yp in zip(y_true, y_pred) if yt == 1 and yp ==1)
    false_pos = sum(1 for yt,yp in zip(y_true, y_pred) if yt ==0 and yp ==1)
    false_neg = sum(1 for yt,yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    return [ precision, recall ]
    pass
