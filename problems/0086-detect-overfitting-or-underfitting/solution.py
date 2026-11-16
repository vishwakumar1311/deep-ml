import torch

def model_fit_quality(training_accuracy: torch.Tensor, test_accuracy: torch.Tensor) -> torch.Tensor:
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: torch.Tensor or float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: torch.Tensor or float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: torch.Tensor scalar, one of 1 (overfitting), -1 (underfitting), or 0 (good fit).
    """
    # Your code here
    train_acc = torch.as_tensor(training_accuracy, dtype=torch.float32)
    test_acc = torch.as_tensor(test_accuracy, dtype=torch.float32)
    
    gap = train_acc - test_acc

    # 1. Overfitting: strong training performance with a large drop on test data
    if train_acc >= 0.7 and gap > 0.1:
        return torch.tensor(1, device=train_acc.device)
    
    # 2. Underfitting: poor performance on both training and test sets
    elif train_acc < 0.7 and test_acc < 0.7:
        return torch.tensor(-1, device=train_acc.device)
    
    # 3. Good fit: high performance with low generalization gap
    else:
        return torch.tensor(0, device=train_acc.device)
    pass