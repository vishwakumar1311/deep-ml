import torch
from typing import Union

def accuracy_score(y_true: Union[torch.Tensor, list, "np.ndarray"],
                   y_pred: Union[torch.Tensor, list, "np.ndarray"]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
    # Your implementation here
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.as_tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.as_tensor(y_pred)

    true_tensor = torch.eq(y_true,y_pred)
    value =(true_tensor==True).sum().item()/true_tensor.shape[0]
    return value
    pass
