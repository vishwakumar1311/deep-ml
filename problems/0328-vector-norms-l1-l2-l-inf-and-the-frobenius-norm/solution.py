import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    match norm_type:
        case 'l1':
            result_l1 =0.0
            for element in arr.flatten():
                result_l1 += abs(element.item())
            return result_l1
        case 'l2' :
            arr_flat = arr.flatten()
            result_l2 = torch.linalg.vector_norm(arr_flat.float())
            return result_l2.item()
        case 'linf':
            result = arr.float().flatten().abs().max()
            return result.item()
        case 'frobenius':
            result_frob = 0.0
            if arr.ndim == 2:
                 result_frob = torch.linalg.vector_norm(arr.float(),dim = None)
                 return  result_frob.item()
            else:
                raise ValueError("Frobenius norm requires a 2D tensor.")

    pass
