import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute inverse of a 2Ã2 matrix using PyTorch.
    Input can be Python list, NumPy array, or torch Tensor.
    Returns a 2Ã2 tensor or None if the matrix is singular.
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    if torch.linalg.det(m) == 0:
        return None
    else:
        return torch.linalg.inv(m)
    pass
