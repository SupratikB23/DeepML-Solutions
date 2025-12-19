import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2Ã2 matrix using PyTorch.
    Input: 2Ã2 tensor; Output: 1-D tensor with the two eigenvalues in ascending order.
    """
    # Your implementation here
    x = torch.linalg.eigvalsh(matrix)
    return torch.round(x)
    pass
