import torch
from typing import List
import torch.nn.functional as F

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    sc = torch.as_tensor(scores, dtype=torch.float)
    x = F.log_softmax(sc, dim=0)
    return x
    pass
