import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities,
        each rounded to 4 decimals.
    """
    # Your implementation here
    sc = torch.as_tensor(scores, dtype=torch.float)
    x = F.softmax(sc, dim = 0)
    return x
    pass
