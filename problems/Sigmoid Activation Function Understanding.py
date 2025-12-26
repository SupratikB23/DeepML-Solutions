import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z1 = torch.as_tensor(z, dtype=torch.float)
    x = torch.sigmoid(z1)
    return round(x.tolist(),4)
    pass
