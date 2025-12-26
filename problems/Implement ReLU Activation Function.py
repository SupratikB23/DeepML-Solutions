import torch

def relu(z: torch.Tensor) -> torch.Tensor:
    """
    Implements the ReLU (Rectified Linear Unit) activation function using PyTorch.
    
    Args:
        z: Input tensor of any shape
    
    Returns:
        Tensor with ReLU applied element-wise: max(0, z)
    """
    # Your implementation here
    x = torch.relu(z)
    return x
    pass
