import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    # Your implementation here
    mean = data_t.mean(dim=0)
    std = data_t.std(dim=0, unbiased=False)
    std_data = (data_t - mean) / (std)

    min_vals = data_t.min(dim=0).values
    max_vals = data_t.max(dim=0).values
    norm_data = (data_t - min_vals) / (max_vals - min_vals)

    return (torch.round(std_data, decimals=4), torch.round(norm_data, decimals=4))
    pass
