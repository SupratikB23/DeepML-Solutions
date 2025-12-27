import torch
from typing import List


def transform_basis(B: List[List[float]], C: List[List[float]]) -> List[List[float]]:
    """Return the change-of-basis matrix **P = Câ»Â¹ B**.

    - *B*, *C* may be 2Ã2 or 3Ã3 nested lists.
    - Result is rounded to 4 decimals and returned as a nested list.
    """
    # Your implementation here
    b_t = torch.as_tensor(B, dtype=torch.float)
    c_t = torch.as_tensor(C, dtype=torch.float)

    x = torch.linalg.inv(c_t) @ b_t
    return x
    pass
