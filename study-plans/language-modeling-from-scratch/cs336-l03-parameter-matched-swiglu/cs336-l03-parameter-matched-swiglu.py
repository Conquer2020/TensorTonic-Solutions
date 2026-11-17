import torch


import torch.nn.functional as F
import math
def parameter_matched_swiglu(
    x: torch.Tensor, w_g: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, base_params: int,
) -> dict:
    """
    Returns output, hidden_width, and parameter_count in a dictionary.
    """
    d, H = w_g.shape
    h = min(H, max(1, math.floor(base_params / (3*d) + 0.5)))
    wg = w_g.float()[:, :h]
    wv = w_v.float()[:, :h]
    wo = w_o.float()[:h, :]
    xf = x.float()
    
    xg = torch.matmul(xf, wg)
    xg = F.silu(xg)
    xv = torch.matmul(xf, wv)
    o = torch.matmul(xg*xv, wo)

    return {"output": o.to(x.dtype), "hidden_width": h, "parameter_count": 3*d*h}
