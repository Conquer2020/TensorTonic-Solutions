import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    z = x.float()
    y = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + epsilon)

    return (y * g.float()).to(x.dtype)
