import torch


def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    values = x.float()
    gain = g.float()
    mean_square = values.square().mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + epsilon)
    normalized = values * inverse_rms
    return (normalized * gain).to(x.dtype)
