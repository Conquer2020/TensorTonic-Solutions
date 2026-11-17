import torch


def rotary_embed(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor):
    if positions.ndim == 1:
        pos = positions.float()[None, None, :, None]
    else:
        B, S = positions.shape
        pos = positions.float()[:, None, :, None]

    angles = pos * inv_freq.float().reshape(1, 1, 1, -1)
    cos, sin = angles.cos(), angles.sin()

    def rotate(x):
        a = x[..., 0::2].float()
        b = x[..., 1::2].float()

        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).to(x.dtype)
    return {'q_rotated': rotate(q), 'k_rotated': rotate(k)}
