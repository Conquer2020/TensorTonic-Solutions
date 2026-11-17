import torch


def rotary_embed(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor):

    if positions.ndim == 1:
        pos = positions.float()[None, None, :, None]
    else:
        pos = positions.float()[:, None, :, None]

    angle = pos * inv_freq.float().view(1, 1, 1, -1)
    cos = angle.cos()
    sin = angle.sin()

    def rotate(x):
        even = x[..., 0::2].float()
        odd = x[..., 1::2].float()
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)

    return {'q_rotated': rotate(q), 'k_rotated': rotate(k)}
        