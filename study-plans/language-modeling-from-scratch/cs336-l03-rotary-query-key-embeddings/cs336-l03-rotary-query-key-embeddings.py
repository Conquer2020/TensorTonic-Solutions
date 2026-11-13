import math
import torch

def rotary_embed(q: torch.Tensor,k: torch.Tensor,positions: torch.Tensor,inv_freq: torch.Tensor) -> dict:
    pos=positions.float()
    if pos.ndim==1:pos=pos[None,None,:,None]
    else:pos=pos[:,None,:,None]
    angle=pos*inv_freq.float().view(1,1,1,-1)
    c,s=angle.cos(),angle.sin()
    def rotate(x):
        a=x[...,0::2].float();b=x[...,1::2].float()
        return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(-2).to(x.dtype)
    return {'q_rotated':rotate(q),'k_rotated':rotate(k)}
