import math
import torch

def topk_moe_router(x:torch.Tensor,w_router:torch.Tensor,expert_outputs:torch.Tensor,shared_output:torch.Tensor,k:int) -> dict:
    scores=x.float()@w_router.float()
    indices=scores.argsort(dim=-1,descending=True,stable=True)[:,:k]
    selected=scores.gather(1,indices).softmax(-1)
    width=expert_outputs.shape[-1]
    expert=expert_outputs.float().gather(1,indices[:,:,None].expand(-1,-1,width))
    out=shared_output.float()+(expert*selected[:,:,None]).sum(1)
    hits=torch.zeros_like(scores).scatter_(1,indices,1.)
    return {'output':out.to(x.dtype),'selected_experts':indices,'selected_weights':selected.to(x.dtype),'token_fraction':hits.mean(0).to(x.dtype),'probability_mass':scores.softmax(-1).mean(0).to(x.dtype)}
