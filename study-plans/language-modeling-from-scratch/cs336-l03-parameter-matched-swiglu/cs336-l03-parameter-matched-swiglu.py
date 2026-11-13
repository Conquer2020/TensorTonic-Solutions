import math
import torch

def parameter_matched_swiglu(x:torch.Tensor,w_g:torch.Tensor,w_v:torch.Tensor,w_o:torch.Tensor,base_params:int) -> dict:
    d=x.shape[-1];maximum=w_g.shape[-1]
    # Integer arithmetic implements floor(P/(3*d)+0.5), avoiding bankers rounding.
    h=min(maximum,max(1,(2*base_params+3*d)//(6*d)))
    z=x.float();gate=z@w_g[:,:h].float();up=z@w_v[:,:h].float()
    y=(gate*torch.sigmoid(gate)*up)@w_o[:h,:].float()
    return {'output':y.to(x.dtype),'hidden_width':h,'parameter_count':3*d*h}
