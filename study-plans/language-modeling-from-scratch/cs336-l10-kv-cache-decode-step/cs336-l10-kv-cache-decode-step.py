import math
import torch

def kv_cache_decode_step(query:torch.Tensor,new_key:torch.Tensor,new_value:torch.Tensor,key_cache:torch.Tensor,value_cache:torch.Tensor,num_query_heads:int,num_kv_heads:int) -> dict:
    kc=torch.cat((key_cache,new_key.unsqueeze(2)),dim=2)
    vc=torch.cat((value_cache,new_value.unsqueeze(2)),dim=2)
    b,hq,dim=query.shape;g=hq//num_kv_heads
    q=query.float().reshape(b,num_kv_heads,g,dim)
    scores=torch.matmul(q,kc.float().transpose(-1,-2))/math.sqrt(dim)
    out=torch.matmul(scores.softmax(-1),vc.float()).reshape(b,hq,dim)
    return {'output':out.to(query.dtype),'new_key_cache':kc,'new_value_cache':vc}
