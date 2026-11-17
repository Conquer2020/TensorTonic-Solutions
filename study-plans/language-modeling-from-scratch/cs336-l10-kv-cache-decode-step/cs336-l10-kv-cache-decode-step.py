import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    B, Hq, D = query.shape 
    B, Hkv, S, D = key_cache.shape 
    G = Hq // Hkv

    q = query.float().view(B, Hkv, G, 1, D)

    if S == 0:
        k = new_key.float().view(B, Hkv, 1, 1, D)
        v = new_value.float().view(B, Hkv, 1, 1, D)
    else:
        old_k = key_cache.float().view(B, Hkv, 1, S, D)
        old_v = value_cache.float().view(B, Hkv, 1, S, D)
        new_k = new_key.float().view(B, Hkv, 1, 1, D)
        new_v = new_value.float().view(B, Hkv, 1, 1, D)

        k = torch.cat((old_k, new_k), dim=-2)
        v = torch.cat((old_v, new_v), dim=-2)

    logits = torch.matmul(q, k.transpose(-1, -2)) / D ** 0.5
    logits = torch.softmax(logits, dim=-1)

    o = torch.matmul(logits, v)


    o = o.contiguous().view(B, Hq, D).to(query.dtype)
    k = k.contiguous().view(B, Hkv, S+1, D).to(key_cache.dtype)
    v = v.contiguous().view(B, Hkv, S+1, D).to(value_cache.dtype)
    return {"output": o,"new_key_cache": k,"new_value_cache":v}