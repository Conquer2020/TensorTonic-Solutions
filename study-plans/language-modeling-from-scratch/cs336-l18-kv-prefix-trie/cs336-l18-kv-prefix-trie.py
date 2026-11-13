import math
import torch

def kv_prefix_trie(cached_sequences:list,requests:list) -> dict:
    children=[{}]
    for seq in cached_sequences:
        node=0
        for token in seq:
            if token not in children[node]:
                children[node][token]=len(children);children.append({})
            node=children[node][token]
    matches=[]
    for seq in requests:
        node=0;length=0
        for token in seq:
            if token not in children[node]:break
            node=children[node][token];length+=1
        matches.append({'matched_length':length,'cache_node':node,'uncached_suffix':seq[length:]})
    return {'matches':matches}
