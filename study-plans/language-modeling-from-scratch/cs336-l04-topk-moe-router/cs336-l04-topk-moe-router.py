
import torch

def topk_moe_router(x: torch.Tensor, w_router: torch.Tensor, 
                    expert_outputs: torch.Tensor, 
                    shared_output: torch.Tensor, k: int) -> dict:
    """
    Returns tensors in a dict: output, selected_experts, selected_weights, token_fraction, probability_mass.
    """
    # [T, D]
    z = x.float()
    # [T, D]x[D, E] = [T, E]
    score = z @ w_router.float()
    # [T, E]
    router_score, router_index = torch.sort(score, stable=True, descending=True, dim=-1)
    router_index = router_index.to(torch.int64)
    # [T, K]
    selected_experts = router_index[:, :k]
    selected_score = router_score[:, :k]

    # [T, E, Dout]  
    Dout = expert_outputs.shape[-1]
    # [T, K]
    selected_weights = selected_score.softmax(-1)

    # index must be same dim: [T, K, Dout]
    selected_expert = torch.gather(expert_outputs, dim=1, index = selected_experts[:,:,None].expand(-1, -1, Dout))

    # sum dim MUST is 1, not -1. [T, K, Dout] --> [T, Dout]
    y = shared_output.float() + (selected_weights[:, :, None] * selected_expert.float()).sum(dim=1)

    hist = torch.zeros_like(router_score).scatter_(1, selected_experts, 1.)

    return {"output": y.to(x.dtype), "selected_experts":selected_experts, 
            "selected_weights": selected_weights.to(x.dtype),"token_fraction": hist.mean(0).to(x.dtype), "probability_mass":score.softmax(-1).mean(0).to(x.dtype)}

    

