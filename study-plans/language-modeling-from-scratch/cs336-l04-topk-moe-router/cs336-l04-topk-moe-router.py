import torch

def topk_moe_router(x: torch.Tensor, w_router: torch.Tensor, 
                    expert_outputs: torch.Tensor, 
                    shared_output: torch.Tensor, k: int) -> dict:
    """
    Returns tensors in a dict: output, selected_experts, selected_weights, token_fraction, probability_mass.
    """

    z = x.float()
    # [T, E]
    scores = z @ w_router.float()
    # [T, E]
    sorted_scores, sorted_index = torch.sort(scores, dim=-1, descending=True, stable=True)
    # [T, K]
    selected_scores = sorted_scores[:, :k]
    selected_experts = sorted_index[:, :k]

    selected_weights = selected_scores.softmax(-1)

    Dout = expert_outputs.shape[-1]
    output_idx = selected_experts[:, :, None].expand(-1, -1, Dout)
    # [T, K, Dout]
    selected_outputs = torch.gather(expert_outputs, dim=1, index=output_idx)

    y = shared_output.float() + (selected_outputs.float() * selected_weights[:, :, None]).sum(dim=1)

    # [T, E] --> [T, K]
    hist = torch.zeros_like(scores).scatter(dim=1, index=selected_experts, value=1.0)
    f = hist.mean(0)
    m = scores.softmax(dim=-1).mean(0)

    return {"output":y.to(x.dtype),"selected_experts": selected_experts, 
            "selected_weights":selected_weights.to(x.dtype),"token_fraction":f.to(x.dtype),"probability_mass":m.to(x.dtype)}

