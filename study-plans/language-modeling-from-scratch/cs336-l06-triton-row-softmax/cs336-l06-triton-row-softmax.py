import torch
import triton
import triton.language as tl



@triton.jit
def row_softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK_SIZE):
    '''
    return nothing; writes normalized rows through out_ptr
    '''
    row = tl.program_id(0)
    lane = tl.arange(0, 1024)

    ma = tl.full((), float('-inf'), tl.float32)

    for block in range(tl.cdiv(n_cols, 1024)):
        col = block * 1024 +  lane 

        x = tl.load(x_ptr + row * row_stride + col, mask= col < n_cols, other=float('-inf')).to(tl.float32)
        ma = tl.maximum(ma, tl.max(x, axis=0))

    denom = tl.full((), 0., tl.float32)

    for block in range(tl.cdiv(n_cols, 1024)):
        col = block * 1024 + lane 

        x = tl.load(x_ptr + row * row_stride + col, mask= col < n_cols, other=float('-inf')).to(tl.float32)
        denom += tl.sum(tl.exp(x - ma), axis=0)

    for block in range(tl.cdiv(n_cols, 1024)):
        col = block * 1024 + lane 
        x = tl.load(x_ptr + row * row_stride + col, mask= col < n_cols, other=float('-inf')).to(tl.float32)

        # NOTE: MUST be n_cols, not row_stride(only for load func). NOT forget tl.exp()
        tl.store(out_ptr + row * n_cols + col, tl.exp(x - ma) / denom, mask = col < n_cols)
    
def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x must be a nonempty-width matrix")
    if x.shape != out.shape or x.device != out.device or x.dtype != out.dtype:
        raise ValueError("x and out must have matching shape, device, and dtype")
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("x must be a supported CUDA tensor")
    if x.stride(1) != 1 or not out.is_contiguous():
        raise ValueError("input columns and output must be contiguous")
    if x.shape[1] > 65536:
        raise ValueError("row width exceeds the supported limit")
    if x.shape[0] == 0:
        return
    kernel_x = x
    kernel_out = out
    if x.dtype == torch.bfloat16:
        kernel_x = x.to(torch.float32)
        kernel_out = torch.empty_like(x, dtype=torch.float32)
    block_size = triton.next_power_of_2(x.shape[1])
    grid = (x.shape[0],)
    row_softmax_kernel[grid](
        kernel_x,
        kernel_out,
        kernel_x.stride(0),
        x.shape[1],
        BLOCK_SIZE=block_size,
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
