import torch
import torch.distributed as dist
from xformers.ops.fmha import (
    memory_efficient_attention_backward, 
    memory_efficient_attention_partial,
)
from .utils import RingComm, update_out_and_lse

def forward(
    process_group, 
    q, 
    k, 
    v, 
    attn_bias, 
    scale,
):
    comm = RingComm(process_group)

    out = None
    lse = None
    next_k, next_v = None, None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()
        
        if step <= comm.rank:
            block_out, block_lse = memory_efficient_attention_partial(
                query = q,
                key = k,
                value = v,
                attn_bias = attn_bias,
                scale = scale
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v
    
    out = out.to(q.dtype)
    lse = lse.to(q.dtype)
    return out, lse