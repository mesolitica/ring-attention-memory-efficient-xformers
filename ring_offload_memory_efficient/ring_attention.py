import torch
import torch.distributed as dist
from xformers.ops.fmha import (
    memory_efficient_attention_backward, 
    memory_efficient_attention_partial,
    merge_attentions
)
from .utils import RingComm

def forward(
    process_group, 
    query, 
    key, 
    value, 
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
            output, lse = memory_efficient_attention_partial(
                query = query,
                key = key,
                value = value,
                attn_bias = attn_bias,
                scale = scale
            )
            if out is None:
                out = block_out
                lse = block_lse
            else:
                out, lse = merge_attentions([out, block_out], [lse, block_lse])
        
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v
    
    out = out.to(q.dtype)
    lse = lse.to(q.dtype)
    return out, lse
        
        