import torch
from xformers.ops.fmha import (
    memory_efficient_attention_backward, 
    memory_efficient_attention_partial,
    merge_attentions
)

def forward(
    process_group, 
    query, 
    keys, 
    values, 
    attn_biases, 
    scale,
):

    out = None
    lse = None
    for i in range(len(keys)):
        output, lse = memory_efficient_attention_partial(
                query = query,
                key = keys[i],
                value = values[i],
                attn_bias = attn_biases[i],
                scale = scale
            )
        if out is None:
            out = block_out
            lse = block_lse
        else:
            out = merge_attentions([out, block_out], [lse, block_lse])

    return out, lse