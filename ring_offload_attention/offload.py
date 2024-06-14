import torch
from xformers.ops.fmha import (
    memory_efficient_attention_forward,
    memory_efficient_attention_backward, 
    memory_efficient_attention_partial,
    merge_attentions
)
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import update_out_and_lse

def forward(
    query, 
    keys, 
    values, 
    attn_biases, 
    scale,
):

    out = None
    lse = None
    for i in range(len(keys)):

        with torch.no_grad():
            block_out, block_lse = memory_efficient_attention_partial(
                    query = query,
                    key = keys[i],
                    value = values[i],
                    attn_bias = attn_biases[i] if attn_biases is not None else None,
                    scale = scale
                )
        out, lse = update_out_and_lse(
            out = out, 
            lse = lse, 
            block_out = block_out, 
            block_lse = block_lse,
        )

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

class Sdpa(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        q, 
        k, 
        v, 
        attn_bias,
        scale,
        chunk_size,
    ):
        k = k.contiguous()
        v = v.contiguous()

        q = q.to("cpu", non_blocking = True)
        k = k.to("cpu", non_blocking = True)
        v = v.to("cpu", non_blocking = True)
        
        q_chunks = q.chunk(chunk_size, dim = 2)
        k_chunks = k.chunk(chunk_size, dim = 2)
        v_chunks = v.chunk(chunk_size, dim = 2)
        if attn_bias is not None:
            attn_bias_chunks = attn_bias.chunk(chunk_size, dim = 2)

        out = None
        lse = None

        for i in range(len(q_chunks)):
            with torch.no_grad():
                block_out, block_lse = memory_efficient_attention_partial(
                    query = query,
                    key = keys[i],
                    value = values[i],
                    attn_bias = attn_biases[i] if attn_biases is not None else None,
                    scale = scale
                )
                out, lse = update_out_and_lse(
                    out = out, 
                    lse = lse, 
                    block_out = block_out, 
                    block_lse = block_lse,
                )

class Flash(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        q, 
        k, 
        v, 
        softmax_scale,
        causal,
        chunk_size,
    ):
        k = k.contiguous()
        v = v.contiguous()

        q = q.to("cpu", non_blocking = True)
        k = k.to("cpu", non_blocking = True)
        v = v.to("cpu", non_blocking = True)
        
        q_chunks = q.chunk(chunk_size, dim = 2)
        k_chunks = k.chunk(chunk_size, dim = 2)
        v_chunks = v.chunk(chunk_size, dim = 2)

        out = None
        lse = None
        results = []

        for i in range(len(q_chunks)):
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q_chunks[i],
                k_chunks[i],
                k_chunks[i],
                dropout_p = 0.0,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=(-1, 1),
                alibi_slopes=None,
                return_softmax=False,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            results.append(out)
        
        out = out.to(q.dtype)
        lse = lse.squeeze(dim=-1).transpose(1, 2)
        ctx.save_for_backward(q, k, v, out, lse)
        return torch.concat(out, dim = 2)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None

        
                

def offload_flash_func(
    q,
    k,
    v,
    attn_bias = None,
    scale = None,
    chunk_size = 4,
):
    """
    For Encoder-Decoder, eg T5,
        On Encoder self-attention,
            k: [batch_size, head_size, encoder_sequence_len, dim]
            v: [batch_size, head_size, encoder_sequence_len, dim]
            q: [batch_size, head_size, encoder_sequence_len, dim]
            attn_bias: [batch_size, head_size, encoder_sequence_len, encoder_sequence_len]

        On Decoder self-attention,
            k: [batch_size, head_size, decoder_sequence_len, dim]
            v: [batch_size, head_size, decoder_sequence_len, dim]
            q: [batch_size, head_size, decoder_sequence_len, dim]
            attn_bias: [batch_size, head_size, decoder_sequence_len, decoder_sequence_len]

        On Decoder cross-attention
            k: [batch_size, head_size, decoder_sequence_len, dim]
            v: [batch_size, head_size, encoder_sequence_len, dim]
            v: [batch_size, head_size, encoder_sequence_len, dim]
            attn_bias: [batch_size, head_size, decoder_sequence_len, encoder_sequence_len]

    For Decoder, eg Llama,
        On Decoder self-attention,
            k: [batch_size, head_size, decoder_sequence_len, dim]
            v: [batch_size, head_size, decoder_sequence_len, dim]
            q: [batch_size, head_size, decoder_sequence_len, dim]
            attn_bias: None
    """
    
    assert q.shape[2] % chunk_size == 0, "q sequence dimension must divisible by chunk size"
    assert k.shape[2] % chunk_size == 0, "k sequence dimension must divisible by chunk size"
    assert v.shape[2] % chunk_size == 0, "v sequence dimension must divisible by chunk size"


    


