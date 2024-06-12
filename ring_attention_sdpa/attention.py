import torch
import torch.distributed as dist
from xformers.ops.fmha import memory_efficient_attention_backward, memory_efficient_attention_partial
from .utils import RingComm, update_out_and_lse

