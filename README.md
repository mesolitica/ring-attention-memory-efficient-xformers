# ring-offload-attention

1. Ring Attention based on Memory Efficient Xformers, https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py#L529
2. Offload Attention based on Memory Efficient Xformers, https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py#L529
3. Offload Attention based on Flash Attention

Right now Offload Attention use more memory on short sequence length, still debugging :/
