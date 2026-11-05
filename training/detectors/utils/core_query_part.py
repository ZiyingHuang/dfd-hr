"""
Same as core_query_loss.forward_query_loss, but additionally returns the merged
patch-token map of the local (split) branch so that part queries can use it.
The returned `out` and `loss_spearman` are identical to the original function.
"""

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from .utils import split_chessboard, merge_chessboard


def forward_query_loss_with_local(model, query_token, query_attn, input, scales=None, img_sizes=None,
                                  max_split_size=None, resize_output_to_idx=0, num_prefix_token=1):
    assert input.dim() == 4 and input.shape[2] == input.shape[3]
    b, c, input_size, _ = input.shape

    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]
    max_split_size = max_split_size or input_size
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    outs_multiscale = []
    feature_global, _ = model(input_multiscale[0], None)
    outs_multiscale.append(feature_global[0])
    n_local = num_splits[1] ** 2
    feature_local, loss_spearman = model(input_multiscale[1],
                                         feature_global[1].repeat(n_local, 1, 1)[:, 0, :])
    outs_multiscale.append(feature_local[0])

    outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
    outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    outs_multiscale = [rearrange(out, 'b (h w) c -> b c h w',
                                 h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
                       for out in outs_multiscale]
    outs_multiscale = [merge_chessboard(out, num_split=num_split)
                       for num_split, out in zip(num_splits, outs_multiscale)]

    # Full-resolution local token map (e.g. B x 768 x 32 x 32 for 448 input).
    local_map = outs_multiscale[1]

    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                   mode='area').to(outs_multiscale[i].dtype)
                     for i in range(len(outs_multiscale))], dim=1)
    out = rearrange(out, 'b c h w -> b (h w) c')

    outs_prefix_multiscale[0] = torch.stack(outs_prefix_multiscale[0].split(b, dim=0), dim=0).mean(dim=0)
    split_token_bs = torch.stack(outs_prefix_multiscale[1].split(b, dim=0), dim=0).permute(2, 1, 0, 3).squeeze(0)
    query_token_bs = query_token.repeat(b, 1, 1)
    global_token, _ = query_attn(query_token_bs, split_token_bs, split_token_bs, need_weights=False)
    outs_prefix_multiscale[1] = global_token
    out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
    out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out, loss_spearman, local_map
