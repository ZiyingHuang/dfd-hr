"""Part-aware MIL Fusion (PMIL).

K learnable part queries attend to the local-branch patch tokens; the attention
logits of query k are biased by log(M_k) so each query reads (softly) only its
own facial part. Each part gets an independent linear verifier, and the part
logits are aggregated as a multiple-instance bag:

    real face  <=> every part is real
    fake face  <=> at least one part is fake  (smooth max over parts)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartMILFusion(nn.Module):
    def __init__(self, dim=768, num_parts=5, num_heads=8, tau=0.5, mask_eps=0.02,
                 learnable_tau=False):
        super().__init__()
        self.K = num_parts
        self.mask_eps = mask_eps
        self.part_query = nn.Parameter(torch.randn(num_parts, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # One independent linear verifier per part (stored as one grouped weight).
        self.head_w = nn.Parameter(torch.randn(num_parts, dim) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(num_parts))
        log_tau = torch.tensor(math.log(tau))
        if learnable_tau:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer('log_tau', log_tau, persistent=False)

    def aggregate(self, part_logits):
        tau = self.log_tau.exp()
        return tau * torch.logsumexp(part_logits / tau, dim=-1) - tau * math.log(self.K)

    def forward(self, tokens, part_masks):
        """
        Args:
            tokens: (B, N, D) patch tokens.
            part_masks: (K, N) shared template, or (B, K, N) per-image masks, values in [0, 1].
        Returns:
            part_logits (B, K), bag_logit (B,), part_feats (B, K, D), attn (B, K, N)
        """
        B = tokens.size(0)
        q = self.part_query.unsqueeze(0).expand(B, -1, -1)
        bias = torch.log(part_masks.float().clamp_min(self.mask_eps)).to(tokens.dtype)
        if bias.dim() == 3:  # per-image masks -> (B * heads, K, N)
            bias = bias.repeat_interleave(self.attn.num_heads, dim=0)
        z, attn = self.attn(q, tokens, tokens, attn_mask=bias, need_weights=True)
        z = self.norm(z)
        part_logits = (z * self.head_w.unsqueeze(0)).sum(-1) + self.head_b
        return part_logits, self.aggregate(part_logits), z, attn


def part_mil_loss(part_logits, bag_logit, label, real_weight=1.0):
    """
    - bag BCE on every sample (real -> 0, fake -> 1)
    - real samples: every part logit -> 0 (all parts real)
    - fake samples: no per-part target (we don't know which part is forged)
    """
    y = label.float()
    loss_bag = F.binary_cross_entropy_with_logits(bag_logit, y)
    real = label == 0
    if real.any():
        p = part_logits[real]
        loss_real = F.binary_cross_entropy_with_logits(p, torch.zeros_like(p))
    else:
        loss_real = part_logits.sum() * 0.0
    return loss_bag + real_weight * loss_real
