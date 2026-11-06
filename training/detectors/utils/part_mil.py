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


def mask_part_labels(mask, part_masks, label, thr=0.2):
    """
    Per-part fake labels from the manipulation mask (FF++ provides one per fake frame).

    Args:
        mask: (B, H, W), (B, H, W, 1) or (B, 1, H, W) manipulation mask in [0, 1] (zeros for real frames).
        part_masks: (K, N) template on the G x G token grid (N = G * G).
        label: (B,) 0 real / 1 fake.
        thr: a part is fake if more than `thr` of its (soft) area is manipulated.
    Returns:
        targets (B, K) in {0, 1}, valid (B,) bool, overlap ratio (B, K).
        Real frames are always valid (all parts real). Fake frames are valid only if
        their mask is non-empty; otherwise only the bag loss applies to them.
    """
    if mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    if mask.shape[1] > 1:
        mask = mask.mean(1, keepdim=True)
    grid = int(round(part_masks.shape[1] ** 0.5))
    tok = F.adaptive_avg_pool2d(mask, grid).flatten(1)                     # B x N fraction manipulated
    pm = part_masks.float()
    ratio = tok @ pm.t() / pm.sum(1).clamp_min(1e-6)                        # B x K
    fake = label != 0
    targets = ((ratio > thr) & fake.unsqueeze(1)).float()
    valid = (~fake) | (tok.sum(1) > 0)
    return targets, valid, ratio


def part_mask_loss(part_logits, bag_logit, label, targets, valid, part_weight=1.0):
    """Bag BCE on every sample + per-part BCE against mask-derived part labels."""
    loss_bag = F.binary_cross_entropy_with_logits(bag_logit, label.float())
    if valid.any():
        loss_part = F.binary_cross_entropy_with_logits(part_logits[valid], targets[valid])
    else:
        loss_part = part_logits.sum() * 0.0
    return loss_bag + part_weight * loss_part


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
