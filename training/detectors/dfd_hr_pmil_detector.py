'''
DFD-HR + Part-aware MIL Fusion (PMIL).

Relative to DFDHRDetector (all original modules are inherited unchanged):
  1. features() additionally returns the local-branch patch-token map (B x 768 x 32 x 32),
     which the original model never uses for classification.
  2. K part queries (eyes/brows, nose, mouth, face-boundary band, cheek skin) attend to those
     tokens under a fixed template of soft part masks; each part has its own fake logit.
  3. Part logits are aggregated as a MIL bag (smooth max) and trained with part_mil_loss.
  4. Final fake logit = (1 - alpha) * original DFD-HR logit + alpha * bag logit.

Two training regimes (config `part_mil`):
  - freeze_base: true  + init_from: official DFD-HR ckpt  -> only the PMIL branch is trained
    on top of the frozen, fully trained DFD-HR (fast; the original branch is bit-identical).
  - freeze_base: false (+ optional init_from)             -> end-to-end training.
'''

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from detectors import DETECTOR
from .dfd_hr_detector import DFDHRDetector
from .utils.core_query_part import forward_query_loss_with_local
from .utils.part_mil import PartMILFusion, mask_part_labels, part_mask_loss, part_mil_loss
from .utils.part_prior import PART_NAMES

logger = logging.getLogger(__name__)


def _normalize_ckpt_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        key = key.replace('module.', '')
        key = key.replace('base_model.', 'backbone.')
        key = key.replace('classifier.', 'head.')
        normalized[key] = value
    return normalized


@DETECTOR.register_module(module_name='dfd_hr_pmil')
class DFDHRPMILDetector(DFDHRDetector):
    def __init__(self, config):
        super().__init__(config)
        pcfg = config['part_mil']
        self.pcfg = pcfg

        template_path = Path(pcfg['template_path'])
        if not template_path.is_file():
            raise FileNotFoundError(
                f'part template not found: {template_path} '
                '(build it with scripts/build_part_template.py)')
        masks = np.load(template_path).astype(np.float32)                 # 5 x G x G
        parts = pcfg.get('parts', PART_NAMES)
        masks = masks[[PART_NAMES.index(p) for p in parts]]
        if pcfg.get('uniform_mask', False):                               # ablation: free queries
            masks = np.ones_like(masks)
        self.grid = masks.shape[-1]
        self.register_buffer('part_masks', torch.from_numpy(masks).flatten(1), persistent=False)

        self.part_fusion = PartMILFusion(
            dim=768,
            num_parts=len(parts),
            num_heads=pcfg.get('num_heads', 8),
            tau=pcfg.get('tau', 0.5),
            mask_eps=pcfg.get('mask_eps', 0.02),
            learnable_tau=pcfg.get('learnable_tau', False),
        )
        self.mil_weight = pcfg.get('mil_weight', 0.5)
        self.real_weight = pcfg.get('real_weight', 1.0)
        self.fuse_mode = pcfg.get('fuse', 'avg')          # avg | orig | part
        self.fuse_alpha = pcfg.get('fuse_alpha', 0.5)
        self.aggregate_mode = pcfg.get('aggregate', 'lse')  # lse | mean
        self.use_mil_loss = pcfg.get('use_mil_loss', True)
        self.freeze_base = pcfg.get('freeze_base', False)
        # 'mil': weak MIL supervision; 'mask': per-part labels from the FF++ manipulation masks
        self.supervision = pcfg.get('supervision', 'mil')
        self.part_label_thr = pcfg.get('part_label_thr', 0.2)
        self.part_weight = pcfg.get('part_weight', 1.0)
        self.base_lr_scale = pcfg.get('base_lr_scale', 1.0)
        if self.supervision == 'mask' and not config.get('with_mask', False):
            raise ValueError("part_mil.supervision=mask requires with_mask: true")

        # Load the official / previous DFD-HR weights only when training;
        # evaluation loads the full checkpoint with strict=True anyway.
        init_from = pcfg.get('init_from')
        if init_from and config.get('mode') == 'train':
            self._load_base(init_from)
        if self.freeze_base:
            for name, param in self.named_parameters():
                param.requires_grad = name.startswith('part_fusion.')
        else:
            # Fine-tune only the DFD-HR modules of the last layers: earlier layers then build no
            # autograd graph (CLIP is frozen), which keeps ViT-L x 5 crops within 24 GB GPUs.
            self.unfreeze_from_layer = pcfg.get('unfreeze_from_layer', 0)
            if self.unfreeze_from_layer > 0:
                per_layer = ('adapters_attn.', 'adapters_mlp.', 'token_router.', 'layer_router.')
                for name, param in self.named_parameters():
                    if name.startswith(per_layer) and int(name.split('.')[1]) < self.unfreeze_from_layer:
                        param.requires_grad = False

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'[PMIL] parts={parts} grid={self.grid} freeze_base={self.freeze_base} '
                    f'init_from={init_from} trainable_params={n_train}')
        print(f'[PMIL] parts={parts} grid={self.grid} freeze_base={self.freeze_base} '
              f'supervision={self.supervision} base_lr_scale={self.base_lr_scale} '
              f'trainable_params={n_train}')

    def optimizer_param_groups(self, lr):
        """PMIL branch at the base lr; the (pre-trained) DFD-HR modules at lr * base_lr_scale."""
        new, base = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                (new if name.startswith('part_fusion.') else base).append(param)
        groups = [{'params': new, 'lr': lr}]
        if base:
            groups.append({'params': base, 'lr': lr * self.base_lr_scale})
        return groups

    def _load_base(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state = _normalize_ckpt_keys(ckpt.get('state_dict', ckpt))
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not k.startswith('part_fusion.')]
        if bad_missing or unexpected:
            raise RuntimeError(f'init_from {path}: missing={bad_missing[:10]} unexpected={unexpected[:10]}')
        print(f'[PMIL] loaded base weights from {path} ({len(state)} tensors)')

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, 'freeze_base', False) and mode:
            # Frozen DFD-HR runs exactly as at inference (deterministic layer exit, no dropout).
            for name, child in self.named_children():
                if name != 'part_fusion':
                    child.eval()
        return self

    def features(self, data_dict: dict):
        outputs, loss_spearman, local_map = forward_query_loss_with_local(
            self.forward_features_loss,
            self.query_token,
            self.query_attn,
            data_dict['image'],
            scales=[0.5, 1],
            max_split_size=224,
            num_prefix_token=1,
        )
        return outputs[:, 0, :], loss_spearman, local_map

    def forward(self, data_dict: dict, inference=False) -> dict:
        if self.freeze_base:
            with torch.no_grad():
                feat, loss_spearman, local_map = self.features(data_dict)
                pred = self.classifier(feat)
        else:
            feat, loss_spearman, local_map = self.features(data_dict)
            pred = self.classifier(feat)                                   # B x 2

        if local_map.shape[-1] != self.grid:
            local_map = F.adaptive_avg_pool2d(local_map.float(), self.grid)
        tokens = local_map.flatten(2).transpose(1, 2)                      # B x N x 768
        part_logits, bag_logit, _, part_attn = self.part_fusion(tokens, self.part_masks)
        if self.aggregate_mode == 'mean':
            bag_logit = part_logits.mean(-1)

        orig_logit = pred[:, 1] - pred[:, 0]
        if self.fuse_mode == 'orig':
            final_logit = orig_logit
        elif self.fuse_mode == 'part':
            final_logit = bag_logit
        else:
            final_logit = (1 - self.fuse_alpha) * orig_logit + self.fuse_alpha * bag_logit

        # softmax(cls)[:, 1] == sigmoid(final_logit): trainer / metric code stays unchanged.
        cls = torch.stack([-final_logit / 2, final_logit / 2], dim=1)
        return {
            'cls': cls, 'prob': torch.sigmoid(final_logit), 'feat': feat,
            'cls_orig': pred, 'part_logits': part_logits, 'bag_logit': bag_logit,
            'part_attn': part_attn, 'loss_spearman': loss_spearman,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        mask = data_dict.get('mask')
        if self.supervision == 'mask' and mask is not None:
            targets, valid, _ = mask_part_labels(mask, self.part_masks, label, self.part_label_thr)
            loss_part = part_mask_loss(pred_dict['part_logits'], pred_dict['bag_logit'], label,
                                       targets, valid, self.part_weight)
        elif self.use_mil_loss:
            loss_part = part_mil_loss(pred_dict['part_logits'], pred_dict['bag_logit'], label,
                                      real_weight=self.real_weight)
        else:  # ablation: plain BCE on the bag only
            loss_part = F.binary_cross_entropy_with_logits(pred_dict['bag_logit'], label.float())
        loss_cls = self.loss_func(pred_dict['cls_orig'], label)
        if self.freeze_base:
            loss = loss_part
        else:
            loss_sp = pred_dict['loss_spearman']
            loss_sp = loss_sp if loss_sp is not None else loss_cls.new_zeros(())
            loss = loss_cls + 0.1 * loss_sp + self.mil_weight * loss_part
        return {'overall': loss, 'cls': loss_cls.detach(), 'part': loss_part.detach()}
