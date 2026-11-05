'''
# author: Jiamu Sun
# email: genisun@tencent.com
# date: 2026-05-28
# description: Class for the DFDHRDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from transformers import AutoProcessor, CLIPModel
from transformers.modeling_outputs import BaseModelOutput
from .utils import forward_query_loss as multiscale_forward_query_loss
from .utils import TokenRouter, LayerRouter, MoEAdapter, gumbel_sigmoid_sample, spearman_corr


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='dfd_hr')
class DFDHRDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config).vision_model
        self.query_token = nn.Parameter(torch.randn(1, 1, 768))
        self.query_attn = nn.MultiheadAttention(768, num_heads=4, batch_first=True)
        # Insert MoE adapters into both attn and mlp branches.
        self.adapters_attn = nn.ModuleList([
            MoEAdapter(
                D_features=1024,
                num_experts=4,
                k=4,
                mlp_ratio=0.25,
                expert_mlp_ratio=0.25,
                skip_connect=True,
                noise=False
            ) for _ in range(24)
        ])
        self.adapters_mlp = nn.ModuleList([
            MoEAdapter(
                D_features=1024,
                num_experts=4,
                k=4,
                mlp_ratio=0.25,
                expert_mlp_ratio=0.25,
                skip_connect=True,
                noise=False
            ) for _ in range(24)
        ])
        # Token-level and layer-level routers
        self.token_router = nn.ModuleList(TokenRouter(1024) for _ in range(24))
        self.layer_router = nn.ModuleList(LayerRouter(1024) for _ in range(24))
        self.capacity = config['backbone_config']['capacity']
        self.visual_projection = self.build_backbone(config).visual_projection
        self.head = nn.Linear(1536, 2)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        # Please download the checkpoints from the link below.
        if config['backbone_name'] == 'ViT-L/14' or config['backbone_name'] == 'ViT-L/14_proj':
            # use CLIP-large-14
            _, backbone = get_clip_visual(model_name="openai/clip-vit-large-patch14", backbone_name=config['backbone_name'])
        elif config['backbone_name'] == 'ViT-L/14-336px' or config['backbone_name'] == 'ViT-L/14-336px_proj':
            # use CLIP-large-14-336px
            _, backbone = get_clip_visual(model_name="openai/clip-vit-large-patch14-336", backbone_name=config['backbone_name'])
        return backbone

    def build_loss(self, config):
        # Prepare the loss function.
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def forward_features_loss(self, images: torch.tensor, feature_global=None) -> torch.tensor:
        hidden_states = self.backbone.embeddings(images)
        hidden_states = self.backbone.pre_layrnorm(hidden_states)

        encoder_outputs, loss_spearman = self.features_encoder(hidden_states, feature_global)

        output = encoder_outputs[0]
        B, L, D = output.size(0), output.size(1), output.size(2)
        feat = self.visual_projection(self.backbone.post_layernorm(output.reshape(-1, D))).reshape(B, L, -1)
        return [feat, output], loss_spearman

    def features(self, data_dict: dict) -> torch.tensor:
        
        outputs, loss_spearman = multiscale_forward_query_loss(
            self.forward_features_loss,
            self.query_token,
            self.query_attn,
            data_dict['image'],
            scales=[0.5, 1],
            max_split_size=224,
            num_prefix_token=1,
        )
        feat = outputs[:, 0, :]

        return feat, loss_spearman

    def features_encoder(self, inputs_embeds: torch.tensor, feature_global=None) -> torch.tensor:
        # Early Layer Pruning
        attention_mask = None
        causal_attention_mask = None
        output_attentions = self.backbone.encoder.config.output_attentions
        output_hidden_states = (
            self.backbone.encoder.config.output_hidden_states
        )
        return_dict = self.backbone.encoder.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        loss_spearman = hidden_states.new_zeros(())

        # MoD full-layer drop (gumbel exit).
        batch_size = hidden_states.size(0)
        # active_mask == 1 means the sample is still "active" and keeps computing layers.
        active_mask = torch.ones(batch_size, device=hidden_states.device)

        for idx, encoder_layer in enumerate(self.backbone.encoder.layers[0:self.config['backbone_config']['layer']]):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # MoD full-layer drop (gumbel exit).
            if idx >= self.config['backbone_config']['remain_layer'] and idx <= 22:
                hidden_states_cls = hidden_states[:, 0, :]

                logits = self.layer_router[idx](hidden_states_cls)  # shape: (batch_size, 1) or (batch_size, dim)
                if self.training:
                    # Sampled probability.
                    p_soft = gumbel_sigmoid_sample(logits)
                    p_hard = (p_soft > 0.5).float()
                    p = p_hard.detach() - p_soft.detach() + p_soft
                else:
                    p = (torch.sigmoid(logits) > 0.5).float()  # 0 or 1

                # Update active_mask: samples that already exited stay 0, others follow current p.
                active_mask = active_mask * p.squeeze(1)

                # For samples with active_mask == 1, compute the current layer; otherwise keep hidden_states unchanged.
                if active_mask.sum() > 0:
                    # Select active samples via index_select / mask.
                    active_indices = active_mask.nonzero(as_tuple=True)[0]

                    # Gather hidden_states for active samples.
                    active_hidden_states = hidden_states[active_indices]

                    if idx == self.config['backbone_config']['remain_layer'] and feature_global is not None:
                        active_feature_global = feature_global[active_indices]
                        # Compute the output of this layer.
                        layer_outputs, loss_spearman = self.features_encoder_layer_select(
                            idx,
                            active_hidden_states,
                            attention_mask,
                            causal_attention_mask,
                            output_attentions,
                            active_feature_global,
                        )
                    else:
                        # Compute the output of this layer.
                        layer_outputs = self.features_encoder_layer_select(
                            idx,
                            active_hidden_states,
                            attention_mask,
                            causal_attention_mask,
                            output_attentions,
                            None,
                        )
                    updated_active_hidden = layer_outputs[0]

                    # Scatter back to the original positions in hidden_states.
                    hidden_states[active_indices] = updated_active_hidden

                    if output_attentions:
                        # Keep attentions corresponding to active_indices; aggregation can happen later.
                        all_attentions = all_attentions + (layer_outputs[1],)
                else:
                    # All samples have exited, hidden_states no longer changes.
                    pass

            else:
                # idx < remain_layer: compute all samples normally.
                layer_outputs = self.features_encoder_layer(
                    idx,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        if feature_global is None:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), None
        else:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), loss_spearman

    def features_encoder_layer(self, idx: int, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = False):
        residual = hidden_states

        hidden_states = self.backbone.encoder.layers[idx].layer_norm1(hidden_states)
        hidden_states, attn_weights = self.backbone.encoder.layers[idx].self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        # Insert MoE adapter.
        hidden_states = self.adapters_attn[idx](hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.backbone.encoder.layers[idx].layer_norm2(hidden_states)
        hidden_states = self.backbone.encoder.layers[idx].mlp(hidden_states)

        # Insert MoE adapter.
        hidden_states = self.adapters_mlp[idx](hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def features_encoder_layer_select(self, idx: int, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = False, feature_global=None):

        b, s, d = hidden_states.shape
        # Compute importance weight for each token.
        weights = self.token_router[idx](hidden_states)

        if feature_global is not None:
            # 1) Normalize first.
            hidden_states_norm = F.normalize(hidden_states, p=2, dim=-1)       # (B, 257, 1024)
            feature_global_norm = F.normalize(feature_global, p=2, dim=-1)     # (B, 1024)

            # 2) Compute similarity via inner product.
            sim = torch.einsum('btd,bd->bt', hidden_states_norm, feature_global_norm)    # (B, 257)

            rank_loss = spearman_corr(weights, sim).mean()
            # The target is for rank_loss to approach 1 (the larger the better), so take the complement -> range [0, 2].
            loss_spearman = 1 - rank_loss

        # Decide how many tokens to process.
        k = max(1, int(self.capacity * s))
        top_k_values, top_k_indices = torch.topk(weights, k, dim=1, sorted=True)  # [B, k]

        # Build the mask via top_k_indices.
        selected_mask = torch.zeros_like(weights, dtype=torch.bool).to(hidden_states.device)
        selected_mask = selected_mask.scatter(1, top_k_indices, torch.ones_like(top_k_indices, dtype=torch.bool))

        # Initialize the output (unselected tokens stay unchanged).
        output = hidden_states.clone()
        selected_tokens = torch.zeros(b, k, d).to(hidden_states.device)
        selected_weights = torch.zeros(b, k).to(hidden_states.device)

        # Handle each sample individually.
        for i in range(b):
            current_mask = selected_mask[i]
            selected_tokens[i:i + 1, :, :] = hidden_states[i][current_mask].unsqueeze(0)
            selected_weights[i:i + 1, :] = weights[i][current_mask].unsqueeze(0)

        residual = selected_tokens

        hidden_states = self.backbone.encoder.layers[idx].layer_norm1(selected_tokens)
        hidden_states, attn_weights = self.backbone.encoder.layers[idx].self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        # Insert MoE adapter.
        hidden_states = self.adapters_attn[idx](hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.backbone.encoder.layers[idx].layer_norm2(hidden_states)
        hidden_states = self.backbone.encoder.layers[idx].mlp(hidden_states)

        # Insert MoE adapter.
        hidden_states = self.adapters_mlp[idx](hidden_states)

        # Weighted (after).
        hidden_states = residual + hidden_states
        selected_weights = F.softmax(selected_weights, dim=1)
        selected_weights = selected_weights.unsqueeze(-1)
        weighted_output = hidden_states * (1 + selected_weights)

        # Get the positions of the selected tokens.
        selected_indices = selected_mask.nonzero(as_tuple=True)

        # Scatter the processed selected tokens back to their original positions.
        output[selected_indices] = weighted_output.view(-1, d)
        outputs = (output,)

        if output_attentions:
            outputs += (attn_weights,)

        if feature_global is not None:
            return outputs, loss_spearman

        return outputs

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss_spearman = pred_dict['loss_spearman']
        loss = self.loss_func(pred, label) + 0.1 * loss_spearman
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # Compute metrics for batch data.
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # Get features via the backbone.
        features, loss_spearman = self.features(data_dict)
        # Get predictions via the classifier.
        pred = self.classifier(features)
        # Get the probability of the prediction.
        prob = torch.softmax(pred, dim=1)[:, 1]
        # Build the prediction dict for each output.
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'loss_spearman': loss_spearman}

        return pred_dict


def get_clip_visual(model_name="openai/clip-vit-base-patch16", backbone_name=None):
    if model_name == 'openai/clip-vit-large-patch14':
        processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
        model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    if model_name == 'openai/clip-vit-large-patch14-336':
        processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336')
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    return processor, model
