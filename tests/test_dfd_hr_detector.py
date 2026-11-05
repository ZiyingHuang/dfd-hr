from types import SimpleNamespace

import torch
from torch import nn

from training.detectors.dfd_hr_detector import DFDHRDetector


class _AlwaysDropRouter(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_full((hidden_states.size(0), 1), -100.0)


def test_features_encoder_returns_zero_spearman_when_all_samples_exit() -> None:
    detector = object.__new__(DFDHRDetector)
    nn.Module.__init__(detector)
    detector.config = {"backbone_config": {"layer": 24, "remain_layer": 20}}
    detector.backbone = SimpleNamespace(
        encoder=SimpleNamespace(
            config=SimpleNamespace(
                output_attentions=False,
                output_hidden_states=False,
                use_return_dict=True,
            ),
            layers=nn.ModuleList(nn.Identity() for _ in range(24)),
        )
    )
    detector.layer_router = nn.ModuleList(_AlwaysDropRouter() for _ in range(24))

    def passthrough_layer(
        idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor]:
        return (hidden_states,)

    detector.features_encoder_layer = passthrough_layer

    hidden_states = torch.randn(2, 4, 8)
    feature_global = torch.randn(2, 8)

    output, loss_spearman = detector.features_encoder(hidden_states, feature_global)

    assert output.last_hidden_state.shape == hidden_states.shape
    assert loss_spearman is not None
    assert loss_spearman.shape == torch.Size([])
    assert loss_spearman.item() == 0.0
