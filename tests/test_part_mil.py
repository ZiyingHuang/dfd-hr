import pytest

torch = pytest.importorskip('torch')

from training.detectors.utils.part_mil import PartMILFusion, part_mil_loss  # noqa: E402


def test_shapes_and_mask_restricts_attention():
    torch.manual_seed(0)
    B, N, D, K = 3, 64, 32, 2
    module = PartMILFusion(dim=D, num_parts=K, num_heads=4, mask_eps=1e-4)
    masks = torch.zeros(K, N)
    masks[0, :10] = 1
    masks[1, 30:] = 1
    part_logits, bag, z, attn = module(torch.randn(B, N, D), masks)
    assert part_logits.shape == (B, K) and bag.shape == (B,) and z.shape == (B, K, D)
    assert attn.shape == (B, K, N)
    assert attn[:, 0, :10].sum(-1).min() > 0.99
    assert attn[:, 1, 30:].sum(-1).min() > 0.99


def test_bag_is_smooth_max():
    module = PartMILFusion(dim=8, num_parts=4, num_heads=2, tau=0.1)
    logits = torch.tensor([[-3.0, -3.0, 4.0, -3.0]])
    bag = module.aggregate(logits)
    assert 2.5 < bag.item() <= 4.0


def test_per_image_masks_and_loss_backward():
    module = PartMILFusion(dim=16, num_parts=3, num_heads=2)
    tokens = torch.randn(4, 20, 16, requires_grad=True)
    part_logits, bag, _, _ = module(tokens, torch.rand(4, 3, 20))
    loss = part_mil_loss(part_logits, bag, torch.tensor([0, 1, 1, 0]))
    loss.backward()
    assert torch.isfinite(loss) and module.part_query.grad is not None
    only_fake = part_mil_loss(part_logits.detach(), bag.detach(), torch.tensor([1, 1, 1, 1]))
    assert torch.isfinite(only_fake)
