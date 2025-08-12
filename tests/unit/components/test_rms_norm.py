import torch

from transformer_lens.components.rms_norm import RMSNorm
from transformer_lens.hook_points import HookPoint


def test_rms_norm_hook_normalized_fires_on_output():
    cfg = {
        "d_model": 4,
        "n_layers": 1,
        "n_ctx": 8,
        "d_head": 4,
        "act_fn": "relu",
        "dtype": torch.float32,
    }
    layer: RMSNorm = RMSNorm(cfg)
    layer.w.data = torch.randn_like(layer.w)

    captured: dict[str, torch.Tensor] = {}

    def save_activation(
        activation: torch.Tensor, *, hook: HookPoint
    ) -> torch.Tensor:  # noqa: ARG001
        captured["act"] = activation.detach().clone()
        return activation

    layer.hook_normalized.add_hook(save_activation, dir="fwd")

    x: torch.Tensor = torch.randn(2, 3, cfg["d_model"], dtype=torch.float32)
    y: torch.Tensor = layer(x)

    assert "act" in captured
    assert torch.allclose(captured["act"], y)
