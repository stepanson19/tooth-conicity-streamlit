from __future__ import annotations

import numpy as np
import torch

from tooth_service.trained_inference import predict_prepared_tooth_mask


class _StubModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), -10.0, dtype=torch.float32, device=x.device)
        logits[:, :, 8:24, 10:22] = 10.0
        logits[:, :, 1:3, 1:3] = 10.0
        return logits


def test_predict_prepared_tooth_mask_resizes_to_original_shape_and_keeps_largest_component():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :] = np.array([10, 20, 30], dtype=np.uint8)

    mask = predict_prepared_tooth_mask(
        image,
        model=_StubModel(),
        device="cpu",
        image_size=32,
        threshold=0.5,
        largest_component_only=True,
    )

    assert mask.shape == (64, 96)
    assert mask.dtype == bool
    assert mask.any()
    assert mask[4, 4] == 0
    assert mask[32, 48] == 1
