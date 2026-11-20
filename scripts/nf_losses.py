"""Point losses aligned with the challenge SCORE (Pearson), for NeuralForecast."""

from __future__ import annotations

import torch
from neuralforecast import register_loss
from neuralforecast.losses.pytorch import BasePointLoss


@register_loss
class PearsonCorrLoss(BasePointLoss):
    """Minimize ``1 − Pearson(y, y_hat)`` over the masked batch.

    Competition SCORE is daily Pearson of prediction vs clipped ``y`` (scale-free).
    MSE pushes magnitude; this pushes co-movement, which matches the metric.

    With ``h=1`` the correlation is across the batch/windows dimension — keep
    ``windows_batch_size`` reasonably large (hundreds) so the estimate is stable.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(
            horizon_weight=None, outputsize_multiplier=1, output_names=[""]
        )
        self.eps = float(eps)

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y_insample  # unused; NF still passes it
        if mask is None:
            mask = torch.ones_like(y)
        weights = self._compute_weights(y=y, mask=mask)
        flat_w = weights.reshape(-1)
        y_f = y.reshape(-1)
        yh_f = y_hat.reshape(-1)
        keep = flat_w > 0
        y_f = y_f[keep]
        yh_f = yh_f[keep]
        n = int(y_f.numel())
        if n < 3:
            # Too small for a stable corr — fall back so grads still flow.
            return (y_f - yh_f).pow(2).mean() if n else y.new_zeros(())

        y_c = y_f - y_f.mean()
        yh_c = yh_f - yh_f.mean()
        denom = torch.sqrt((y_c * y_c).sum() * (yh_c * yh_c).sum()).clamp_min(self.eps)
        corr = (y_c * yh_c).sum() / denom
        # Clamp for numerical safety; loss in [0, 2] when corr in [-1, 1].
        corr = corr.clamp(-1.0, 1.0)
        return 1.0 - corr


def build_loss(name: str):
    """Factory: ``mse`` | ``pearson``."""
    from neuralforecast.losses.pytorch import MSE

    key = name.strip().lower()
    if key == "mse":
        return MSE()
    if key in ("pearson", "corr", "pearson_corr"):
        return PearsonCorrLoss()
    raise ValueError(f"unknown loss {name!r}; use mse|pearson")
