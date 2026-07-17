import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _cosine_similarity(x1: th.Tensor, x2: th.Tensor) -> th.Tensor:
    x1_norm = F.normalize(x1, dim=-1)
    x2_norm = F.normalize(x2, dim=-1)
    return x1_norm @ x2_norm.T


class InfoNceLoss(nn.Module):
    def __init__(self,
                 temperature: float,
                 trainable_temperature: bool = False,
                 temperature_eps: float = 1e-6):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                f"temperature must be finite and strictly positive, got {temperature}."
            )
        if not math.isfinite(temperature_eps) or temperature_eps <= 0:
            raise ValueError(
                "temperature_eps must be finite and strictly positive, "
                f"got {temperature_eps}."
            )

        self.trainable_temperature = trainable_temperature
        self.temperature_eps = temperature_eps
        if trainable_temperature:
            if temperature <= temperature_eps:
                raise ValueError(
                    "A trainable temperature must exceed temperature_eps; "
                    f"got temperature={temperature}, "
                    f"temperature_eps={temperature_eps}."
                )
            initial_value = temperature - temperature_eps
            raw_temperature = initial_value + math.log(
                -math.expm1(-initial_value)
            )
            self.t = nn.Parameter(
                th.tensor(raw_temperature, dtype=th.float32)
            )
        else:
            self.register_buffer(
                "t",
                th.tensor(temperature, dtype=th.float32),
            )

    def _temperature(self) -> th.Tensor:
        if self.trainable_temperature:
            temperature = F.softplus(self.t) + self.temperature_eps
        else:
            temperature = self.t
        if not th.isfinite(temperature).all() or th.any(temperature <= 0):
            raise RuntimeError("InfoNCE temperature must remain finite and positive.")
        return temperature

    def forward(self,
                img_data: th.Tensor,     # shape: [N, D]
                ast_data: th.Tensor,     # shape: [N, D]
                match_pairs: th.Tensor,   # shape: [N, 2], dtype=torch.long
                ) -> th.Tensor:
        if not isinstance(img_data, th.Tensor) or not isinstance(ast_data, th.Tensor):
            raise TypeError("img_data and ast_data must be torch.Tensor instances.")
        if img_data.ndim != 2 or ast_data.ndim != 2:
            raise ValueError(
                "img_data and ast_data must both have shape [N, D], got "
                f"{tuple(img_data.shape)} and {tuple(ast_data.shape)}."
            )
        if img_data.shape[1] != ast_data.shape[1]:
            raise ValueError(
                "img_data and ast_data must have the same feature dimension, got "
                f"{img_data.shape[1]} and {ast_data.shape[1]}."
            )
        if not img_data.is_floating_point() or not ast_data.is_floating_point():
            raise TypeError("img_data and ast_data must use floating-point dtypes.")
        if img_data.dtype != ast_data.dtype:
            raise TypeError(
                "img_data and ast_data must have the same dtype, got "
                f"{img_data.dtype} and {ast_data.dtype}."
            )
        if img_data.device != ast_data.device:
            raise ValueError(
                "img_data and ast_data must be on the same device, got "
                f"{img_data.device} and {ast_data.device}."
            )
        if not isinstance(match_pairs, th.Tensor):
            raise TypeError("match_pairs must be a torch.Tensor.")
        if match_pairs.ndim != 2 or match_pairs.shape[1] != 2:
            raise ValueError(
                f"match_pairs must have shape [N, 2], got {tuple(match_pairs.shape)}."
            )
        if match_pairs.dtype != th.long:
            raise TypeError(f"match_pairs must have dtype torch.long, got {match_pairs.dtype}.")
        if match_pairs.shape[0] < 2:
            raise ValueError(
                "InfoNCE requires at least two matched pairs; a single pair "
                "produces a 1x1 logit matrix and has no learning signal."
            )
        if match_pairs.device != img_data.device:
            match_pairs = match_pairs.to(img_data.device, non_blocking=True)
        img_indices = match_pairs[:, 0]
        ast_indices = match_pairs[:, 1]
        if th.any(img_indices < 0) or th.any(ast_indices < 0):
            raise IndexError("match_pairs must not contain negative indices.")
        if th.any(img_indices >= img_data.shape[0]):
            raise IndexError(
                "match_pairs contains an image index outside img_data."
            )
        if th.any(ast_indices >= ast_data.shape[0]):
            raise IndexError(
                "match_pairs contains an astrometry index outside ast_data."
            )
        logits_sim = _cosine_similarity(
            img_data[img_indices],
            ast_data[ast_indices],
        )
        temperature = self._temperature().to(dtype=logits_sim.dtype)
        logits = logits_sim / temperature
        labels = th.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss
        
