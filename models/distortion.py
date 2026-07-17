import math

from omegaconf import DictConfig

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CrossMultiLayer(nn.Module):
    def __init__(self,
                 stage_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.stage_dim = stage_dim
        self.hidden_dim = hidden_dim

        self.ly_x = nn.Linear(in_features=stage_dim, out_features=hidden_dim)
        self.ly_y = nn.Linear(in_features=stage_dim, out_features=hidden_dim)

        self.ly_xy = nn.Linear(in_features=hidden_dim, out_features=stage_dim)

    def forward(self,
                x: th.Tensor,
                y: th.Tensor,
                ) -> th.Tensor:
        if x.shape != y.shape:
            raise ValueError(
                "CrossMultiLayer inputs must have the same shape, got "
                f"{tuple(x.shape)} and {tuple(y.shape)}."
            )
        x = self.ly_x(x)
        y = self.ly_y(y)

        xy = self.ly_xy(x * y)
        return xy


class CrossMultiBlock(nn.Module):
    def __init__(self,
                 stage_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.stage_dim = stage_dim
        self.hidden_dim = hidden_dim

        self.ly1 = CrossMultiLayer(stage_dim, hidden_dim)
        self.ly2 = CrossMultiLayer(stage_dim, hidden_dim)
        self.ly3 = CrossMultiLayer(stage_dim, hidden_dim)

    def forward(self,
                x: th.Tensor,
                y: th.Tensor,
                ) -> th.Tensor:
        x = x + \
            self.ly1(x, x) + \
            self.ly2(x, y) + \
            self.ly3(y, y)
        return x


class HCMModel(nn.Module):
    def __init__(self,
                 image_size: int,
                 mdl_cfg: DictConfig):
        super().__init__()
        self.image_size = image_size

        self.stage_dim = mdl_cfg.backbone.stage_dim
        self.hidden_dim = mdl_cfg.backbone.hidden_dim
        self.blk_num = mdl_cfg.backbone.blk_num
        self.normalization_scale = float(mdl_cfg.normalization_scale)
        if not math.isfinite(self.normalization_scale) or self.normalization_scale <= 0:
            raise ValueError(
                "model.correct.normalization_scale must be a finite positive "
                f"number, got {self.normalization_scale}."
            )

        self.proj_up1 = nn.Linear(1, self.stage_dim)
        self.proj_up2 = nn.Linear(1, self.stage_dim)
        
        self.blk1 = nn.ModuleList([CrossMultiBlock(self.stage_dim,
                                                   self.hidden_dim)
                                   for i in range(self.blk_num)])
        self.blk2 = nn.ModuleList([CrossMultiBlock(self.stage_dim,
                                                   self.hidden_dim)
                                   for i in range(self.blk_num)])

        self.proj_out = nn.Linear(self.stage_dim * 2, 2)

    def forward(self,
                x: th.Tensor
                ) -> th.Tensor:
        image_center = self.image_size // 2
        if image_center <= 0:
            raise ValueError(f"image_size must be at least 2, got {self.image_size}.")
        coordinate_scale = image_center / self.normalization_scale
        x = (x - image_center) / coordinate_scale

        x1 = self.proj_up1(x[:, 0].unsqueeze(-1))
        x2 = self.proj_up2(x[:, 1].unsqueeze(-1))

        for blk1, blk2 in zip(self.blk1, self.blk2):
            x1_r = blk1(x1, x2)
            x2_r = blk2(x2, x1)

            x1 = x1_r
            x2 = x2_r

        x_r = th.cat([x1_r, x2_r], dim=-1)
        x_r = self.proj_out(x_r)
        x_r = x_r * coordinate_scale + image_center
        return x_r
