from collections.abc import Sequence
from typing import Optional

import torch as th
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class SinLU(nn.Module):
    """Trainable sinusoidal linear unit used by the released NAGS model."""

    def __init__(self, init_a: float = 1.0, init_b: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(th.tensor(init_a))
        self.b = nn.Parameter(th.tensor(init_b))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x + self.a * th.sin(self.b * x)) * self.sigmoid(x)


_ACT_FACTORY = {
    "empty": lambda **_: nn.Identity(),
    "silu": lambda **_: nn.SiLU(),
    "relu": lambda **_: nn.ReLU(),
    "elu": lambda **_: nn.ELU(),
    "leakyrelu": lambda **_: nn.LeakyReLU(),
    "gelu": lambda **_: nn.GELU(),
    "sigmoid": lambda **_: nn.Sigmoid(),
    "tanh": lambda **_: nn.Tanh(),
    "softsign": lambda **_: nn.Softsign(),
    "sinlu": lambda **_: SinLU(),
}


def _is_supported_activation(act_str: str) -> bool:
    return act_str in _ACT_FACTORY


class _Activation(nn.Module):
    def __init__(self, act_str: str, _dim: Optional[int] = None):
        super().__init__()
        # Keep the historical attribute name so released checkpoints retain
        # their ``...ac.act.*`` state-dict keys.
        self.act = _ACT_FACTORY[act_str]()

    def forward(self, x: th.Tensor, *_args, **_kwargs) -> th.Tensor:
        return self.act(x)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


class NAGSBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 heads_dim: int,
                 act_layer: str,
                 stage_index: int):
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.heads_dim = _positive_int(heads_dim, "heads_dim")
        if self.output_dim % self.heads_dim != 0:
            raise ValueError(
                "output_dim must be divisible by heads_dim so that the attention "
                f"and residual branches have the same width; got "
                f"output_dim={self.output_dim}, heads_dim={self.heads_dim}."
            )
        if not isinstance(stage_index, int) or stage_index < 0:
            raise ValueError(f"stage_index must be a non-negative integer, got {stage_index!r}.")
        if not isinstance(act_layer, str) or not _is_supported_activation(act_layer):
            raise ValueError(f"Unsupported activation: {act_layer!r}.")

        self.stage_index = stage_index
        num_heads = self.output_dim // self.heads_dim
        self.proj_up = nn.Linear(self.input_dim, self.output_dim)
        self.attn = GATv2Conv(
            self.input_dim,
            self.heads_dim,
            heads=num_heads,
            add_self_loops=False,
        )
        self.ln = nn.LayerNorm(self.output_dim)
        self.ac = _Activation(act_layer, self.output_dim)

    def forward(self, graph: Data) -> th.Tensor:
        x = graph.x[:, :self.input_dim]
        edge_index = graph.edge_index_high if self.stage_index % 2 else graph.edge_index_low
        x = self.proj_up(x) + self.attn(x=x, edge_index=edge_index)
        return self.ac(self.ln(x), edge_index, None)


class NAGSModel(nn.Module):
    def __init__(self, mdl_cfg: DictConfig):
        super().__init__()
        if mdl_cfg is None:
            raise ValueError("A matching backbone configuration is required.")

        self.input_dim = _positive_int(mdl_cfg.get("input_dim"), "backbone.input_dim")
        self.heads_dim = _positive_int(mdl_cfg.get("heads_dim"), "backbone.heads_dim")
        self.act_layer = mdl_cfg.get("act_layer", "sinlu")
        if not isinstance(self.act_layer, str) or not _is_supported_activation(self.act_layer):
            raise ValueError(f"Unsupported backbone.act_layer: {self.act_layer!r}.")

        stage_dim = mdl_cfg.get("stage_dim")
        if isinstance(stage_dim, (str, bytes)) or not isinstance(stage_dim, Sequence) or not stage_dim:
            raise ValueError("backbone.stage_dim must be a non-empty sequence of positive integers.")
        self.stage_dim = [
            _positive_int(width, f"backbone.stage_dim[{index}]")
            for index, width in enumerate(stage_dim)
        ]
        for index, width in enumerate(self.stage_dim):
            if width % self.heads_dim != 0:
                raise ValueError(
                    f"backbone.stage_dim[{index}]={width} must be divisible by "
                    f"backbone.heads_dim={self.heads_dim}."
                )

        input_dim = self.input_dim
        self.blks = nn.ModuleList()
        for stage_index, output_dim in enumerate(self.stage_dim):
            self.blks.append(
                NAGSBlock(input_dim, output_dim, self.heads_dim, self.act_layer, stage_index)
            )
            input_dim = output_dim

    def _validate_graph(self, data: Data):
        if not isinstance(data, Data):
            raise TypeError(f"NAGSModel expects torch_geometric.data.Data, got {type(data)!r}.")
        x = getattr(data, "x", None)
        if not isinstance(x, th.Tensor) or x.ndim != 2:
            raise ValueError("Graph data.x must be a two-dimensional tensor.")
        if x.shape[1] < self.input_dim:
            raise ValueError(
                f"Graph features need at least {self.input_dim} columns, got {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError(f"Graph features must be floating point, got {x.dtype}.")
        if not th.isfinite(x).all():
            raise ValueError("Graph features must not contain NaN or infinite values.")

        num_nodes = x.shape[0]
        for edge_name in ("edge_index_high", "edge_index_low"):
            if not hasattr(data, edge_name):
                raise ValueError(f"Graph is missing required {edge_name}.")
            edge_index = getattr(data, edge_name)
            if not isinstance(edge_index, th.Tensor):
                raise TypeError(f"{edge_name} must be a torch.Tensor, got {type(edge_index)!r}.")
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(
                    f"{edge_name} must have shape [2, E], got {tuple(edge_index.shape)}."
                )
            if edge_index.dtype != th.long:
                raise TypeError(f"{edge_name} must have dtype torch.long, got {edge_index.dtype}.")
            if edge_index.device != x.device:
                raise ValueError(f"{edge_name} and data.x must be on the same device.")
            if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
                raise ValueError(f"{edge_name} contains node indices outside [0, {num_nodes}).")

    def forward(self, data: Data) -> Data:
        """Return embeddings in a cloned graph without modifying ``data`` in place."""
        self._validate_graph(data)
        output = data.clone()
        output.x = output.x[:, :self.input_dim]
        for block in self.blks:
            output.x = block(output)
        return output
