"""Shared MLP helpers for the hybrid model's learned components (tire, direction, Sigma)."""

import torch.nn as nn


def _mlp(in_dim: int, hidden: int, layers: int, out_dim: int) -> nn.Sequential:
    """MLP with ``layers`` tanh-activated hidden layers of ``hidden`` neurons."""
    modules: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        modules += [nn.Linear(dim, hidden), nn.Tanh()]
        dim = hidden
    modules.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*modules)


def _mlp_shape(net: nn.Sequential) -> tuple[int, int]:
    """``(hidden width, hidden layer count)`` of an :func:`_mlp`, for checkpointing."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    return linears[0].out_features, len(linears) - 1
