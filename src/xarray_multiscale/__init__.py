from __future__ import annotations

from .multiscale import downscale, multiscale
from .reducers import (
    windowed_max,
    windowed_mean,
    windowed_min,
    windowed_rank,
)

__all__ = [
    "downscale",
    "multiscale",
    "windowed_mean",
    "windowed_mode",
    "windowed_max",
    "windowed_min",
    "windowed_rank",
]
