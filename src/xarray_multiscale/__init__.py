__version__ = "0.1.1"
from .multiscale import downscale, multiscale  # noqa: F401
from .reducers import (  # noqa: F401
    windowed_mean,
    windowed_mode,
    windowed_max,
    windowed_min,
)
