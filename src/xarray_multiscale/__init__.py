__version__ = "0.1.0"
from .multiscale import downscale, multiscale  # noqa: F401
from .reducers import (windowed_mean,  # noqa: F401
                       windowed_mode,  # noqa: F401
                       windowed_max,  # noqa: F401
                       windowed_min)  # noqa: F401
