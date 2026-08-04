__version__ = "0.2.2"
from .multiscale import downscale, multiscale
from .pyramid import Multiscale, open_multiscale

__all__ = ["downscale", "multiscale", "Multiscale", "open_multiscale"]
