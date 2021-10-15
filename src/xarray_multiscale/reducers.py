from typing import Any, Optional, Tuple
from scipy.stats import mode as scipy_mode
import numpy as np


def windowed_mean(array: Any, window_size: Tuple[int, ...], **kwargs: Any):
    new_shape = []
    for s, f in zip(array.shape, window_size):
        new_shape.extend((s // f, f))
    reshaped = array.reshape(new_shape)
    result = reshaped.mean(axis=tuple(range(1, len(new_shape), 2)), **kwargs)
    return result


def mode(a: Any, axis: Optional[int] = None) -> Any:
    """
    Coarsening by computing the n-dimensional mode, compatible with da.coarsen. If input is all 0s, the mode is not computed.
    """
    if axis is None:
        return a
    elif a.max() == 0:
        return np.min(a, axis)
    else:
        transposed = a.transpose(*range(0, a.ndim, 2), *range(1, a.ndim, 2))
        reshaped = transposed.reshape(*transposed.shape[: a.ndim // 2], -1)
        modes = scipy_mode(reshaped, axis=reshaped.ndim - 1).mode
        result = modes.squeeze(axis=-1)
        return result
