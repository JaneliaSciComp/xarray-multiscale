from typing import Any, Sequence, Tuple
from numpy.core.fromnumeric import reshape
from scipy.stats import mode


def windowed_mean(array: Any, window_size: Tuple[int, ...], **kwargs: Any):
    """
    Compute the windowed mean of an array.
    """
    reshaped = reshape_with_windows(array, window_size)
    result = reshaped.mean(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    return result


def windowed_mode(array: Any, window_size: Tuple[int, ...], **kwargs: Any) -> Any:
    """
    Coarsening by computing the n-dimensional mode.
    """
    reshaped = reshape_with_windows(array, window_size)
    transposed_shape = tuple(range(0, reshaped.ndim, 2)) + tuple(
        range(1, reshaped.ndim, 2)
    )
    transposed = reshaped.transpose(transposed_shape)
    collapsed = transposed.reshape(tuple(reshaped.shape[slice(0, None, 2)]) + (-1,))
    result = mode(collapsed, axis=collapsed.ndim - 1).mode.squeeze(axis=-1)
    return result


def reshape_with_windows(array, window_size: Sequence[int]):
    new_shape = []
    for s, f in zip(array.shape, window_size):
        new_shape.extend((s // f, f))
    return array.reshape(new_shape)
