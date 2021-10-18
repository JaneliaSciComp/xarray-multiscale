from typing import Any, Tuple
from scipy.stats import mode as scipy_mode


def windowed_mean(array: Any, window_size: Tuple[int, ...], **kwargs: Any):
    """
    Compute the windowed mean of an array.
    """
    new_shape = []
    for s, f in zip(array.shape, window_size):
        new_shape.extend((s // f, f))
    reshaped = array.reshape(new_shape)
    result = reshaped.mean(axis=tuple(range(1, len(new_shape), 2)), **kwargs)
    return result


def windowed_mode(a: Any, window_size: Tuple[int, ...], **kwargs: Any) -> Any:
    """
    Coarsening by computing the n-dimensional mode.
    """
    new_shape = []
    for s, f in zip(array.shape, window_size):
        new_shape.extend((s // f, f))
    transposed_shape = new_shape[slice(0, None, 2)] +  new_shape[slice(1,None,2)]
    reshaped = array.reshape(new_shape).transpose(transposed_shape).reshape(new_shape[slice(0, None, 2)] + (-1,))
    modes = scipy_mode(reshaped, axis=-1).mode
    result = modes.squeeze(axis=-1)
    return result
