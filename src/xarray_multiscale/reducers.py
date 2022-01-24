from typing import Any, Sequence, Tuple, cast, TypeVar, Dict
from scipy.stats import mode
from numpy.typing import NDArray


def windowed_mean(
    array: NDArray[Any], window_size: Tuple[int, ...], **kwargs: Dict[Any, Any]
) -> NDArray[Any]:
    """
    Compute the windowed mean of an array.
    """
    reshaped = reshape_with_windows(array, window_size)
    result = reshaped.mean(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    cast(NDArray[Any], result)
    return result


def windowed_mode(array: NDArray[Any], window_size: Tuple[int, ...]) -> NDArray[Any]:
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


def reshape_with_windows(
    array: NDArray[Any], window_size: Sequence[int]
) -> NDArray[Any]:
    new_shape = ()
    for s, f in zip(array.shape, window_size):
        new_shape += (s // f, f)
    return array.reshape(new_shape)
