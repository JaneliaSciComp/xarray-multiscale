from typing import Any, Dict, Protocol, Sequence, Tuple, cast

from numpy.typing import NDArray
from scipy.stats import mode


class WindowedReducer(Protocol):
    def __call__(
        self, array: NDArray[Any], window_size: Sequence[int], **kwargs: Any
    ) -> NDArray[Any]:
        ...


def windowed_mean(
    array: NDArray[Any], window_size: Tuple[int, ...], **kwargs: Any
) -> NDArray[Any]:
    """
    Compute the windowed mean of an array.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be downscaled. The array must have ``reshape`` and
        ``mean`` methods.

    window_size: Tuple of ints
        The window to use for aggregations. The array is partitioned into
        non-overlapping regions with size equal to ``window_size``, and the
        values in each window are aggregated to generate the result.

    **kwargs: dict, optional
        Extra keyword arguments passed to ``array.mean``

    Returns
    -------
    Array-like
        The result of the windowed mean. The length of each axis of this array
        will be a fraction of the input. The datatype is determined by the
        behavior of ``array.mean`` given the kwargs (if any) passed to it.

    Notes
    -----
    This function works by first reshaping the array, then computing the
    mean along extra axes

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale.reducers import windowed_mean
    >>> data = np.arange(16).reshape(4, 4)
    >>> windowed_mean(data, (2, 2))
    array([[ 2.5,  4.5],
           [10.5, 12.5]])
    """
    reshaped = reshape_windowed(array, window_size)
    result = reshaped.mean(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    return result


def windowed_mode(array: NDArray[Any], window_size: Tuple[int, ...]) -> NDArray[Any]:
    """
    Compute the windowed mode of an array. Input will be coerced to a numpy
    array.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be downscaled. The array must have a ``reshape``
        method.

    window_size: Tuple of ints
        The window to use for aggregation. The array is partitioned into
        non-overlapping regions with size equal to ``window_size``, and the
        values in each window are aggregated to generate the result.

    Returns
    -------
    Numpy array
        The result of the windowed mode. The length of each axis of this array
        will be a fraction of the input. The datatype is determined by the
        behavior of ``scipy.mean`` given the kwargs (if any) passed to it.

    Notes
    -----
    This function wraps ``scipy.stats.mode``.

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale.reducers import windowed_mode
    >>> data = np.arange(16).reshape(4, 4)
    >>> windowed_mode(data, (2, 2))
    array([[ 0,  2],
           [ 8, 10]])
    """
    reshaped = reshape_windowed(array, window_size)
    transposed_shape = tuple(range(0, reshaped.ndim, 2)) + tuple(
        range(1, reshaped.ndim, 2)
    )
    transposed = reshaped.transpose(transposed_shape)
    collapsed = transposed.reshape(tuple(reshaped.shape[slice(0, None, 2)]) + (-1,))
    result = mode(collapsed, axis=collapsed.ndim - 1).mode.squeeze(axis=-1)
    return result


def reshape_windowed(array: NDArray[Any], window_size: Tuple[int]) -> NDArray[Any]:
    """
    Reshape an array to support windowed operations. New
    dimensions will be added to the array, one for each element of
    `window_size`.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be reshaped. The array must have a ``reshape`` method.

    window_size: Tuple of ints
        The window size. The length of ``window_size`` must match the
        dimensionality of ``array``.

    Returns
    -------
    The input array reshaped with extra dimensions.
        E.g., for an ``array`` with shape ``(10, 2)``,
        ``reshape_windowed(array, (2, 2))`` returns
        output with shape ``(5, 2, 1, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale.reducers import reshape_windowed
    >>> data = np.arange(12).reshape(3, 4)
    >>> reshaped = reshape_windowed(data, (1, 2))
    >>> reshaped.shape
    (3, 1, 2, 2)
    """

    new_shape: Tuple[int, ...] = ()
    for s, f in zip(array.shape, window_size):
        new_shape += (s // f, f)
    return array.reshape(new_shape)
