import math
from functools import reduce
from itertools import combinations
from typing import Any, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mode


class WindowedReducer(Protocol):
    def __call__(
        self, array: NDArray[Any], window_size: Sequence[int], **kwargs: Any
    ) -> NDArray[Any]:
        ...


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
    if len(window_size) != array.ndim:
        raise ValueError(
            f"""Length of window_size must match array dimensionality.
                 Got {len(window_size)}, expected {array.ndim}"""
        )
    new_shape: Tuple[int, ...] = ()
    for s, f in zip(array.shape, window_size):
        new_shape += (s // f, f)
    return array.reshape(new_shape)


def windowed_mean(
    array: NDArray[Any], window_size: Tuple[int, ...], **kwargs: Any
) -> NDArray[Any]:
    """
    Compute the windowed mean of an array.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be downscaled. The array must have
        ``reshape`` and ``mean`` methods that obey the
        ``np.reshape`` and ``np.mean`` APIs.

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
    This function works by first reshaping the array to have an extra
    axis per element of ``window_size``, then computing the
    mean along those extra axes.

    See ``xarray_multiscale.reductions.reshape_windowed`` for the
    implementation of the array reshaping routine.

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


def windowed_max(
    array: NDArray[Any], window_size: Tuple[int, ...], **kwargs: Any
) -> NDArray[Any]:
    """
    Compute the windowed maximum of an array.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be downscaled. The array must have ``reshape`` and
        ``max`` methods.

    window_size: Tuple of ints
        The window to use for aggregations. The array is partitioned into
        non-overlapping regions with size equal to ``window_size``, and the
        values in each window are aggregated to generate the result.

    **kwargs: dict, optional
        Extra keyword arguments passed to ``array.mean``

    Returns
    -------
    Array-like
        The result of the windowed max. The length of each axis of this array
        will be a fraction of the input. The datatype of the return value will
        will be the same as the input.

    Notes
    -----
    This function works by first reshaping the array to have an extra
    axis per element of ``window_size``, then computing the
    max along those extra axes.

    See ``xarray_multiscale.reductions.reshape_windowed`` for
    the implementation of the array reshaping routine.

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale.reducers import windowed_mean
    >>> data = np.arange(16).reshape(4, 4)
    >>> windowed_max(data, (2, 2))
    array([[ 5,  7],
           [13, 15]])
    """
    reshaped = reshape_windowed(array, window_size)
    result = reshaped.max(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    return result


def windowed_min(
    array: NDArray[Any], window_size: Tuple[int, ...], **kwargs: Any
) -> NDArray[Any]:
    """
    Compute the windowed minimum of an array.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be downscaled. The array must have ``reshape`` and
        ``min`` methods.

    window_size: Tuple of ints
        The window to use for aggregations. The array is partitioned into
        non-overlapping regions with size equal to ``window_size``, and the
        values in each window are aggregated to generate the result.

    **kwargs: dict, optional
        Extra keyword arguments passed to ``array.mean``

    Returns
    -------
    Array-like
        The result of the windowed min. The length of each axis of this array
        will be a fraction of the input. The datatype of the return value will
        will be the same as the input.

    Notes
    -----
    This function works by first reshaping the array to have an extra
    axis per element of ``window_size``, then computing the
    min along those extra axes.

    See ``xarray_multiscale.reductions.reshape_windowed``
    for the implementation of the array reshaping routine.

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale.reducers import windowed_mean
    >>> data = np.arange(16).reshape(4, 4)
    >>> windowed_min(data, (2, 2))
    array([[0,  2],
           [8, 10]])
    """
    reshaped = reshape_windowed(array, window_size)
    result = reshaped.min(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    return result


def windowed_mode(array: NDArray[Any], window_size: Tuple[int, ...]) -> NDArray[Any]:
    """
    Compute the windowed mode of an array using either
    `windowed_mode_countess` or `windowed_mode_scipy`
    Input will be coerced to a numpy array.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be downscaled. The array must have a ``reshape``
        method.

    window_size: Tuple of ints
        The window to use for aggregation. The array is partitioned into
        non-overlapping regions with size equal to ``window_size``, and the
        values in each window are aggregated to generate the result.
        If the product of the elements of ``window_size`` is 16 or less, then
        ``windowed_mode_countless`` will be used. Otherwise,
        ``windowed_mode_scipy`` is used. This is a speculative cutoff based
        on the documentation of the countless algorithm used in
        ``windowed_mode_countless`` which was created by William Silversmith.

    Returns
    -------
    Numpy array
        The result of the windowed mode. The length of each axis of this array
        will be a fraction of the input.

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale.reducers import windowed_mode
    >>> data = np.arange(16).reshape(4, 4)
    >>> windowed_mode(data, (2, 2))
    array([[ 0,  2],
           [ 8, 10]])
    """

    if np.prod(window_size) <= 16:
        return windowed_mode_countless(array, window_size)
    else:
        return windowed_mode_scipy(array, window_size)


def windowed_mode_scipy(
    array: NDArray[Any], window_size: Tuple[int, ...]
) -> NDArray[Any]:
    """
    Compute the windowed mode of an array using scipy.stats.mode.
    Input will be coerced to a numpy array.

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
        will be a fraction of the input.

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
    result = mode(collapsed, axis=collapsed.ndim - 1, keepdims=False).mode
    return result


def windowed_mode_countless(
    array: NDArray[Any], window_size: Tuple[int, ...]
) -> NDArray[Any]:
    """
    countless downsamples labeled images (segmentations)
    by finding the mode using vectorized instructions.
    It is ill advised to use this O(2^N-1) time algorithm
    and O(NCN/2) space for N > about 16 tops.
    This means it's useful for the following kinds of downsampling.
    This could be implemented for higher performance in
    C/Cython more simply, but at least this is easily
    portable.
    2x2x1 (N=4), 2x2x2 (N=8), 4x4x1 (N=16), 3x2x1 (N=6)
    and various other configurations of a similar nature.
    c.f. https://medium.com/@willsilversmith/countless-3d-vectorized-2x-downsampling-of-labeled-volume-images-using-python-and-numpy-59d686c2f75

    This function has been modified from the original
    to avoid mutation of the input argument.

    Parameters
    ----------
    array: Numpy array
        The array to be downscaled.

    window_size: Tuple of ints
        The window size. The length of ``window_size`` must match the
        dimensionality of ``array``.

    """  # noqa
    sections = []

    mode_of = reduce(lambda x, y: x * y, window_size)
    majority = int(math.ceil(float(mode_of) / 2))

    for offset in np.ndindex(window_size):
        part = 1 + array[tuple(np.s_[o::f] for o, f in zip(offset, window_size))]
        sections.append(part)

    def pick(a, b):
        return a * (a == b)

    def lor(a, b):
        return a + (a == 0) * b

    subproblems = [{}, {}]
    results2 = None
    for x, y in combinations(range(len(sections) - 1), 2):
        res = pick(sections[x], sections[y])
        subproblems[0][(x, y)] = res
        if results2 is not None:
            results2 = lor(results2, res)
        else:
            results2 = res

    results = [results2]
    for r in range(3, majority + 1):
        r_results = None
        for combo in combinations(range(len(sections)), r):
            res = pick(subproblems[0][combo[:-1]], sections[combo[-1]])

            if combo[-1] != len(sections) - 1:
                subproblems[1][combo] = res

            if r_results is not None:
                r_results = lor(r_results, res)
            else:
                r_results = res
        results.append(r_results)
        subproblems[0] = subproblems[1]
        subproblems[1] = {}

    results.reverse()
    final_result = lor(reduce(lor, results), sections[-1]) - 1

    return final_result
