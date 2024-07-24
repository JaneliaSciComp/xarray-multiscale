from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any, Sequence

    import numpy.typing as npt

import math
from functools import reduce
from itertools import combinations

import numpy as np
from scipy.stats import mode


class WindowedReducer(Protocol):
    def __call__(
        self, array: npt.NDArray[Any], window_size: Sequence[int], **kwargs: Any
    ) -> npt.NDArray[Any]: ...


def reshape_windowed(array: npt.NDArray[Any], window_size: tuple[int, ...]) -> npt.NDArray[Any]:
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
    new_shape: tuple[int, ...] = ()
    for s, f in zip(array.shape, window_size):
        new_shape += (s // f, f)
    return array.reshape(new_shape)


def windowed_mean(
    array: npt.NDArray[Any], window_size: tuple[int, ...], **kwargs: Any
) -> npt.NDArray[Any]:
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
    result: npt.NDArray[Any] = reshaped.mean(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    return result


def windowed_max(
    array: npt.NDArray[Any], window_size: tuple[int, ...], **kwargs: Any
) -> npt.NDArray[Any]:
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
    result: npt.NDArray[Any] = reshaped.max(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    return result


def windowed_min(
    array: npt.NDArray[Any], window_size: tuple[int, ...], **kwargs: Any
) -> npt.NDArray[Any]:
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
    result: npt.NDArray[Any] = reshaped.min(axis=tuple(range(1, reshaped.ndim, 2)), **kwargs)
    return result


def windowed_mode(array: npt.NDArray[Any], window_size: tuple[int, ...]) -> npt.NDArray[Any]:
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


def windowed_mode_scipy(array: npt.NDArray[Any], window_size: tuple[int, ...]) -> npt.NDArray[Any]:
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
    transposed_shape = tuple(range(0, reshaped.ndim, 2)) + tuple(range(1, reshaped.ndim, 2))
    transposed = reshaped.transpose(transposed_shape)
    collapsed = transposed.reshape(tuple(reshaped.shape[slice(0, None, 2)]) + (-1,))
    result: npt.NDArray[Any] = mode(collapsed, axis=collapsed.ndim - 1, keepdims=False).mode
    return result


def _pick(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> Any:
    return a * (a == b)


def _lor(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> Any:
    return a + (a == 0) * b


def windowed_mode_countless(
    array: npt.NDArray[Any], window_size: tuple[int, ...]
) -> npt.NDArray[Any]:
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

    """
    sections = []

    mode_of = reduce(lambda x, y: x * y, window_size)
    majority = int(math.ceil(float(mode_of) / 2))

    for offset in np.ndindex(window_size):
        part = 1 + array[tuple(np.s_[o::f] for o, f in zip(offset, window_size))]
        sections.append(part)

    subproblems: list[dict[tuple[int, int], npt.ArrayLike]] = [{}, {}]
    results2 = None
    for x, y in combinations(range(len(sections) - 1), 2):
        res = _pick(sections[x], sections[y])
        subproblems[0][(x, y)] = res
        if results2 is not None:
            results2 = _lor(results2, res)  # type: ignore[unreachable]
        else:
            results2 = res

    results = [results2]
    for r in range(3, majority + 1):
        r_results = None
        for combo in combinations(range(len(sections)), r):
            res = _pick(subproblems[0][combo[:-1]], sections[combo[-1]])  # type: ignore[index, arg-type]

            if combo[-1] != len(sections) - 1:
                subproblems[1][combo] = res  # type: ignore[index]

            if r_results is not None:
                r_results = _lor(r_results, res)  # type: ignore[unreachable]
            else:
                r_results = res
        results.append(r_results)
        subproblems[0] = subproblems[1]
        subproblems[1] = {}

    results.reverse()
    final_result: npt.NDArray[Any] = _lor(reduce(_lor, results), sections[-1]) - 1  # type: ignore[arg-type]

    return final_result


def windowed_rank(
    array: npt.NDArray[Any], window_size: tuple[int, ...], rank: int = -1
) -> npt.NDArray[Any]:
    """
    Compute the windowed rank order filter of an array.
    Input will be coerced to a numpy array.

    Parameters
    ----------
    array: Array-like, e.g. Numpy array, Dask array
        The array to be downscaled. The array must have a ``reshape``
        method.

    window_size: tuple[int, ...]
        The window to use for aggregation. The array is partitioned into
        non-overlapping regions with size equal to ``window_size``, and the
        values in each window are sorted to generate the result.

    rank: int, default=-1
        The index to take from the sorted values in each window. If non-negative, then
        rank must be between 0 and the product of the elements of ``window_size`` minus one,
        (inclusive).
        Rank may be negative, in which case it denotes an index relative to the end of the sorted
        values following normal python indexing rules.
        E.g., when rank is -1 (the default), this takes the maxmum value of each window.

    Returns
    -------
    Numpy array
        The result of the windowed rank filter. The length of each axis of this array
        will be a fraction of the input.

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale.reducers import windowed_rank
    >>> data = np.arange(16).reshape(4, 4)
    >>> windowed_rank(data, (2, 2), -2)
    array([[ 4  6]
           [12 14]])
    """
    max_rank = np.prod(window_size) - 1
    if rank > max_rank or rank < -max_rank - 1:
        msg = (
            f"Invalid rank: {rank} for window_size: {window_size} ",
            f"If rank is negative then between either -1 and {-max_rank-1}, inclusive",
            f"If rank is non-negtaive, then it must be between 0 and {max_rank}, inclusive.",
        )
        raise ValueError(msg)
    reshaped = reshape_windowed(array, window_size)
    transposed_shape = tuple(range(0, reshaped.ndim, 2)) + tuple(range(1, reshaped.ndim, 2))
    transposed = reshaped.transpose(transposed_shape)
    collapsed = transposed.reshape(tuple(reshaped.shape[slice(0, None, 2)]) + (-1,))
    result: npt.NDArray[Any] = np.take(np.sort(collapsed, axis=-1), rank, axis=-1)
    return result
