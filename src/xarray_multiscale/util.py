from typing import Dict, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from xarray import DataArray


def adjust_shape(array: DataArray, scale_factors: Sequence[int]) -> DataArray:
    """
    Pad or crop array such that its new dimensions are evenly
    divisible by a set of integers.

    Parameters
    ----------
    array : ndarray
        Array that will be padded.

    scale_factors : Sequence of ints
        The output array is guaranteed to have dimensions that are each
        evenly divisible by the corresponding scale factor, and chunks
        that are smaller than or equal to the scale factor
        (if the array has chunks)

    Returns
    -------
    DataArray
    """
    result = array
    misalignment = np.any(np.mod(array.shape, scale_factors))
    if misalignment:
        new_shape = np.subtract(array.shape, np.mod(array.shape, scale_factors))
        result = array.isel({d: slice(s) for d, s in zip(array.dims, new_shape)})
    return result


def logn(x: npt.ArrayLike, n: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Compute the logarithm of x base n.

    Parameters
    ----------
    x : float or int.
    n: float or int.

    Returns
    -------
    float
        np.log(x) / np.log(n)

    """
    result: npt.NDArray[np.float64] = np.log(x) / np.log(n)
    return result


def broadcast_to_rank(
    value: Union[int, Sequence[int], Dict[int, int]], rank: int
) -> Tuple[int, ...]:
    result_dict = {}
    if isinstance(value, int):
        result_dict = {k: value for k in range(rank)}
    elif isinstance(value, Sequence):
        if not (len(value) == rank):
            raise ValueError(f"Length of value {len(value)} must match rank: {rank}")
        else:
            result_dict = {k: v for k, v in enumerate(value)}
    elif isinstance(value, dict):
        for dim in range(rank):
            result_dict[dim] = value.get(dim, 1)
    else:
        msg = (  # type: ignore[unreachable]
            "The first argument must be an integer, a sequence of integers",
            f"or a dictionary of integers. Got {type(value)}",
        )
        raise ValueError(msg)
    result = tuple(result_dict.values())
    typecheck = tuple(isinstance(val, int) for val in result)
    if not all(typecheck):
        bad_values = tuple(result[idx] for idx, val in enumerate(typecheck) if not val)
        msg = (
            "All elements of the first argument of this function must be integers. "
            f"Got non-integer values: {bad_values}"
        )
        raise ValueError(msg)

    return result
