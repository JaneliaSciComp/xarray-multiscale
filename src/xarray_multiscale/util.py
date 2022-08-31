from xarray import DataArray
from typing import Sequence, Union, Tuple, Dict
import numpy as np


def adjust_shape(
    array: DataArray, scale_factors: Sequence[int], mode: Union[str, None]
) -> DataArray:
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

    mode : str
        If set to "crop", then the input array will be cropped as needed.
        Otherwise, this is the edge mode used by the padding routine.
        This parameter will be passed to `dask.array.pad` as the `mode`
        keyword.

    Returns
    -------
    dask array
    """
    result = array
    misalignment = np.any(np.mod(array.shape, scale_factors))
    if misalignment and (mode is not None):
        if mode == "crop":
            new_shape = np.subtract(array.shape, np.mod(array.shape, scale_factors))
            result = array.isel({d: slice(s) for d, s in zip(array.dims, new_shape)})
        else:
            new_shape = np.add(
                array.shape,
                np.subtract(scale_factors, np.mod(array.shape, scale_factors)),
            )
            pw = {
                dim: (0, int(new - old))
                for dim, new, old in zip(array.dims, new_shape, array.shape)
                if old != new
            }
            result = array.pad(pad_width=pw, mode=mode)
    return result


def logn(x: float, n: float) -> float:
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
    result: float = np.log(x) / np.log(n)
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
        raise ValueError(
            f"The first argument must be an int, a sequence of ints, or a dict of ints. Got {type(value)}"
        )
    result = tuple(result_dict.values())
    typecheck = tuple(isinstance(val, int) for val in result)
    if not all(typecheck):
        bad_values = tuple(result[idx] for idx, val in enumerate(typecheck) if not val)
        raise ValueError(
            f"All elements of the first argument of this function must be ints. Non-integer values: {bad_values}"
        )
    return result
