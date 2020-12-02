import numpy as np
import dask.array as da
from xarray import DataArray
from typing import Any, List, Optional, Tuple, Union, Sequence, Callable
from scipy.interpolate import interp1d
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.array import coarsen


def multiscale(
    array: Any,
    reduction: Callable[[Any], Any],
    scale_factors: Union[Sequence[int], int],
    pad_mode: Optional[str] = None,
    preserve_dtype: bool = True,
) -> List[DataArray]:
    """
    Lazily generate a multiscale representation of an array

    Parameters
    ----------
    array: ndarray to be downscaled.

    reduction: a function that aggregates data over windows.

    scale_factors: an iterable of integers that specifies how much to downscale each axis of the array.

    pad_mode: How (or if) the input should be padded. When set to `None` the input will be trimmed as needed.

    preserve_dtype: boolean, defaults to True, determines whether lower levels of the pyramid are coerced to the same dtype as the input. This assumes that
    the reduction function accepts a "dtype" kwarg, e.g. numpy.mean(x, dtype='int').

    Returns a list of DataArrays, one per level of downscaling. These DataArrays have `coords` properties that track the changing offset (if any)
    induced by the downsampling operation. Additionally, the scale factors are stored each DataArray's attrs propery under the key `scale_factors`
    -------

    """
    if isinstance(scale_factors, int):
        scale_factors = (scale_factors,) * array.ndim
    else:
        assert len(scale_factors) == array.ndim

    if pad_mode is None:
        # with pad_mode set to "none", dask will trim the data such that it can be tiled
        # by the scale factors
        padded_shape = np.subtract(array.shape, np.mod(array.shape, scale_factors))
    else:
        padded_shape = prepad(array, scale_factors, pad_mode=pad_mode).shape

    # figure out the maximum depth
    levels = range(0, get_downscale_depth(padded_shape, scale_factors))
    scales: Tuple[Tuple[int]] = tuple(
        tuple(s ** l for s in scale_factors) for l in levels
    )
    result = [_ingest_array(array, scales[0])]
    data = result[0].data
    base_attrs = result[0].attrs
    base_coords = result[0].coords

    for scale in scales[1:]:
        downscaled = downscale(
            data, reduction, scale, pad_mode=pad_mode, preserve_dtype=preserve_dtype
        )

        # hideous
        new_coords = tuple(
            DataArray(
                (offset * (base_coords[bc][1] - base_coords[bc][0]))
                + (base_coords[bc][:s] * sc),
                name=base_coords[bc].name,
                attrs=base_coords[bc].attrs,
            )
            for s, bc, offset, sc in zip(
                downscaled.shape, base_coords, get_downsampled_offset(scale), scale
            )
        )

        result.append(
            DataArray(
                data=downscaled,
                coords=new_coords,
                attrs=base_attrs,
                name=result[0].name
            )
        )
    return result


def _ingest_array(array: Any, scales: Sequence[int]):
    if hasattr(array, "coords"):
        # if the input is a xarray.DataArray, assign a new variable to the DataArray and use the variable
        # `array` to refer to the data property of that array
        data = da.asarray(array.data)
        dims = array.dims
        # ensure that key order matches dimension order
        coords = {d: array.coords[d] for d in dims}
        attrs = array.attrs
        name = array.name
    else:
        data = da.asarray(array)
        dims = tuple(f'dim_{d}' for d in range(data.ndim))
        coords = {
            dim: DataArray(offset + np.arange(s, dtype="float32"), dims=dim)
            for dim, s, offset in zip(dims, array.shape, get_downsampled_offset(scales))
        }
        name = None
        attrs = {}

    result = DataArray(
        data=data,
        coords=coords,
        dims=dims,
        attrs=attrs,
        name=name
    )
    return result


def even_padding(length: int, window: int) -> int:
    """
    Compute how much to add to `length` such that the resulting value is evenly divisible by `window`.

    Parameters
    ----------
    length : int
    window: int
    """
    return (window - (length % window)) % window


def logn(x: float, n: float) -> float:
    """
    Compute the logarithm of x base n.

    Parameters
    ----------
    x : float or int.
    n: float or int.

    Returns np.log(x) / np.log(n)
    -------

    """
    result: float = np.log(x) / np.log(n)
    return result


def prepad(
    array: Any,
    scale_factors: Sequence[int],
    pad_mode: Optional[str] = "reflect",
    rechunk: bool = True,
) -> da.array:
    """
    Pad an array such that its new dimensions are evenly divisible by some integer.

    Parameters
    ----------
    array: An ndarray that will be padded.

    scale_factors: An iterable of integers. The output array is guaranteed to have dimensions that are each evenly divisible
    by the corresponding scale factor, and chunks that are smaller than or equal to the scale factor (if the array has chunks)

    mode: String. The edge mode used by the padding routine. See `dask.array.pad` for more documentation.

    Returns a dask array with padded dimensions.
    -------

    """

    if pad_mode == None:
        # no op
        return array

    pw = tuple(
        (0, even_padding(ax, scale)) for ax, scale in zip(array.shape, scale_factors)
    )

    result = da.pad(array, pw, mode=pad_mode)

    # rechunk so that small extra chunks added by padding are fused into larger chunks, but only if we had to add chunks after padding
    if rechunk and np.any(pw):
        new_chunks = tuple(
            np.multiply(
                scale_factors, np.ceil(np.divide(result.chunksize, scale_factors))
            ).astype("int")
        )
        result = result.rechunk(new_chunks)

    if hasattr(array, "coords"):
        new_coords = {}
        for p, k in zip(pw, array.coords):
            old_coord = array.coords[k]
            if np.diff(p) == 0:
                new_coords[k] = old_coord
            else:
                extended_coords = interp1d(
                    np.arange(len(old_coord.values)),
                    old_coord.values,
                    fill_value="extrapolate",
                )(np.arange(len(old_coord.values) + p[-1])).astype(old_coord.dtype)
                new_coords[k] = DataArray(
                    extended_coords, dims=k, attrs=old_coord.attrs
                )
        result = DataArray(
            result, coords=new_coords, dims=array.dims, attrs=array.attrs
        )
    return result


def downscale(
    array: Union[np.array, da.array],
    reduction: Callable,
    scale_factors: Sequence[int],
    pad_mode: Optional[str] = None,
    preserve_dtype: bool = True,
    **kwargs,
) -> DataArray:
    """
    Downscale an array using windowed aggregation. This function is a light wrapper for `dask.array.coarsen`.

    Parameters
    ----------
    array: The narray to be downscaled.

    reduction: The function to apply to each window of the array.

    scale_factors: A list if ints specifying how much to downscale the array per dimension.

    trim_excess: A boolean that determines whether the size of the input array should be increased or decreased such that
    each scale factor tiles its respective array axis. Defaults to False, which will result in the input being padded.

    **kwargs: extra kwargs passed to dask.array.coarsen

    Returns the downscaled version of the input as a dask array.
    -------
    """
    trim_excess = False
    if pad_mode == None:
        trim_excess = True

    to_coarsen = prepad(da.asarray(array), scale_factors, pad_mode=pad_mode)

    coarsened = coarsen(
        reduction,
        to_coarsen,
        {d: s for d, s in enumerate(scale_factors)},
        trim_excess=trim_excess,
        **kwargs,
    )

    if preserve_dtype:
        coarsened = coarsened.astype(array.dtype)

    return coarsened


def get_downscale_depth(shape: Tuple[int], scale_factors: Sequence[int]) -> int:
    """
    For an array and a sequence of scale factors, calculate the maximum possible number of downscaling operations.
    If `scale_factors` is uniformly less than 1, this function returns 0.
    """
    if len(shape) != len(scale_factors):
        raise ValueError(
            f"Shape (length == {len(shape)} ) and scale factors (length == {len(scale_factors)}) do not align."
        )

    _scale_factors: Any = np.array(scale_factors).astype("int")
    _shape: Any = np.array(shape).astype("int")

    # If any of the scale factors are greater than the respective shape, return 0
    if np.any((_scale_factors - shape) > 0):
        result = 0

    # If all scale factors are 1, return 0
    if np.all(_scale_factors == 1):
        result = 0
    else:
        depths = {}
        for ax, s in enumerate(_scale_factors):
            if s > 1:
                depths[ax] = np.ceil(logn(shape[ax], s)).astype("int")
        result = max(depths.values())
    return result


def get_downsampled_offset(scale_factors: Sequence[int]) -> Any:
    """
    For a given number of dimensions and a sequence of downscale factors, calculate the starting offset of the downscaled
    array in the units of the full-resolution data.
    """
    return np.array([np.arange(s).mean() for s in scale_factors])


def downscale_slice(sl: slice, scale: int) -> slice:
    """
    Downscale the start, stop, and step of a slice by an integer factor. Ceiling division is used, i.e.
    downscale_slice(Slice(0, 10, None), 3) returns Slice(0, 4, None).
    """

    start, stop, step = sl.start, sl.stop, sl.step
    if start:
        start = int(np.ceil(sl.start / scale))
    if stop:
        stop = int(np.ceil(sl.stop / scale))
    if step:
        step = int(np.ceil(sl.step / scale))
    result = slice(start, stop, step)

    return result


def slice_span(sl: slice) -> int:
    """
    Measure the length of a slice
    """
    return sl.stop - sl.start


def blocked_pyramid(
    arr, block_size: Sequence, scale_factors: Sequence[int] = (2, 2, 2), **kwargs
):
    full_pyr = multiscale(arr, scale_factors=scale_factors, **kwargs)
    slices = slices_from_chunks(normalize_chunks(block_size, arr.shape))
    absolute_block_size = tuple(map(slice_span, slices[0]))

    results = []
    for idx, sl in enumerate(slices):
        regions = [
            tuple(map(downscale_slice, sl, tuple(np.power(scale_factors, exp))))
            for exp in range(len(full_pyr))
        ]
        if tuple(map(slice_span, sl)) == absolute_block_size:
            pyr = multiscale(arr[sl], scale_factors=scale_factors, **kwargs)
        else:
            pyr = [full_pyr[l][r] for l, r in enumerate(regions)]
        assert len(pyr) == len(regions)
        results.append((regions, pyr))
    return results
