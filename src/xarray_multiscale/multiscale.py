from typing import Any, Dict, Hashable, List, Sequence, Union

import numpy as np
import numpy.typing as npt
from dask.array.core import Array
from dask.base import tokenize
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply
from xarray import DataArray

from xarray_multiscale.chunks import align_chunks, normalize_chunks
from xarray_multiscale.reducers import WindowedReducer
from xarray_multiscale.util import adjust_shape, broadcast_to_rank, logn


def multiscale(
    array: npt.NDArray[Any],
    reduction: WindowedReducer,
    scale_factors: Union[Sequence[int], int],
    preserve_dtype: bool = True,
    chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
    chained: bool = True,
) -> List[DataArray]:
    """
    Generate a coordinate-aware multiscale representation of an array.

    Parameters
    ----------
    array : Array-like, e.g. Numpy array, Dask array
        The array to be downscaled.

    reduction : callable
        A function that aggregates chunks of data over windows.
        See `xarray_multiscale.reducers.WindowedReducer` for the expected
        signature of this callable.

    scale_factors : int or sequence of ints
        The desired downscaling factors, one for each axis, or a single
        value for all axes.

    preserve_dtype : bool, default=True
        If True, output arrays are all cast to the same data type as the
        input array. If False, output arrays will have data type determined
        by the output of the reduction function.

    chunks : sequence or dict of ints, or the string "auto" (default)
        Set the chunking of the output arrays. Applies only to dask arrays.
        If `chunks` is set to "auto" (the default), then chunk sizes will
        decrease with each level of downsampling.

        Otherwise, this keyword argument will be passed to the
        `xarray.DataArray.chunk` method for each output array,
        producing a list of arrays with the same chunk size.
        Note that rechunking can be computationally expensive
        for arrays with many chunks.

    chained : bool, default=True
        If True (default), the nth downscaled array is generated by
        applying the reduction function on the n-1th downscaled array with
        the user-supplied `scale_factors`. This means that the nth
        downscaled array directly depends on the n-1th downscaled array.
        Note that nonlinear reductions like the windowed mode may give
        inaccurate results with `chained` set to True.

        If False, the nth downscaled array is generated by applying the
        reduction function on the 0th downscaled array
        (i.e., the input array) with the `scale_factors` raised to the nth
        power. This means that the nth downscaled array directly depends
        on the input array.

    Returns
    -------
    result : list of DataArrays
        The first element of this list is the input array, converted to an
        `xarray.DataArray`. Each subsquent element of the list is
        the result of downsampling the previous element of the list.

        The `coords` attributes of these DataArrays track the changing
        offset and scale induced by the downsampling operation.

    Examples
    --------
    >>> import numpy as np
    >>> from xarray_multiscale import multiscale
    >>> from xarray_multiscale.reducers import windowed_mean
    >>> multiscale(np.arange(4), windowed_mean, 2)
    [<xarray.DataArray (dim_0: 4)>
    array([0, 1, 2, 3])
    Coordinates:
      * dim_0    (dim_0) float64 0.0 1.0 2.0 3.0, <xarray.DataArray (dim_0: 2)>
    array([0, 2])
    Coordinates:
      * dim_0    (dim_0) float64 0.5 2.5]
    """
    scale_factors = broadcast_to_rank(scale_factors, array.ndim)
    darray = to_dataarray(array)

    levels = range(1, downsampling_depth(darray.shape, scale_factors))

    result: List[DataArray] = [darray]
    for level in levels:
        if chained:
            scale = scale_factors
            source = result[-1]
        else:
            scale = tuple(s**level for s in scale_factors)
            source = result[0]
        result.append(downscale(source, reduction, scale, preserve_dtype))

    if darray.chunks is not None:
        new_chunks = [normalize_chunks(r, chunks) for r in result]
        result = [r.chunk(ch) for r, ch in zip(result, new_chunks)]

    return result


def to_dataarray(array: Any) -> DataArray:
    """
    Convert the input to DataArray if it is not already one.
    """
    if isinstance(array, DataArray):
        data = array.data
        dims = array.dims
        # ensure that key order matches dimension order
        coords = {d: array.coords[d] for d in dims}
        attrs = array.attrs
        name = array.name
    else:
        data = array
        dims = tuple(f"dim_{d}" for d in range(data.ndim))
        coords = {
            dim: DataArray(np.arange(shape, dtype="float"), dims=dim)
            for dim, shape in zip(dims, array.shape)
        }
        name = None
        attrs = {}

    result = DataArray(data=data, coords=coords, dims=dims, attrs=attrs, name=name)
    return result


def downscale_dask(
    array: Any,
    reduction: WindowedReducer,
    scale_factors: Union[int, Sequence[int], Dict[int, int]],
    **kwargs: Any,
) -> Any:

    if not np.all((np.array(array.shape) % np.array(scale_factors)) == 0):
        raise ValueError(
            f"""
            Coarsening factors {scale_factors} do not align
            with array shape {array.shape}.
            """
        )

    array = align_chunks(array, scale_factors)
    name: str = "downscale-" + tokenize(reduction, array, scale_factors)
    dsk = {
        (name,) + key[1:]: (apply, reduction, [key, scale_factors], kwargs)
        for key in flatten(array.__dask_keys__())
    }
    chunks = tuple(
        tuple(int(size // scale_factors[axis]) for size in sizes)
        for axis, sizes in enumerate(array.chunks)
    )

    meta = reduction(
        np.empty(scale_factors, dtype=array.dtype), scale_factors, **kwargs
    )
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[array])
    return Array(graph, name, chunks, meta=meta)


def downscale(
    array: DataArray,
    reduction: WindowedReducer,
    scale_factors: Sequence[int],
    preserve_dtype: bool = True,
    **kwargs: Any,
) -> Any:

    to_downscale = adjust_shape(array, scale_factors)
    if to_downscale.chunks is not None:
        downscaled_data = downscale_dask(
            to_downscale.data, reduction, scale_factors, **kwargs
        )
    else:
        downscaled_data = reduction(to_downscale.data, scale_factors)
    if preserve_dtype:
        downscaled_data = downscaled_data.astype(array.dtype)
    downscaled_coords = downscale_coords(to_downscale, scale_factors)
    return DataArray(
        downscaled_data, downscaled_coords, attrs=array.attrs, dims=array.dims
    )


def downscale_coords(
    array: DataArray, scale_factors: Sequence[int]
) -> Dict[Hashable, Any]:
    """
    Downscale coordinates by taking the windowed mean of each coordinate array.
    """
    new_coords = {}
    for (
        coord_name,
        coord,
    ) in array.coords.items():
        coarsening_dims = {
            d: scale_factors[idx] for idx, d in enumerate(array.dims) if d in coord.dims
        }
        new_coords[coord_name] = coord.coarsen(coarsening_dims).mean()
    return new_coords


def downsampling_depth(shape: Sequence[int], scale_factors: Sequence[int]) -> int:
    """
    For a shape and a sequence of scale factors, calculate the
    number of downsampling operations that must be performed to produce
    a downsampled shape with at least one singleton value.

    If any element of `scale_factors` is greater than the
    corresponding shape, this function returns 0.

    If all `scale_factors` are 1, this function returns 0.

    Parameters
    ----------
    shape : sequence of positive integers

    scale_factors : sequence of positive integers

    Examples
    --------
    >>> downsampling_depth((8,), (2,))
    3
    >>> downsampling_depth((8,2), (2,2))
    1
    >>> downsampling_depth((7,), (2,))
    2
    """
    if len(shape) != len(scale_factors):
        raise ValueError(
            f"""
            Shape (length == {len(shape)} ) and
            scale factors (length == {len(scale_factors)})
            do not align."""
        )

    _scale_factors = np.array(scale_factors).astype("int")
    _shape = np.array(shape).astype("int")
    valid = _scale_factors > 1
    if not valid.any():
        result = 0
    else:
        depths = np.floor(logn(_shape[valid], _scale_factors[valid]))
        result = min(depths.astype("int"))
    return result
