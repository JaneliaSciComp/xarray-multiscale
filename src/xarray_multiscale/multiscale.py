from dask.base import tokenize
import numpy as np
import dask.array as da
import xarray
from xarray import DataArray
from typing import (
    Any,
    Hashable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Sequence,
    Callable,
    Dict,
    cast,
)
from dask.utils import apply
from dask.core import flatten
from dask.array.routines import aligned_coarsen_chunks
from dask.highlevelgraph import HighLevelGraph
from dask.array import Array
from numpy.typing import NDArray
from xarray.core.utils import is_dict_like

CHUNK_MODES = ("rechunk", "minimum")


def multiscale(
    array: Any,
    reduction: Callable[[NDArray[Any], Tuple[int, ...], Dict[Any, Any]], NDArray[Any]],
    scale_factors: Union[Sequence[int], int],
    depth: int = -1,
    pad_mode: str = "crop",
    preserve_dtype: bool = True,
    chunks: Optional[Union[Sequence[int], Dict[Hashable, int]]] = None,
    chunk_mode: Literal["rechunk", "minimum"] = "minimum",
    chained: bool = True,
) -> List[DataArray]:
    """
    Generate a lazy, coordinate-aware multiscale representation of an array.

    Parameters
    ----------
    array : numpy array, dask array, or xarray DataArray
        The array to be downscaled

    reduction : callable
        A function that aggregates chunks of data over windows.
        See the documentation of `dask.array.coarsen` for the expected
        signature of this callable.

    scale_factors : int or sequence of ints
        The desired downscaling factors, one for each axis.
        If a single int is provide, it will be broadcasted to all axes.

    depth : int, default=-1
        This value determines the number of downscaled arrays to return
        using python indexing semantics. The default value of deth is -1,
        which returns all elements from the multiscale collection.
        Setting depth to a non-negative integer will return a total of
        depth + 1 downscaled arrays.

    pad_mode : string or None, default=None
        How arrays should be padded prior to downscaling in order to ensure that each array dimension
        is evenly divisible by the respective scale factor. When set to `None` (default), the input will be sliced before downscaling
        if its dimensions are not divisible by `scale_factors`.

    preserve_dtype : bool, default=True
        Determines whether the multiresolution arrays are all cast to the same dtype as the input.

    chunks : sequence or dict of ints, or None, default=None
        If `chunks` is supplied, all output arrays are returned with this chunking. If not None, this
        argument is passed directly to the `xarray.DataArray.chunk` method of each output array.

    chunk_mode : str, default='rechunk'
        `chunk_mode` determines how to interpret the `chunks` keyword argument.
        With the default value `rechunk`, all output arrays are rechunked
        to the chunk size specified in `chunks`.
        If `chunk_mode` is set to 'minimum`, output arrays are rechunked
        only if that array has a chunk size smaller than `chunks`.

    chained : bool, default=True
        If True (default), the nth downscaled array is generated by applying the reduction function on the n-1th
        downscaled array with the user-supplied `scale_factors`. This means that the nth downscaled array directly depends on the n-1th
        downscaled array. Note that nonlinear reductions like the windowed mode may give inaccurate results with `chained` set to True.

        If False, the nth downscaled array is generated by applying the reduction function on the 0th downscaled array
        (i.e., the input array) with the `scale_factors` raised to the nth power. This means that the nth downscaled array directly
        depends on the input array.

    Returns
    -------
    result : list of DataArrays
        The `coords` attribute of these DataArrays properties that track the changing offset (if any)
        induced by the downsampling operation. Additionally, the scale factors are stored each DataArray's attrs propery under the key `scale_factors`


    """
    if chunk_mode not in CHUNK_MODES:
        raise ValueError(f"chunk_mode must be one of {CHUNK_MODES}, not {chunk_mode}")

    scale_factors = broadcast_to_rank(scale_factors, array.ndim)
    normalized = normalize_array(array, scale_factors, pad_mode=None)
    needs_padding = not (pad_mode == "crop")

    all_levels = tuple(
        range(
            0,
            1 + get_downscale_depth(normalized.shape, scale_factors, pad=needs_padding),
        )
    )

    if depth < 0:

        indexer = slice(len(all_levels) + depth + 1)
    else:
        indexer = slice(depth + 1)

    levels = all_levels[indexer]
    scales = tuple(tuple(s ** level for s in scale_factors) for level in levels)
    result = [normalized]

    if len(levels) > 1:
        for level in levels[1:]:
            if chained:
                scale = scale_factors
                source = result[-1]
            else:
                scale = scales[level]
                source = result[0]
            downscaled = downscale(source, reduction, scale, pad_mode=pad_mode)
            result.append(downscaled)

    if preserve_dtype:
        result = [r.astype(array.dtype) for r in result]

    if chunks is not None:
        if isinstance(chunks, int):
            new_chunks = ({k: chunks for k in result[0].dims},) * len(result)
        elif isinstance(chunks, Sequence):
            new_chunks = ({k: v for k, v in zip(result[0].dims, chunks)},) * len(result)
        elif isinstance(chunks, dict):
            new_chunks = (chunks,) * len(result)
        else:
            raise ValueError(
                f"Chunks must be an int, a Sequence, or a dict, not {type(chunks)}"
            )

        if chunk_mode == "minimum":
            new_chunks = tuple(
                ensure_minimum_chunks(r.data, normalize_chunks(r.data, chunks))
                for r in result
            )
        result = [r.chunk(ch) for r, ch in zip(result, new_chunks)]

    return result


def normalize_array(
    array: Any, scale_factors: Sequence[int], pad_mode: Union[str, None]
) -> DataArray:
    """
    Ingest an array in preparation for downscaling by converting to DataArray
    and cropping / padding as needed.
    """
    if isinstance(array, DataArray):
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
        dims = tuple(f"dim_{d}" for d in range(data.ndim))
        offset = 0.0
        coords = {
            dim: DataArray(offset + np.arange(shp, dtype="float"), dims=dim)
            for dim, shp in zip(dims, array.shape)
        }
        name = None
        attrs = {}

    dataArray = DataArray(data=data, coords=coords, dims=dims, attrs=attrs, name=name)
    reshaped = adjust_shape(dataArray, scale_factors=scale_factors, mode=pad_mode)
    return reshaped


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


def adjust_shape(
    array: DataArray, scale_factors: Sequence[int], mode: Union[str, None]
) -> DataArray:
    """
    Pad or crop array such that its new dimensions are evenly divisible by a set of integers.

    Parameters
    ----------
    array : ndarray
        Array that will be padded.

    scale_factors : Sequence of ints
        The output array is guaranteed to have dimensions that are each evenly divisible
        by the corresponding scale factor, and chunks that are smaller than or equal
        to the scale factor (if the array has chunks)

    mode : str
        If set to "crop", then the input array will be cropped as needed. Otherwise,
        this is the edge mode used by the padding routine. This parameter will be passed to
        `dask.array.pad` as the `mode` keyword.

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


def downscale_dask(
    array: Any,
    reduction: Callable[[NDArray[Any], Tuple[int, ...]], NDArray[Any]],
    scale_factors: Union[int, Sequence[int], Dict[int, int]],
    **kwargs: Any,
) -> Any:

    if not np.all((np.array(array.shape) % np.array(scale_factors)) == 0):
        raise ValueError(
            f"Coarsening factors {scale_factors} do not align with array shape {array.shape}."
        )

    array = align_chunks(array, scale_factors)
    name = "downscale-" + tokenize(reduction, array, scale_factors)
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
    reduction: Callable[[NDArray[Any], Tuple[int, ...]], NDArray[Any]],
    scale_factors: Sequence[int],
    pad_mode: str,
    **kwargs: Any,
) -> Any:

    to_downscale = normalize_array(array, scale_factors, pad_mode=pad_mode)
    downscaled_data = downscale_dask(
        to_downscale.data, reduction, scale_factors, **kwargs
    )
    downscaled_coords = downscale_coords(to_downscale, scale_factors)
    return DataArray(
        downscaled_data, downscaled_coords, attrs=array.attrs, dims=array.dims
    )


def downscale_coords(
    array: DataArray, scale_factors: Sequence[int]
) -> Dict[Hashable, Any]:
    """
    Take the windowed mean of each coordinate array.
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


def get_downscale_depth(
    shape: Tuple[int, ...], scale_factors: Sequence[int], pad: bool = False
) -> int:
    """
    For an array and a sequence of scale factors, calculate the maximum possible number of downscaling operations.
    If any element of `scale_factors` is greater than the corresponding shape, this function returns 0.
    If all `scale factors` are 1, this function returns 0.
    """
    if len(shape) != len(scale_factors):
        raise ValueError(
            f"Shape (length == {len(shape)} ) and scale factors (length == {len(scale_factors)}) do not align."
        )

    _scale_factors = np.array(scale_factors).astype("int")
    _shape = np.array(shape).astype("int")
    if np.all(_scale_factors == 1):
        result = 0
    elif np.any(_scale_factors > _shape):
        result = 0
    else:
        if pad:
            depths = np.ceil(logn(shape, scale_factors)).astype("int")
        else:
            lg = logn(shape, scale_factors)
            depths = np.floor(logn(shape, scale_factors)).astype("int")
        result = min(depths)
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


def normalize_chunks(
    array: xarray.DataArray, chunks: Union[int, Sequence[int], Dict[Hashable, int]]
) -> Tuple[int, ...]:

    if is_dict_like(chunks):
        chunks = {array.get_axis_num(dim): chunk for dim, chunk in chunks.items()}
    # normalize to explicit chunks, then take the first element from each
    # collection of explicit chunks
    chunks = tuple(c[0] for c in da.core.normalize_chunks(chunks, array.shape))
    cast(Tuple[int, ...], chunks)
    return chunks


def ensure_minimum_chunks(
    array: da.core.Array, chunks: Sequence[int]
) -> Tuple[int, ...]:
    old_chunks = np.array(array.chunksize)
    new_chunks = old_chunks.copy()
    chunk_fitness = np.less(old_chunks, chunks)
    if np.any(chunk_fitness):
        new_chunks[chunk_fitness] = np.array(chunks)[chunk_fitness]
        return tuple(new_chunks.tolist())
    else:
        return tuple(array.chunks)


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


def align_chunks(array: da.core.Array, scale_factors: Sequence[int]) -> da.core.Array:
    """
    Ensure that all chunks are divisible by scale_factors
    """
    new_chunks = {}
    for idx, factor in enumerate(scale_factors):
        aligned = aligned_coarsen_chunks(array.chunks[idx], factor)
        if aligned != array.chunks[idx]:
            new_chunks[idx] = aligned
    if new_chunks:
        array = array.rechunk(new_chunks)
    return array
