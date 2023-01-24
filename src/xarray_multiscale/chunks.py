from typing import Dict, Hashable, Sequence, Set, Union

import dask.array as da
import toolz as tz
import xarray
from dask.array.routines import aligned_coarsen_chunks
from xarray.core.utils import is_dict_like


def normalize_chunks(
    array: xarray.DataArray,
    chunk_size: Union[str, int, Sequence[int], Dict[Hashable, int]],
) -> Dict[Hashable, int]:

    if not isinstance(chunk_size, (int, str, dict)):
        if len(chunk_size) != array.ndim:
            raise ValueError(
                f"""
                Incorrect number of chunks.
                Got {len(chunk_size)}, expected {array.ndim}
                """
            )

    if is_dict_like(chunk_size):
        # dask's normalize chunks routine assumes dict inputs have integer
        # keys, so convert dim names to the corresponding integers

        if len(chunk_size.keys() - set(array.dims)) > 0:
            extra: Set[Hashable] = chunk_size.keys() - set(array.dims)
            raise ValueError(
                f"""
                Keys of chunksize must be a subset of array dims.
                Got extraneous keys: {extra}.
                """
            )
        _chunk_size = dict(zip(range(array.ndim), map(tz.first, array.chunks)))
        _chunk_size.update({array.get_axis_num(d): c for d, c in chunk_size.items()})
        chunk_size = _chunk_size

    new_chunks = map(
        tz.first,
        da.core.normalize_chunks(
            chunk_size,
            array.shape,
            dtype=array.dtype,
            previous_chunks=array.data.chunksize,
        ),
    )

    result = tuple(new_chunks)

    return {dim: result[array.get_axis_num(dim)] for dim in array.dims}


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
