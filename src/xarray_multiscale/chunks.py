
from dask.array.routines import aligned_coarsen_chunks
import xarray
from xarray.core.utils import is_dict_like
from typing import Union, Sequence, Hashable, Dict, Set
import dask.array as da
import toolz as tz


def normalize_chunks(
    array: xarray.DataArray,
    chunksize: Union[str, int, Sequence[int], Dict[Hashable, int]],
    chunk_merge_only: bool = False
) -> Dict[Hashable, int]:

    if not isinstance(chunksize, (int, str, dict)):
        if len(chunksize) != array.ndim:
            raise ValueError(f'Incorrect number of chunks. Got {len(chunksize)}, expected {array.ndim}')

    if is_dict_like(chunksize):
        # dask's normalize chunks routine assumes dict inputs have integer
        # keys, so convert dim names to the corresponding integers

        if len(chunksize.keys() - set(array.dims)) > 0:
            extra: Set[Hashable] = chunksize.keys() - set(array.dims)
            raise ValueError(f'Keys of chunksize must be a subset of array dims. Got extraneous keys: {extra}')
        _chunksize = dict(zip(range(array.ndim), map(tz.first, array.chunks)))
        _chunksize.update({array.get_axis_num(dim): chunk for dim, chunk in chunksize.items()})
        chunksize = _chunksize

    old_chunks = map(tz.first, array.chunks)
    new_chunks = map(tz.first, da.core.normalize_chunks(chunksize, array.shape, dtype=array.dtype))

    if chunk_merge_only:
        result = tuple(map(lambda pair: pair[0] if pair[0] > pair[1] else pair[1],
                           zip(old_chunks, new_chunks)))
    else:
        result = tuple(new_chunks)

    return {dim: result[array.get_axis_num(dim)] for dim in array.dims}


def align_chunks(array: da.core.Array,
                 scale_factors: Sequence[int]) -> da.core.Array:
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
