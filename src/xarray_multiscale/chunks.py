from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, cast

if TYPE_CHECKING:
    from typing import Sequence

import dask.array as da
import toolz as tz
import xarray
from dask.array.routines import aligned_coarsen_chunks
from xarray.core.utils import is_dict_like


def normalize_chunks(
    array: xarray.DataArray, chunk_size: str | int | Sequence[int] | dict[Hashable, int]
) -> dict[Hashable, int]:
    """
    Given an `xarray.DataArray`, normalize a chunk size against that array.

    Parameters
    ----------
    array: xarray.DataArray
        An `xarray.DataArray`.
    chunk_size: Union[str, int, Sequence[int], dict[Hashable, int]]
        A specification of a chunk size.

    Returns
    -------
    dict[Hashable, int]
        An xarray-compatible specification of chunk sizes.
    """
    _chunk_size: str | int | Sequence[int] | dict[Hashable, int]
    if not isinstance(chunk_size, (int, str, dict)):
        if len(chunk_size) != array.ndim:
            msg = msg = f"Incorrect number of chunks. Got {len(chunk_size)}, expected {array.ndim}."
            raise ValueError(msg)

    if is_dict_like(chunk_size):
        # dask's normalize chunks routine assumes dict inputs have integer
        # keys, so convert dim names to the corresponding integers
        chunk_size = cast(dict[Hashable, int], chunk_size)
        if len(chunk_size.keys() - set(array.dims)) > 0:
            extra: set[Hashable] = chunk_size.keys() - set(array.dims)
            msg = f"Keys of chunksize must be a subset of array dims. Got extraneous keys: {extra}."
            raise ValueError(msg)
        _chunk_size = dict(zip(range(array.ndim), map(tz.first, array.chunks)))
        _chunk_size.update({array.get_axis_num(d): c for d, c in chunk_size.items()})
    else:
        _chunk_size = chunk_size

    new_chunks: tuple[int, ...] = tuple(
        map(
            tz.first,
            da.core.normalize_chunks(
                _chunk_size,
                array.shape,
                dtype=array.dtype,
                previous_chunks=array.data.chunksize,
            ),
        )
    )

    return {dim: new_chunks[array.get_axis_num(dim)] for dim in array.dims}


def align_chunks(array: da.core.Array, scale_factors: Sequence[int]) -> da.core.Array:
    """
    Ensure that all chunks of a dask array are divisible by scale_factors, rechunking the array
    if necessary.
    """
    new_chunks = {}
    for idx, factor in enumerate(scale_factors):
        aligned = aligned_coarsen_chunks(array.chunks[idx], factor)
        if aligned != array.chunks[idx]:
            new_chunks[idx] = aligned
    if new_chunks:
        array = array.rechunk(new_chunks)
    return array
