import dask
import dask.array as da
from dask.utils import SerializableLock
import xarray
from typing import Sequence


def blocked_store(
    sources: Sequence[xarray.DataArray], targets, chunks=None
) -> Sequence[dask.delayed]:
    stores = []
    for slices, source in sources:
        if chunks is not None:
            rechunked_sources = [
                s.data.rechunk(chunks) for s, z in zip(source, targets)
            ]
        elif hasattr(targets[0], "chunks"):
            rechunked_sources = [
                s.data.rechunk(z.chunks) for s, z in zip(source, targets)
            ]
        else:
            rechunked_sources = [s.data for s in source]

        stores.append(
            da.store(
                rechunked_sources,
                targets,
                lock=SerializableLock(),
                regions=slices,
                compute=False,
            )
        )
    return stores
