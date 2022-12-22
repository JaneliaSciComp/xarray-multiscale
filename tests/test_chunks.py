from xarray_multiscale.chunks import align_chunks, normalize_chunks
from xarray import DataArray
import dask.array as da


def test_normalize_chunks():
    data1 = DataArray(da.zeros((4, 6), chunks=(1, 1)))
    assert normalize_chunks(data1, {"dim_0": 2, "dim_1": 1}) == {"dim_0": 2, "dim_1": 1}

    data2 = DataArray(da.zeros((4, 6), chunks=(1, 1)), dims=("a", "b"))
    assert normalize_chunks(data2, {"a": 2, "b": 1}) == {"a": 2, "b": 1}

    data3 = DataArray(da.zeros((4, 6), chunks=(1, 1)), dims=("a", "b"))
    assert normalize_chunks(data3, {"a": -1, "b": -1}) == {"a": 4, "b": 6}

    data4 = DataArray(da.zeros((4, 6), chunks=(1, 1)), dims=("a", "b"))
    assert normalize_chunks(data4, {"a": -1}) == {"a": 4, "b": 1}


def test_align_chunks():
    data = da.arange(10, chunks=1)
    rechunked = align_chunks(data, scale_factors=(2,))
    assert rechunked.chunks == ((2,) * 5,)

    data = da.arange(10, chunks=2)
    rechunked = align_chunks(data, scale_factors=(2,))
    assert rechunked.chunks == ((2,) * 5,)

    data = da.arange(10, chunks=(1, 1, 3, 5))
    rechunked = align_chunks(data, scale_factors=(2,))
    assert rechunked.chunks == (
        (
            2,
            2,
            2,
            4,
        ),
    )
