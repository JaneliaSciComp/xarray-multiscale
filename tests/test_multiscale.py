import pytest
import xarray
from xarray.core import dataarray
from xarray_multiscale.multiscale import (
    downscale,
    prepad,
    multiscale,
    even_padding,
    get_downscale_depth,
    normalize_chunks,
    ensure_minimum_chunks
)
import dask.array as da
import numpy as np
from xarray import DataArray
from xarray.testing import assert_equal


def test_downscale_depth():
    assert get_downscale_depth((1,), (1,)) == 0
    assert get_downscale_depth((2, 2, 2), (2, 2, 2)) == 1
    assert get_downscale_depth((1, 2, 2), (2, 2, 2)) == 0
    assert get_downscale_depth((4, 4, 4), (2, 2, 2)) == 2
    assert get_downscale_depth((4, 2, 2), (2, 2, 2)) == 1
    assert get_downscale_depth((5, 2, 2), (2, 2, 2)) == 1
    assert get_downscale_depth((5, 3, 3), (2, 2, 2), pad=True) == 2
    assert get_downscale_depth((7, 2, 2), (2, 2, 2)) == 1
    assert get_downscale_depth((7, 3, 3), (2, 2, 2), pad=True) == 2
    assert get_downscale_depth((1500, 5495, 5200), (2, 2, 2)) == 10


@pytest.mark.parametrize(("size", "scale"), ((10, 2), (11, 2), (12, 2), (13, 2)))
def test_even_padding(size: int, scale: int) -> None:
    assert (size + even_padding(size, scale)) % scale == 0


@pytest.mark.parametrize("dim", (1, 2, 3, 4))
def test_prepad(dim: int) -> None:
    size = (10,) * dim
    chunks = (9,) * dim
    scale = (2,) * dim

    arr = da.zeros(size, chunks=chunks)
    arr2 = DataArray(arr)

    padded = prepad(arr, scale)
    assert np.all(np.mod(padded.shape, scale) == 0)

    padded2 = prepad(arr2, scale)
    assert np.all(np.mod(padded2.shape, scale) == 0)


def test_downscale_2d():
    chunks = (2, 2)
    scale = (2, 1)

    arr_numpy = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype="uint8"
    )
    arr_dask = da.from_array(arr_numpy, chunks=chunks)
    arr_xarray = DataArray(arr_dask)

    downscaled_numpy_float = downscale(arr_numpy, np.mean, scale).compute()

    downscaled_dask_float = downscale(arr_dask, np.mean, scale).compute()

    downscaled_xarray_float = downscale(arr_xarray, np.mean, scale).compute()

    answer_float = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])

    assert np.array_equal(downscaled_numpy_float, answer_float)
    assert np.array_equal(downscaled_dask_float, answer_float)
    assert np.array_equal(downscaled_xarray_float, answer_float)


def test_multiscale():
    ndim = 3
    chunks = (2,) * ndim
    shape = (9,) * ndim
    cropslice = tuple(slice(s) for s in shape)
    cell = np.zeros(np.prod(chunks)).astype("float")
    cell[0] = 1
    cell = cell.reshape(*chunks)
    base_array = np.tile(cell, np.ceil(np.divide(shape, chunks)).astype("int"))[
        cropslice
    ]
    pyr_trimmed = multiscale(base_array, np.mean, 2, pad_mode=None)
    pyr_padded = multiscale(base_array, np.mean, 2, pad_mode="reflect")
    pyr_trimmed_unchained = multiscale(
        base_array, np.mean, 2, pad_mode=None, chained=False
    )
    assert [p.shape for p in pyr_padded] == [
        shape,
        (5, 5, 5),
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
    ]
    assert [p.shape for p in pyr_trimmed] == [shape, (4, 4, 4), (2, 2, 2), (1, 1, 1)]

    # check that the first multiscale array is identical to the input data
    assert np.array_equal(pyr_padded[0].data.compute(), base_array)
    assert np.array_equal(pyr_trimmed[0].data.compute(), base_array)

    assert np.array_equal(
        pyr_trimmed[-2].data.mean().compute(), pyr_trimmed[-1].data.compute().mean()
    )
    assert np.array_equal(
        pyr_trimmed_unchained[-2].data.mean().compute(),
        pyr_trimmed_unchained[-1].data.compute().mean(),
    )
    assert np.allclose(pyr_padded[0].data.mean().compute(), 0.17146776406035666)


def test_chunking():
    ndim = 3
    shape = (9,) * ndim
    base_array = da.zeros(shape, chunks=(1,) * ndim)
    chunks = (1,) * ndim
    multi = multiscale(base_array, np.mean, 2, chunks=chunks)
    assert all([m.data.chunksize == chunks for m in multi])

    chunks = (3,) * ndim
    multi = multiscale(base_array, np.mean, 2, chunks=chunks)
    for m in multi:
        assert m.data.chunksize == chunks or m.data.chunksize == m.data.shape

    chunks = (3,) * ndim
    multi = multiscale(base_array, np.mean, 2, chunks=chunks, chunk_mode='minimum')
    for m in multi:
        assert np.greater_equal(m.data.chunksize, chunks).all() or m.data.chunksize == m.data.shape

    chunks = 3
    multi = multiscale(base_array, np.mean, 2, chunks=chunks, chunk_mode='minimum')
    for m in multi:
        assert np.greater_equal(m.data.chunksize, (chunks,) * ndim).all() or m.data.chunksize == m.data.shape 


def test_depth():
    ndim = 3
    shape = (16,) * ndim
    base_array = np.zeros(shape)

    full = multiscale(base_array, np.mean, 2, depth=-1)
    assert len(full) == 5

    partial = multiscale(base_array, np.mean, 2, depth=-2)
    assert len(partial) == len(full) - 1 
    [assert_equal(a,b) for a,b in zip(full, partial)]

    partial = multiscale(base_array, np.mean, 2, depth=2)
    assert len(partial) == 3 
    [assert_equal(a,b) for a,b in zip(full, partial)]

    partial = multiscale(base_array, np.mean, 2, depth=0)
    assert len(partial) == 1 
    [assert_equal(a,b) for a,b in zip(full, partial)]


def test_coords():
    dims = ("z", "y", "x")
    shape = (16,) * len(dims)
    base_array = np.random.randint(0, 255, shape, dtype="uint8")

    translates = (0.0, -10, 10)
    scales = (1.0, 2.0, 3.0)
    coords = tuple(
        (d, sc * (np.arange(shp) + tr))
        for d, sc, shp, tr in zip(dims, scales, base_array.shape, translates)
    )
    dataarray = DataArray(base_array, coords=coords)
    downscaled = dataarray.coarsen({"z": 2, "y": 2, "x": 2}).mean()

    multi = multiscale(dataarray, np.mean, (2, 2, 2), preserve_dtype=False)

    assert_equal(multi[0], dataarray)
    assert_equal(multi[1], downscaled)


def test_normalize_chunks():
    data = DataArray(da.zeros((4,6), chunks=(1,1)))
    assert normalize_chunks(data, {'dim_0' : 2, 'dim_1' : 1}) == (2,1)

def test_ensure_minimum_chunks():
    data = da.zeros((4,6), chunks=(1,1))
    assert ensure_minimum_chunks(data, (2,2)) == (2,2)

    data = da.zeros((4,6), chunks=(4,1))
    assert ensure_minimum_chunks(data, (2,2)) == (4,2)