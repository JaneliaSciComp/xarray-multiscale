import dask.array as da
import numpy as np
import pytest
from xarray import DataArray
from xarray.testing import assert_equal

from xarray_multiscale.multiscale import (adjust_shape, downsampling_depth,
                                          downscale, downscale_coords,
                                          downscale_dask, multiscale)
from xarray_multiscale.reducers import windowed_mean


def test_downscale_depth():
    assert downsampling_depth((1,), (1,)) == 0
    assert downsampling_depth((2,), (3,)) == 0
    assert downsampling_depth((2, 1), (2, 1)) == 1
    assert downsampling_depth((2, 2, 2), (2, 2, 2)) == 1
    assert downsampling_depth((1, 2, 2), (2, 2, 2)) == 0
    assert downsampling_depth((4, 4, 4), (2, 2, 2)) == 2
    assert downsampling_depth((4, 2, 2), (2, 2, 2)) == 1
    assert downsampling_depth((5, 2, 2), (2, 2, 2)) == 1
    assert downsampling_depth((7, 2, 2), (2, 2, 2)) == 1
    assert downsampling_depth((1500, 5495, 5200), (2, 2, 2)) == 10


@pytest.mark.parametrize(("size", "scale"), ((10, 2), (11, 2), ((10, 11), (2, 3))))
def test_adjust_shape(size, scale):
    arr = DataArray(np.zeros(size))
    scale_array = np.array(scale)
    old_shape_array = np.array(arr.shape)

    cropped = adjust_shape(arr, scale)
    new_shape_array = np.array(cropped.shape)
    if np.all((old_shape_array % scale_array) == 0):
        assert np.array_equal(new_shape_array, old_shape_array)
    else:
        assert np.array_equal(
            new_shape_array, old_shape_array - (old_shape_array % scale_array)
        )


def test_downscale_2d():
    scale = (2, 1)

    data = DataArray(
        np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype="uint8"
        ),
    )
    answer = DataArray(np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]))
    downscaled = downscale(data, windowed_mean, scale, preserve_dtype=False)
    downscaled_old_dtype = downscale(data, windowed_mean, scale, preserve_dtype=True)
    assert np.array_equal(downscaled, answer)
    assert np.array_equal(
        downscaled_old_dtype,
        answer.astype(data.dtype),
    )


def test_downscale_coords():
    data = DataArray(np.zeros((10, 10)), dims=("x", "y"), coords={"x": np.arange(10)})
    scale_factors = (2, 1)
    downscaled = downscale_coords(data, scale_factors)
    answer = {"x": data["x"].coarsen({"x": scale_factors[0]}).mean()}

    assert downscaled.keys() == answer.keys()
    for k in downscaled:
        assert_equal(answer[k], downscaled[k])

    data = DataArray(
        np.zeros((10, 10)),
        dims=("x", "y"),
        coords={"x": np.arange(10), "y": 5 + np.arange(10)},
    )
    scale_factors = (2, 1)
    downscaled = downscale_coords(data, scale_factors)
    answer = {
        "x": data["x"].coarsen({"x": scale_factors[0]}).mean(),
        "y": data["y"].coarsen({"y": scale_factors[1]}).mean(),
    }

    assert downscaled.keys() == answer.keys()
    for k in downscaled:
        assert_equal(answer[k], downscaled[k])

    data = DataArray(
        np.zeros((10, 10)),
        dims=("x", "y"),
        coords={"x": np.arange(10), "y": 5 + np.arange(10), "foo": 5},
    )
    scale_factors = (2, 2)
    downscaled = downscale_coords(data, scale_factors)
    answer = {
        "x": data["x"].coarsen({"x": scale_factors[0]}).mean(),
        "y": data["y"].coarsen({"y": scale_factors[1]}).mean(),
        "foo": data["foo"],
    }

    assert downscaled.keys() == answer.keys()
    for k in downscaled:
        assert_equal(answer[k], downscaled[k])


def test_invalid_multiscale():
    with pytest.raises(ValueError):
        downscale_dask(np.arange(10), windowed_mean, (3,))
    with pytest.raises(ValueError):
        downscale_dask(np.arange(16).reshape(4, 4), windowed_mean, (3, 3))


@pytest.mark.parametrize("chained", (True, False))
@pytest.mark.parametrize("ndim", (1, 2, 3, 4))
def test_multiscale(ndim: int, chained: bool):
    chunks = (2,) * ndim
    shape = (9,) * ndim
    cropslice = tuple(slice(s) for s in shape)
    cell = np.zeros(np.prod(chunks)).astype("float")
    cell[0] = 1
    cell = cell.reshape(*chunks)
    base_array = np.tile(cell, np.ceil(np.divide(shape, chunks)).astype("int"))[
        cropslice
    ]

    pyr = multiscale(base_array, windowed_mean, 2, chained=chained)
    assert [p.shape for p in pyr] == [shape, (4,) * ndim, (2,) * ndim]

    # check that the first multiscale array is identical to the input data
    assert np.array_equal(pyr[0].data, base_array)


def test_chunking():
    ndim = 3
    shape = (9,) * ndim
    base_array = da.zeros(shape, chunks=(1,) * ndim)
    chunks = (1,) * ndim
    reducer = windowed_mean
    multi = multiscale(base_array, reducer, 2, chunks=chunks)
    assert all([m.data.chunksize == chunks for m in multi])

    chunks = (3,) * ndim
    multi = multiscale(base_array, reducer, 2, chunks=chunks)
    for m in multi:
        assert m.data.chunksize == chunks or m.data.chunksize == m.data.shape

    chunks = (3,) * ndim
    multi = multiscale(base_array, reducer, 2, chunks=chunks)
    for m in multi:
        assert (
            np.greater_equal(m.data.chunksize, chunks).all()
            or m.data.chunksize == m.data.shape
        )

    chunks = 3
    multi = multiscale(base_array, reducer, 2, chunks=chunks)
    for m in multi:
        assert (
            np.greater_equal(m.data.chunksize, (chunks,) * ndim).all()
            or m.data.chunksize == m.data.shape
        )


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
    array = DataArray(base_array, coords=coords)
    downscaled = array.coarsen({"z": 2, "y": 2, "x": 2}).mean()

    multi = multiscale(array, windowed_mean, (2, 2, 2), preserve_dtype=False)

    assert_equal(multi[0], array)
    assert_equal(multi[1], downscaled)
