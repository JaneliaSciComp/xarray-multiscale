import pytest
from xarray_multiscale.multiscale import (
    align_chunks,
    downscale,
    broadcast_to_rank,
    adjust_shape,
    downscale_coords,
    downscale_dask,
    multiscale,
    get_downscale_depth,
    normalize_chunks,
    ensure_minimum_chunks
)
from xarray_multiscale.reducers import reshape_with_windows, windowed_mean, windowed_mode
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


@pytest.mark.parametrize(("size", "scale"), ((10, 2), (11, 2), ((10,11), (2,3))))
def test_adjust_shape(size, scale):
    arr = DataArray(np.zeros(size))
    padded = adjust_shape(arr, scale, mode="constant")
    scale_array = np.array(scale)
    old_shape_array = np.array(arr.shape)
    new_shape_array = np.array(padded.shape)
    
    if np.all((old_shape_array % scale_array) == 0):
        assert np.array_equal(new_shape_array, old_shape_array)
    else:
        assert np.array_equal(new_shape_array, old_shape_array + ((scale_array - (old_shape_array % scale_array))))

    cropped = adjust_shape(arr, scale, mode="crop")
    new_shape_array = np.array(cropped.shape)
    if np.all((old_shape_array % scale_array) == 0):
        assert np.array_equal(new_shape_array, old_shape_array)
    else:
        assert np.array_equal(new_shape_array, old_shape_array - (old_shape_array % scale_array))

def test_downscale_2d():
    chunks = (2, 2)
    scale = (2, 1)

    data = DataArray(da.from_array(np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype="uint8"
    ), chunks=chunks))
    answer = DataArray(np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]))
    downscaled = downscale(data, windowed_mean, scale, pad_mode='crop').compute()
    assert np.array_equal(downscaled, answer)


def test_downscale_coords():
    data = DataArray(np.zeros((10, 10)), dims=('x','y'), coords={'x': np.arange(10)})
    scale_factors = (2,1)
    downscaled = downscale_coords(data, scale_factors)
    answer = {'x': data['x'].coarsen({'x' : scale_factors[0]}).mean()}
    
    assert downscaled.keys() == answer.keys()
    for k in downscaled:
        assert_equal(answer[k], downscaled[k])

    data = DataArray(np.zeros((10, 10)), 
                     dims=('x','y'), 
                     coords={'x': np.arange(10), 
                             'y': 5 + np.arange(10)})
    scale_factors = (2,1)
    downscaled = downscale_coords(data, scale_factors)
    answer = {'x': data['x'].coarsen({'x' : scale_factors[0]}).mean(),
             'y' : data['y'].coarsen({'y' : scale_factors[1]}).mean()}
    
    assert downscaled.keys() == answer.keys()
    for k in downscaled:
        assert_equal(answer[k], downscaled[k])

    data = DataArray(np.zeros((10, 10)), 
                     dims=('x','y'), 
                     coords={'x': np.arange(10), 
                             'y': 5 + np.arange(10),
                             'foo' : 5})
    scale_factors = (2,2)
    downscaled = downscale_coords(data, scale_factors)
    answer = {'x': data['x'].coarsen({'x' : scale_factors[0]}).mean(),
             'y' : data['y'].coarsen({'y' : scale_factors[1]}).mean(),
             'foo': data['foo']}
    
    assert downscaled.keys() == answer.keys()
    for k in downscaled:
        assert_equal(answer[k], downscaled[k])


def test_invalid_multiscale():
    with pytest.raises(ValueError):
        downscale_dask(np.arange(10), windowed_mean, (3,))
    with pytest.raises(ValueError):
        downscale_dask(np.arange(16).reshape(4,4), windowed_mean, (3,3))


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
    pyr_trimmed = multiscale(base_array, windowed_mean, 2, pad_mode="crop")
    pyr_padded = multiscale(base_array, windowed_mean, 2, pad_mode="constant")
    pyr_trimmed_unchained = multiscale(
        base_array, windowed_mean, 2, pad_mode="crop", chained=False
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
    reducer = windowed_mean
    multi = multiscale(base_array, reducer, 2, chunks=chunks)
    assert all([m.data.chunksize == chunks for m in multi])

    chunks = (3,) * ndim
    multi = multiscale(base_array, reducer, 2, chunks=chunks)
    for m in multi:
        assert m.data.chunksize == chunks or m.data.chunksize == m.data.shape

    chunks = (3,) * ndim
    multi = multiscale(base_array, reducer, 2, chunks=chunks, chunk_mode='minimum')
    for m in multi:
        assert np.greater_equal(m.data.chunksize, chunks).all() or m.data.chunksize == m.data.shape

    chunks = 3
    multi = multiscale(base_array, reducer, 2, chunks=chunks, chunk_mode='minimum')
    for m in multi:
        assert np.greater_equal(m.data.chunksize, (chunks,) * ndim).all() or m.data.chunksize == m.data.shape 


def test_depth():
    ndim = 3
    shape = (16,) * ndim
    base_array = np.zeros(shape)
    reducer = windowed_mean
    full = multiscale(base_array, reducer, 2, depth=-1)
    assert len(full) == 5

    partial = multiscale(base_array, reducer, 2, depth=-2)
    assert len(partial) == len(full) - 1 
    [assert_equal(a,b) for a,b in zip(full, partial)]

    partial = multiscale(base_array, reducer, 2, depth=2)
    assert len(partial) == 3 
    [assert_equal(a,b) for a,b in zip(full, partial)]

    partial = multiscale(base_array, reducer, 2, depth=0)
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

    multi = multiscale(dataarray, windowed_mean, (2, 2, 2), preserve_dtype=False)

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


def test_broadcast_to_rank():
    assert broadcast_to_rank(2, 1) == (2,)
    assert broadcast_to_rank(2, 2) == (2,2)
    assert broadcast_to_rank((2,3), 2) == (2,3)
    assert broadcast_to_rank({0 : 2}, 3) == (2,1,1)


def test_align_chunks():
    data = da.arange(10, chunks=1)
    rechunked = align_chunks(data, scale_factors=(2,))
    assert rechunked.chunks == ((2,) * 5,)

    data = da.arange(10, chunks=2)
    rechunked = align_chunks(data, scale_factors=(2,))
    assert rechunked.chunks == ((2,) * 5,)

    data = da.arange(10, chunks=(1,1,3,5))
    rechunked = align_chunks(data, scale_factors=(2,))
    assert rechunked.chunks == ((2, 2, 2, 4,),)


def test_reshape_with_windows():
    data = np.arange(36).reshape(6,6)
    assert reshape_with_windows(data, (2,2)).shape == (3,2,3,2)