import pytest
from xarray_multiscale.multiscale import (
    downscale,
    prepad,
    multiscale,
    even_padding,
    get_downscale_depth,
)
import dask.array as da
import numpy as np
from xarray import DataArray


def test_downscale_depth():
    assert get_downscale_depth((1,), (1,)) == 0
    assert get_downscale_depth((2, 2, 2), (2, 2, 2)) == 1
    assert get_downscale_depth((1, 2, 2), (2, 2, 2)) == 0
    assert get_downscale_depth((4, 4, 4), (2, 2, 2)) == 2
    assert get_downscale_depth((4, 2, 2), (2, 2, 2)) == 2
    assert get_downscale_depth((5, 2, 2), (2, 2, 2)) == 2
    assert get_downscale_depth((5, 2, 2), (2, 2, 2), pad=True) == 3
    assert get_downscale_depth((7, 2, 2), (2, 2, 2)) == 2
    assert get_downscale_depth((7, 2, 2), (2, 2, 2), pad=True) == 3

@pytest.mark.parametrize(("size","scale"), ((10,2), (11,2), (12,2), (13,2)))
def test_even_padding(size: int, scale: int) -> None:
    assert (size + even_padding(size, scale)) % scale == 0

@pytest.mark.parametrize('dim', (1,2,3,4))
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

    downscaled_numpy_float = downscale(
        arr_numpy, np.mean, scale, preserve_dtype=False
    ).compute()
    downscaled_dask_float = downscale(
        arr_dask, np.mean, scale, preserve_dtype=False
    ).compute()

    answer_float = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    assert np.array_equal(downscaled_numpy_float, answer_float)
    assert np.array_equal(downscaled_dask_float, answer_float)

    downscaled_numpy_int = downscale(
        arr_numpy, np.mean, scale, dtype=arr_numpy.dtype
    ).compute()
    downscaled_dask_int = downscale(
        arr_dask, np.mean, scale, dtype=arr_numpy.dtype
    ).compute()

    answer_int = answer_float.astype("int")
    assert np.array_equal(downscaled_numpy_int, answer_int)
    assert np.array_equal(downscaled_dask_int, answer_int)


def test_multiscale():
    ndim = 3
    chunks = (2,) * ndim
    shape = (9,) * ndim
    cropslice = tuple(slice(s) for s in shape)
    cell = np.zeros(np.prod(chunks)).astype("float")
    cell[0] = 1
    cell = cell.reshape(*chunks)
    array = np.tile(cell, np.ceil(np.divide(shape, chunks)).astype("int"))[cropslice]

    pyr_trimmed = multiscale(array, np.mean, 2, pad_mode=None)
    pyr_padded = multiscale(array, np.mean, 2, pad_mode="reflect")

    assert [p.shape for p in pyr_padded] == [
        shape,
        (5, 5, 5),
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
    ]
    assert [p.shape for p in pyr_trimmed] == [shape, (4, 4, 4), (2, 2, 2), (1,1,1)]

    # check that the first multiscale array is identical to the input data
    assert np.array_equal(pyr_padded[0].data.compute(), array)
    assert np.array_equal(pyr_trimmed[0].data.compute(), array)

    assert np.array_equal(
        pyr_trimmed[-2].data.mean().compute(), pyr_trimmed[-1].data.compute().mean()
    )
    assert np.allclose(pyr_padded[0].data.mean().compute(), 0.17146776406035666)
