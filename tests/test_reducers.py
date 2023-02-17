from typing import Tuple

import dask.array as da
import numpy as np
import pytest

from xarray_multiscale.reducers import (
    reshape_windowed,
    windowed_max,
    windowed_mean,
    windowed_min,
    windowed_mode,
)


@pytest.mark.parametrize("ndim", (1, 2, 3))
@pytest.mark.parametrize("window_size", (1, 2, 3, 4, 5))
def test_windowed_mean(ndim: int, window_size: int):
    cell = (window_size,) * ndim
    cell_scaling = 4
    size = np.array(cell) * cell_scaling
    data = da.random.randint(0, 255, size=size.tolist(), chunks=cell)
    result = windowed_mean(data.compute(), cell)
    test = data.map_blocks(lambda v: v.mean(keepdims=True)).compute()
    assert np.array_equal(result, test)


@pytest.mark.parametrize("ndim", (1, 2, 3))
@pytest.mark.parametrize("window_size", (1, 2, 3, 4, 5))
def test_windowed_max(ndim: int, window_size: int):
    cell = (window_size,) * ndim
    cell_scaling = 4
    size = np.array(cell) * cell_scaling
    data = da.random.randint(0, 255, size=size.tolist(), chunks=cell)
    result = windowed_max(data.compute(), cell)
    test = data.map_blocks(lambda v: v.max(keepdims=True)).compute()
    assert np.array_equal(result, test)


@pytest.mark.parametrize("ndim", (1, 2, 3))
@pytest.mark.parametrize("window_size", (1, 2, 3, 4, 5))
def test_windowed_min(ndim: int, window_size: int):
    cell = (window_size,) * ndim
    cell_scaling = 4
    size = np.array(cell) * cell_scaling
    data = da.random.randint(0, 255, size=size.tolist(), chunks=cell)
    result = windowed_min(data.compute(), cell)
    test = data.map_blocks(lambda v: v.min(keepdims=True)).compute()
    assert np.array_equal(result, test)


def test_windowed_mode():
    data = np.arange(16) % 3 + np.arange(16) % 2
    answer = np.array([2, 0, 1, 2])
    results = windowed_mode(data, (4,))
    # only compare regions with a majority value
    assert np.array_equal(results[[0, 2, 3]], answer[[0, 2, 3]])

    data = np.arange(16).reshape(4, 4) % 3
    answer = np.array([[1, 0], [0, 2]])
    results = windowed_mode(data, (2, 2))
    assert np.array_equal(results, answer)


@pytest.mark.parametrize("windows_per_dim", (1, 2, 3, 4, 5))
@pytest.mark.parametrize(
    "window_size", ((1,), (2,), (1, 2), (2, 2), (2, 2, 2), (1, 2, 3), (3, 3, 3, 3))
)
def test_reshape_windowed(windows_per_dim: int, window_size: Tuple[int, ...]):
    size = (windows_per_dim * np.array(window_size)).tolist()
    data = np.arange(np.prod(size)).reshape(size)
    reshaped = reshape_windowed(data, window_size)
    with pytest.raises(ValueError):
        reshape_windowed(data, [*window_size, 1])
    assert reshaped.shape[0::2] == (windows_per_dim,) * len(window_size)
    assert reshaped.shape[1::2] == window_size
    slice_data = tuple(slice(w) for w in window_size)
    slice_reshaped = tuple(
        slice(None) if s % 2 else slice(0, 1) for s in range(reshaped.ndim)
    )
    # because we are reshaping the array, if the first window is correct, all the others
    # will be correct too
    assert np.array_equal(
        data[slice_data].squeeze(), reshaped[slice_reshaped].squeeze()
    )
