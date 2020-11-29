from typing import Callable
from xarray_multiscale.reducers import mode
import dask.array as da
from scipy.stats import mode as scipy_mode
from typing import Any
import numpy as np


def modefunc(v):
    return scipy_mode(v).mode


def coarsened_comparator(
    func: Callable, source_array: Any, coarsened_array: Any
) -> Any:
    """
    Take a reducer function and two arrays; reduce the first array,
    and check that the result is identical to the second array.
    """
    result = np.array([True]).reshape((1,) * source_array.ndim)
    if np.array_equal(func(source_array), coarsened_array):
        result *= False
    return result


def test_mode2():
    ndim = 2
    data_da = da.random.randint(0, 4, size=(2 ** 3,) * ndim, chunks=(2,) * ndim)
    coarsened = da.coarsen(mode, data_da, {idx: 2 for idx in range(data_da.ndim)})
    results = da.map_blocks(
        coarsened_comparator, modefunc, data_da, coarsened, dtype="bool"
    ).compute()
    assert np.all(results)

    ndim = 3
    data_da = da.random.randint(0, 4, size=(2 ** 3,) * ndim, chunks=(2,) * ndim)
    coarsened = da.coarsen(mode, data_da, {idx: 2 for idx in range(data_da.ndim)})
    results = da.map_blocks(
        coarsened_comparator, modefunc, data_da, coarsened, dtype="bool"
    ).compute()
    assert np.all(results)
