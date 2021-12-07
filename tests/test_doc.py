from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mean
import numpy as np
import xarray as xr


def test_xarray_example():
    data = xr.DataArray(np.zeros((1024, 1024)), dims=list("xy"))
    scaled_data = multiscale(data, windowed_mean, (2, 2))
    assert len(scaled_data) == 11, "Incorret Amount of Arrays returnes"