from xarray_multiscale.reducers import windowed_mean, windowed_mode
import numpy as np


def test_windowed_mode():
    data = np.arange(16) % 3 + np.arange(16) % 2
    answer = np.array([2, 0, 1, 2])
    results = windowed_mode(data, (4,))
    assert np.array_equal(results, answer)

    data = np.arange(16).reshape(4,4) % 3
    answer = np.array([[1,0],[0,2]])
    results = windowed_mode(data, (2,2))
    assert np.array_equal(results, answer)

def test_windowed_mean():
    data = np.arange(16).reshape(4,4) % 2
    answer = np.array([[0.5, 0.5],[0.5, 0.5]])
    results = windowed_mean(data, (2,2))
    assert np.array_equal(results, answer)