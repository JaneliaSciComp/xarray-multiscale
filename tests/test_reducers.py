from xarray_multiscale.reducers import windowed_mean, windowed_mode, reshape_windowed
import numpy as np


def test_windowed_mean():
    data = np.arange(16).reshape(4, 4) % 2
    answer = np.array([[0.5, 0.5], [0.5, 0.5]])
    results = windowed_mean(data, (2, 2))
    assert np.array_equal(results, answer)

    data = np.arange(16).reshape(4, 4, 1) % 2
    answer = np.array([[0.5, 0.5], [0.5, 0.5]]).reshape((2, 2, 1))
    results = windowed_mean(data, (2, 2, 1))


def test_windowed_mode():
    data = np.arange(16) % 3 + np.arange(16) % 2
    answer = np.array([2, 0, 1, 2])
    results = windowed_mode(data, (4,))
    assert np.array_equal(results, answer)

    data = np.arange(16).reshape(4, 4) % 3
    answer = np.array([[1, 0], [0, 2]])
    results = windowed_mode(data, (2, 2))
    assert np.array_equal(results, answer)


def test_reshape_windowed():
    data = np.arange(36).reshape(6, 6)
    window = (2, 2)
    windowed = reshape_windowed(data, window)
    assert windowed.shape == (3, 2, 3, 2)
    assert np.all(windowed[0, :, 0, :] == data[: window[0], : window[1]])
