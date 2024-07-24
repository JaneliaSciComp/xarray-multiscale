# xarray-multiscale

Simple tools for creating multiscale representations of large images.

## Installation

`pip install xarray-multiscale`

## Motivation

Many image processing applications benefit from representing images at multiple scales (also known as [image pyramids] (https://en.wikipedia.org/wiki/Pyramid_(image_processing)). This package provides tools for generating lazy multiscale representations of N-dimensional data using [`xarray`](http://xarray.pydata.org/en/stable/) to ensure that the downsampled images have the correct axis coordinates.

Why are coordinates important for this application? Because a downsampled image is typically scaled and *translated* relative to the source image. Without a coordinate-aware representation of the data, the scaling and translation information is easily lost. 


## Usage

Generate a multiscale representation of a numpy array:

```python
from xarray_multiscale import multiscale, windowed_mean
import numpy as np

data = np.arange(4)
multiscale(data, windowed_mean, 2)
"""
[<xarray.DataArray (dim_0: 4)>
array([0, 1, 2, 3])
Coordinates:
* dim_0    (dim_0) float64 0.0 1.0 2.0 3.0,
<xarray.DataArray (dim_0: 2)>
array([0, 2])
Coordinates:
* dim_0    (dim_0) float64 0.5 2.5]
"""
```


By default, the values of the downsampled arrays are cast to the same data type as the input. This behavior can be changed with the ``preserve_dtype`` keyword argument to ``multiscale``.

Generate a multiscale representation of an ``xarray.DataArray``:


```python
from xarray_multiscale import multiscale, windowed_mean
from xarray import DataArray
import numpy as np

data = np.arange(16).reshape((4,4))
coords = (DataArray(np.arange(data.shape[0]), dims=('y',), attrs={'units' : 'm'}),
            DataArray(np.arange(data.shape[0]), dims=('x',), attrs={'units' : 'm'}))

arr = DataArray(data, coords)
multiscale(arr, windowed_mean, (2,2))
"""
[<xarray.DataArray (y: 4, x: 4)>
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15]])
Coordinates:
* y        (y) int64 0 1 2 3
* x        (x) int64 0 1 2 3, <xarray.DataArray (y: 2, x: 2)>
array([[ 2,  4],
    [10, 12]])
Coordinates:
* y        (y) float64 0.5 2.5
* x        (x) float64 0.5 2.5]
"""
```

Dask arrays work too. Note the control over output chunks via the ``chunks`` keyword argument.

```python
from xarray_multiscale import multiscale, windowed_mean
import dask.array as da

arr = da.random.randint(0, 255, (10,10,10))
multiscale(arr, windowed_mean, 2, chunks=2)
"""
[<xarray.DataArray 'randint-f83260ed51a44f24aeccd95bc23e73ae' (dim_0: 10,
                                                            dim_1: 10,
                                                            dim_2: 10)>
dask.array<rechunk-merge, shape=(10, 10, 10), dtype=int64, chunksize=(2, 2, 2), chunktype=numpy.ndarray>
Coordinates:
* dim_0    (dim_0) float64 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
* dim_1    (dim_1) float64 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
* dim_2    (dim_2) float64 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0,
<xarray.DataArray 'astype-0c3c3e397345ddeedff07ecf2d9fad17' (dim_0: 5,
                                                            dim_1: 5, dim_2: 5)>
dask.array<rechunk-merge, shape=(5, 5, 5), dtype=int64, chunksize=(2, 2, 2), chunktype=numpy.ndarray>
Coordinates:
* dim_0    (dim_0) float64 0.5 2.5 4.5 6.5 8.5
* dim_1    (dim_1) float64 0.5 2.5 4.5 6.5 8.5
* dim_2    (dim_2) float64 0.5 2.5 4.5 6.5 8.5,
<xarray.DataArray 'astype-675175a39bec4fea06b8668053458285' (dim_0: 2,
                                                            dim_1: 2, dim_2: 2)>
dask.array<astype, shape=(2, 2, 2), dtype=int64, chunksize=(2, 2, 2), chunktype=numpy.ndarray>
Coordinates:
* dim_0    (dim_0) float64 1.5 5.5
* dim_1    (dim_1) float64 1.5 5.5
* dim_2    (dim_2) float64 1.5 5.5]
"""
```

### Caveats

* Arrays that are not evenly divisible by the downsampling factors will be trimmed as needed. If this behavior is undesirable, consider padding your array appropriately prior to downsampling.
* For chunked arrays (e.g., dask arrays), the current implementation divides the input data into *contiguous* chunks. This means that attempting to use downsampling schemes based on sliding windowed smoothing will produce edge artifacts.

### Development

This project is developed using [`hatch`](https://hatch.pypa.io/latest/). 
Run tests with `hatch run test:pytest`.
Serve docs with `hatch run docs:serve`.
