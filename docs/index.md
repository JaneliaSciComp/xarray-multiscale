# xarray-multiscale

Simple tools for creating multiscale representations of large images.

## Installation

`pip install xarray-multiscale`

## Motivation

Many image processing applications benefit from representing images at multiple scales (also known as [image pyramids](https://en.wikipedia.org/wiki/Pyramid_(image_processing)). This package provides tools for generating lazy multiscale representations of N-dimensional data using [`xarray`](http://xarray.pydata.org/en/stable/) to ensure that the downsampled images have the correct coordinates.

### Coordinates matter when you downsample images

It's obvious that downsampling an image applies a scaling transformation, i.e. downsampling increases the distance between image samples. This is the whole purpose of downsampling the image. But it is less obvious that most downsampling operations also apply a *translation transformation* -- downsampling an image (generally) shifts the origin of the output relative to the source. 

In signal processing terms, image downsampling combines an image filtering step (blending local intensities) with a resampling step (sampling intensities at a set of positions in the signal). When you resample an image, you get to choose which points to resample on, and the best choice for most simple downsampling routines is to resample on points that are slightly translated relative to the original image. For simple windowed downsampling, this means that the first element of the downsampled image lies 
at the center (i.e., the mean) of the coordinate values of the window. 

We can illustrate this with some simple examples:

```
2x windowed downsampling, in one dimension: 

source coordinates:        | 0 | 1 | 
downsampled coordinates:   |  0.5  | 
```

```
3x windowed downsampling, in two dimensions: 

source coordinates:        | (0,0) | (0,1) | (0,2) |
                           | (1,0) | (1,1) | (1,2) |
                           | (2,0) | (2,1) | (2,2) |

downsampled coordinates:   |                       |        
                           |         (1,1)         |
                           |                       | 

```

Another way of thinking about this is that if you downsample an arbitrarily large image to a single value, then the only sensible place to localize that value is at the center of the image. Thus, incrementally downsampling slightly shifts the downsampled image toward that point.

Why should you care? If you work with images where the coordinates matter (for example, images recorded from scientific instruments), then you should care about keeping track of those coordinates. Tools like numpy or scikit-image make it very easy to ignore the coordinates of your image. These tools model images as simple arrays, and from the array perspective `data[0,0]` and `downsampled_data[0,0]` lie on the same position in space because they take the same array index. However, `downsampled_data[0,0]` is almost certainly shifted relative to `data[0,0]`. Coordinate-blind tools like `scikit-image` force your to track the coordinates on your own, which is a recipe for mistakes. This is the value of `xarray`. By explicitly modelling coordinates alongside data values, `xarray` ensures that you never lose track of where your data comes from, which is why `xarray-multiscale` uses it.

### Who needs this

The library `xarray` already supports basic downsampling routines via the [`DataArray.coarsen`](https://docs.xarray.dev/en/stable/user-guide/computation.html#coarsen-large-arrays) API. So if you use `xarray` and just need to compute a windowed mean, then you may not need `xarray-multiscale` at all. But the `DataArray.coarsen` API does not 
allow users to provide their own downsampling functions; If you need something like [windowed mode](./api/reducers.md#xarray_multiscale.reducers.windowed_mode) downsampling, or something you wrote yourself, then `xarray-multiscale` should be useful to you. 


## Usage

Generate a multiscale representation of a numpy array:

```python
from xarray_multiscale import multiscale, windowed_mean
import numpy as np

data = np.arange(4)
print(*multiscale(data, windowed_mean, 2), sep='\n')
"""
<xarray.DataArray 's0' (dim_0: 4)> Size: 32B
array([0, 1, 2, 3])
Coordinates:
  * dim_0    (dim_0) float64 32B 0.0 1.0 2.0 3.0
 
<xarray.DataArray 's1' (dim_0: 2)> Size: 16B
array([0, 2])
Coordinates:
  * dim_0    (dim_0) float64 16B 0.5 2.5
"""
```


By default, the values of the downsampled arrays are cast to the same data type as the input. This behavior can be changed with the ``preserve_dtype`` keyword argument to ``multiscale``:

```python
from xarray_multiscale import multiscale, windowed_mean
import numpy as np

data = np.arange(4)
print(*multiscale(data, windowed_mean, 2, preserve_dtype=False), sep="\n")
"""
<xarray.DataArray 's0' (dim_0: 4)> Size: 32B
array([0, 1, 2, 3])
Coordinates:
  * dim_0    (dim_0) float64 32B 0.0 1.0 2.0 3.0
 
<xarray.DataArray 's1' (dim_0: 2)> Size: 16B
array([0.5, 2.5])
Coordinates:
  * dim_0    (dim_0) float64 16B 0.5 2.5
"""
```

Anisotropic downsampling is supported:

```python
from xarray_multiscale import multiscale, windowed_mean
import numpy as np

data = np.arange(16).reshape((4,4))
print(*multiscale(data, windowed_mean, (1,2)), sep="\n")
"""
<xarray.DataArray 's0' (dim_0: 4, dim_1: 4)> Size: 128B
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
Coordinates:
  * dim_0    (dim_0) float64 32B 0.0 1.0 2.0 3.0
  * dim_1    (dim_1) float64 32B 0.0 1.0 2.0 3.0
 
<xarray.DataArray 's1' (dim_0: 4, dim_1: 2)> Size: 64B
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10],
       [12, 14]])
Coordinates:
  * dim_0    (dim_0) float64 32B 0.0 1.0 2.0 3.0
  * dim_1    (dim_1) float64 16B 0.5 2.5
"""
```


Note that `multiscale` returns an `xarray.DataArray`. 
The `multiscale` function also accepts `DataArray` objects:

```python
from xarray_multiscale import multiscale, windowed_mean
from xarray import DataArray
import numpy as np

data = np.arange(16).reshape((4,4))
coords = (DataArray(np.arange(data.shape[0]), dims=('y',), attrs={'units' : 'm'}),
            DataArray(np.arange(data.shape[0]), dims=('x',), attrs={'units' : 'm'}))

arr = DataArray(data, coords)
print(*multiscale(arr, windowed_mean, (2,2)), sep="\n")
"""
<xarray.DataArray 's0' (y: 4, x: 4)> Size: 128B
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
Coordinates:
  * y        (y) int64 32B 0 1 2 3
  * x        (x) int64 32B 0 1 2 3

<xarray.DataArray 's1' (y: 2, x: 2)> Size: 32B
array([[ 2,  4],
       [10, 12]])
Coordinates:
  * y        (y) float64 16B 0.5 2.5
  * x        (x) float64 16B 0.5 2.5
"""
```

Dask arrays work too. Note the control over output chunks via the ``chunks`` keyword argument.

```python
from xarray_multiscale import multiscale, windowed_mean
import dask.array as da

arr = da.random.randint(0, 255, (10,10,10))
print(*multiscale(arr, windowed_mean, 2, chunks=2), sep="\n")
"""
<xarray.DataArray 's0' (dim_0: 10, dim_1: 10, dim_2: 10)> Size: 8kB
dask.array<rechunk-merge, shape=(10, 10, 10), dtype=int64, chunksize=(2, 2, 2), chunktype=numpy.ndarray>
Coordinates:
  * dim_0    (dim_0) float64 80B 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
  * dim_1    (dim_1) float64 80B 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
  * dim_2    (dim_2) float64 80B 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
  
<xarray.DataArray 's1' (dim_0: 5, dim_1: 5, dim_2: 5)> Size: 1kB
dask.array<rechunk-merge, shape=(5, 5, 5), dtype=int64, chunksize=(2, 2, 2), chunktype=numpy.ndarray>
Coordinates:
  * dim_0    (dim_0) float64 40B 0.5 2.5 4.5 6.5 8.5
  * dim_1    (dim_1) float64 40B 0.5 2.5 4.5 6.5 8.5
  * dim_2    (dim_2) float64 40B 0.5 2.5 4.5 6.5 8.5

<xarray.DataArray 's2' (dim_0: 2, dim_1: 2, dim_2: 2)> Size: 64B
dask.array<astype, shape=(2, 2, 2), dtype=int64, chunksize=(2, 2, 2), chunktype=numpy.ndarray>
Coordinates:
  * dim_0    (dim_0) float64 16B 1.5 5.5
  * dim_1    (dim_1) float64 16B 1.5 5.5
  * dim_2    (dim_2) float64 16B 1.5 5.5
"""
```

### Caveats

- Arrays that are not evenly divisible by the downsampling factors will be trimmed as needed. If this behavior is undesirable, consider padding your array appropriately prior to downsampling.
- For chunked arrays (e.g., dask arrays), the current implementation divides the input data into *contiguous* chunks. This means that attempting to use downsampling schemes based on sliding windowed smoothing will produce edge artifacts.
- `multiscale` generates a sequence of arrays of descending size, where the smallest array is the last 

### Development

This project is developed using [`hatch`](https://hatch.pypa.io/latest/). 
Run tests with `hatch run test:pytest`.
Serve docs with `hatch run docs:serve`.
