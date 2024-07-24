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

read more in the [project documentation](https://JaneliaSciComp.github.io/xarray-multiscale/).
