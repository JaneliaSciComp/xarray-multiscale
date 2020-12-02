*****************
xarray-multiscale
*****************

Simple tools for creating multiscale representations of large images.

Motivation
**********
tbd

Usage
*****

Generate a lazy multiscale representation of a numpy array:

.. code-block:: python

    from xarray_multiscale import multiscale
    import numpy as np

    data = np.arange(4)
    multiscale(data, np.mean, (2,))

which returns this (a collection of DataArrays, each with decreasing size): 

.. code-block:: python

    [<xarray.DataArray 'array-3fddd342b603c7121f36e43be77be0cf' (dim_0: 4)>
dask.array<array, shape=(4,), dtype=int64, chunksize=(4,), chunktype=numpy.ndarray>
Coordinates:
  * dim_0    (dim_0) float32 0.0 1.0 2.0 3.0, <xarray.DataArray 'array-3fddd342b603c7121f36e43be77be0cf' (dim_0: 2)>
dask.array<astype, shape=(2,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>
Coordinates:
  * dim_0    (dim_0) float64 0.5 2.5, <xarray.DataArray 'array-3fddd342b603c7121f36e43be77be0cf' (dim_0: 1)>
dask.array<astype, shape=(1,), dtype=int64, chunksize=(1,), chunktype=numpy.ndarray>
Coordinates:
  * dim_0    (dim_0) float64 1.5]


Generate a lazy multiscale representation of an ``xarray.DataArray``:

.. code-block:: python

    from xarray_multiscale import multiscale
    import numpy as np
    from xarray import DataArray

    data = np.arange(16).reshape((4,4))
    coords = (DataArray(np.arange(data.shape[0]), dims=('y',), attrs={'units' : 'm'}),
              DataArray(np.arange(data.shape[0]), dims=('x',), attrs={'units' : 'm'}))

    dataarray = DataArray(data, coords)
    multiscale(dataarray, np.mean, (2,2))

which returns this:

.. code-block:: python

    [<xarray.DataArray 'array-8220a22a04dfac25908da40f77214fe3' (y: 4, x: 4)>
    dask.array<array, shape=(4, 4), dtype=int64, chunksize=(4, 4), chunktype=numpy.ndarray>
    Coordinates:
    * y        (y) int64 0 1 2 3
    * x        (x) int64 0 1 2 3, <xarray.DataArray 'array-8220a22a04dfac25908da40f77214fe3' (y: 2, x: 2)>
    dask.array<astype, shape=(2, 2), dtype=int64, chunksize=(2, 2), chunktype=numpy.ndarray>
    Coordinates:
    * y        (y) float64 0.5 2.5
    * x        (x) float64 0.5 2.5, <xarray.DataArray 'array-8220a22a04dfac25908da40f77214fe3' (y: 1, x: 1)>
    dask.array<astype, shape=(1, 1), dtype=int64, chunksize=(1, 1), chunktype=numpy.ndarray>
    Coordinates:
    * y        (y) float64 1.5
    * x        (x) float64 1.5]

``xarray_multiscale`` contains functionality for generating metadata required for the visualization tool neuroglancer (demo tbd) 

Caveats / alternatives
**********************
tbd
