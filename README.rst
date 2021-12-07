*****************
xarray-multiscale
*****************

Simple tools for creating multiscale representations of large images.

Installation
************
.. code-block:: bash

    pip install xarray-multiscale

Motivation
**********
Many image processing applications benefit from representing images at multiple scales (also known as `image pyramids <https://en.wikipedia.org/wiki/Pyramid_(image_processing)>`_). This package provides tools for generating lazy multiscale representations of N-dimensional data using `dask <https://dask.org/>`_ and `xarray <http://xarray.pydata.org/en/stable/>`_. Dask is used to create a lazy representation of the image downscaling process, and xarray is used to ensure that the downscaled images have the correct axis coordinates.

Implementation
**************
The top-level function `multiscale` takes two main arguments: data to be downscaled, and a reduction function. The reduction function can use any implementation but it should (eagerly) take array data and a tuple of scale factors as inputs and return downscaled data as an output. See examples of reduction functions in `xarray_multiscale.reducers <https://github.com/JaneliaSciComp/xarray-multiscale/blob/main/src/xarray_multiscale/reducers.py>`_.

Note that the current implementation divides the input data into *contiguous* chunks. This means that attempting to use downscaling schemes based on sliding windowed smoothing will produce edge artifacts. Future versions of this package could enable applying the reduction function to *overlapping* chunks, which would enable more elaborate downscaling routines.


Usage
*****

Generate a lazy multiscale representation of a numpy array:

.. code-block:: python

    from xarray_multiscale import multiscale
    from xarray_multiscale.reducers import windowed_mean

    data = np.arange(4)
    multiscale(data, windowed_mean, (2,))

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
    from xarray_multiscale.reducers import windowed_mean
    from xarray import DataArray

    data = np.arange(16).reshape((4,4))
    coords = (DataArray(np.arange(data.shape[0]), dims=('y',), attrs={'units' : 'm'}),
              DataArray(np.arange(data.shape[0]), dims=('x',), attrs={'units' : 'm'}))

    dataarray = DataArray(data, coords)
    multiscale(dataarray, windowed_mean, (2,2))

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

Development
***********

This project is devloped using `poetry <https://python-poetry.org/>`_. After cloning this repo locally, run :code:`poetry install` to install local dependencies. For development within a conda environment, create a conda environment with :code:`poetry`, then install dependencies, e.g. :code:`conda create -n xarray-multiscale poetry -c conda-forge`, then run :code:`poetry install` in the root directory of this repo to install dependencies.

Tests are rudimentary and use :code:`pytest`.


Caveats / alternatives
**********************
tbd
