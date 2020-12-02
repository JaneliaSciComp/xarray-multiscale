import dask.array as da
import numpy as np
from xarray import DataArray
from xarray_multiscale.metadata.neuroglancer import GroupMeta
from xarray_multiscale.metadata.util import SpatialTransform, infer_c_or_f_contiguous
from xarray_multiscale.metadata import neuroglancer, cosem_ome
import zarr
from xarray_multiscale.multiscale import multiscale

def test_array_order_inferece():
    data= np.arange(10)
    assert infer_c_or_f_contiguous(data) == 'C'

    data = np.arange(10).reshape((2,5))
    assert infer_c_or_f_contiguous(data) == 'C'

    data = np.arange(10).reshape((2,5), order='F')
    assert infer_c_or_f_contiguous(data) == 'F'

    data = zarr.zeros((10,10), order='C')
    assert infer_c_or_f_contiguous(data) == 'C'

    data = zarr.zeros((10,10), order='F')
    assert infer_c_or_f_contiguous(data) == 'F'

    data = da.zeros((10,10))
    assert infer_c_or_f_contiguous(data) == 'C'

def test_SpatialTransform():
    data = DataArray(np.zeros((10,10,10)))
    transform = SpatialTransform.fromDataArray(data)
    assert transform == SpatialTransform(axes=['dim_0','dim_1','dim_2'],
                                         units=[None] * 3, 
                                         translate=[0.0] * 3, 
                                         scale=[1.0] * 3)

    coords = [DataArray(np.arange(10), dims=('z'), attrs={'units': 'nm'}), 
          DataArray(np.arange(10) + 5, dims=('y',), attrs={'units': 'm'}), 
          DataArray(10 + (np.arange(10) * 10), dims=('x',), attrs={'units': 'km'})]

    data = DataArray(np.zeros((10,10,10)), coords=coords)
    transform = SpatialTransform.fromDataArray(data)
    assert transform == SpatialTransform(axes=['z','y','x'],
                                         units=['nm','m','km'], 
                                         translate=[0.0, 5.0, 10.0], 
                                         scale=[1.0, 1.0, 10.0 ] )

    transform = SpatialTransform.fromDataArray(data, reverse_axes=True)
    assert transform == SpatialTransform(axes=['x','y','z'],
                                         units=['km','m','nm'], 
                                         translate=[10.0, 5.0, 0.0], 
                                         scale=[10.0, 1.0, 1.0 ] )

def test_neuroglancer_metadata():
    coords = [DataArray(np.arange(16), dims=('z'), attrs={'units': 'nm'}), 
          DataArray(np.arange(16) + 5, dims=('y',), attrs={'units': 'm'}), 
          DataArray(10 + (np.arange(16) * 10), dims=('x',), attrs={'units': 'km'})]

    data = DataArray(np.zeros((16, 16, 16)), coords=coords)
    multi = multiscale(data, np.mean, (2,2,2))[:2]
    neuroglancer_metadata = neuroglancer.GroupMeta.fromDataArraySequence(multi)
    assert neuroglancer_metadata == neuroglancer.GroupMeta(axes=['x','y','z'],
                                                            units=['km','m','nm'],
                                                            scales=[[1,1,1], [2,2,2]],
                                                            pixelResolution=neuroglancer.PixelResolution(dimensions=[10.0, 1.0, 1.0], unit='km'))

def test_cosem_ome():
    coords = [DataArray(np.arange(16), dims=('z'), attrs={'units': 'nm'}), 
          DataArray(np.arange(16) + 5, dims=('y',), attrs={'units': 'm'}), 
          DataArray(10 + (np.arange(16) * 10), dims=('x',), attrs={'units': 'km'})]

    data = DataArray(np.zeros((16, 16, 16)), coords=coords, name='data')
    multi = multiscale(data, np.mean, (2,2,2))[:2]
    paths = ['s0', 's1']
    cosem_ome_group_metadata = cosem_ome.GroupMeta.fromDataArraySequence(multi, paths=paths)
    scale_metas = [cosem_ome.ScaleMeta(path = p, transform=SpatialTransform.fromDataArray(m)) for p,m in zip(paths, multi)]
    assert cosem_ome_group_metadata == cosem_ome.GroupMeta(name='data', multiscales=[cosem_ome.MultiscaleMeta(datasets=scale_metas)])