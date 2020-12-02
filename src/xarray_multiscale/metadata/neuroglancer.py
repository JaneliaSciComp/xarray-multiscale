from typing import Sequence
from dataclasses import dataclass

from xarray.core.dataarray import DataArray
from typing import Union, List
import numpy as np
from xarray_multiscale.metadata.util import BaseMeta, SpatialTransform

@dataclass
class PixelResolution(BaseMeta):
    # fortran-ordered
    dimensions: Sequence[float]
    unit: Union[str, None]


@dataclass
class GroupMeta(BaseMeta):
    # see https://github.com/google/neuroglancer/issues/176#issuecomment-553027775
    # these properties must be stored in the opposite order of C-contiguous axis indexing 
    axes: Sequence[Union[str, None]]
    units: Sequence[Union[str, None]]
    scales: Sequence[Sequence[int]]
    pixelResolution: PixelResolution

    @classmethod
    def fromDataArraySequence(cls, dataarrays: Sequence[DataArray]) -> "GroupMeta":
        transforms = [SpatialTransform.fromDataArray(array, reverse_axes=True) for array in dataarrays]
        pixelresolution = PixelResolution(transforms[0].scale, transforms[0].units[0])
        scales: List[List[int]] = [np.divide(t.scale, transforms[0].scale).astype('int').tolist() for t in transforms]
        return cls(transforms[0].axes, transforms[0].units, scales, pixelresolution)
        


