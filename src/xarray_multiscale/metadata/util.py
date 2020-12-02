from dataclasses import dataclass, asdict
from typing import Sequence, Union, Dict, Any, Optional
from xarray import DataArray
from dacite import from_dict

def infer_c_or_f_contiguous(array: Any) -> str:
        data_order = 'C'
        if hasattr(array, 'order'):
            data_order = array.order
        elif hasattr(array, 'flags'):
            if array.flags['F_CONTIGUOUS'] and not array.flags['C_CONTIGUOUS']:
                data_order = 'F'
        return data_order

@dataclass
class BaseMeta:    
    def asdict(self):
        return asdict(self)

@dataclass
class SpatialTransform(BaseMeta):
    axes: Sequence[str]
    units: Sequence[Union[str, None]]
    translate: Sequence[float]
    scale: Sequence[float]

    def __post_init__(self):
        assert len(self.axes) == len(self.units) == len(self.translate) == len(self.scale)

    @classmethod
    def fromDataArray(cls, dataarray: DataArray, reverse_axes=False) -> "SpatialTransform":
        """
        Generate a spatial transform from a DataArray. 
        """
        
        orderer = slice(None)
        if reverse_axes:
            orderer = slice(-1, None, -1)
        axes = [str(d) for d in dataarray.dims[orderer]]
        units = [dataarray.coords[ax].attrs.get('units') for ax in axes]
        translate = [float(dataarray.coords[ax][0]) for ax in axes]
        scale = [abs(float(dataarray.coords[ax][1]) - float(dataarray.coords[ax][0])) for ax in axes]

        return cls(axes=axes, units=units, translate=translate, scale=scale)

    @classmethod
    def fromDict(cls, d: Dict[str, Any]):
        return from_dict(cls, d)
