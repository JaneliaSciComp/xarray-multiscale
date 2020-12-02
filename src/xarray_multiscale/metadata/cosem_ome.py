from dataclasses import dataclass, asdict
from typing import Sequence
from xarray import DataArray
from .util import BaseMeta, SpatialTransform
from typing import Optional

@dataclass
class ScaleMeta(BaseMeta):
    path: str 
    transform: SpatialTransform


@dataclass
class MultiscaleMeta(BaseMeta):
    datasets: Sequence[ScaleMeta]


@dataclass
class GroupMeta(BaseMeta):
    name: str
    multiscales: Sequence[MultiscaleMeta]

    @classmethod
    def fromDataArraySequence(cls, dataarrays: Sequence[DataArray], paths: Sequence[str]):
        name: str = str(dataarrays[0].name)
        multiscales = [MultiscaleMeta(datasets=[ScaleMeta(path=path, transform=SpatialTransform.fromDataArray(arr)) for path, arr in zip(paths, dataarrays)])]
        return cls(name=name, multiscales=multiscales)


@dataclass
class ArrayMeta(BaseMeta):
    name: Optional[str]
    transform: SpatialTransform

    @classmethod
    def fromDataArray(cls, data: DataArray) -> "ArrayMeta":
        return cls(name=str(data.name), transform=SpatialTransform.fromDataArray(data))
        