from collections.abc import MutableMapping
from xarray import DataArray
from typing import Any, Dict


class MultiscaleGroup(MutableMapping[str, DataArray]):
    def __init__(self, data: Dict[str, DataArray], attrs: Dict[str, Any]) -> None:
        self._data = data
        self._attrs = attrs

    def __getitem__(self, k: _KT) -> _VT_co:
        return super().__getitem__(k)

    def __iter__(self) -> Iterator[_T_co]:
        return super().__iter__()

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __len__(self) -> int:
        return super().__len__()

    def __repr__(self) -> str:
        return super().__repr__()

    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        self.__init__(*state)

    def __contains__(self, o: object) -> bool:
        return super().__contains__(o)
