"""
Coordinate representation and serialization.

A coordinate is one of exactly two kinds, and the kind determines how it is
stored:

- **array** — a sequence of values, serialized as a zarr array. Any sampling
  can be written this way, including irregular ones.
- **analytic** — a function of the array index, serialized as JSON. Only the
  parameters of the function are stored; there are no values to write.

There are no other paths. In particular, an analytic coordinate cannot be
recovered from an array of its values: going parameters -> values -> parameters
means re-deriving the function from samples of it, and that derivation is
lossy. So the parameters must stay the authoritative record from end to end,
which is what :class:`Affine` provides in memory and what :func:`to_json`
writes out.

OME-NGFF's ``coordinateTransformations`` are the analytic path: a scale and a
translation per axis, in JSON, with no coordinate arrays anywhere in the store.
:func:`to_json` is deliberately information-equivalent to it.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

try:  # functional coordinates: xarray >= 2025.03, experimental
    from xarray.indexes import CoordinateTransform, CoordinateTransformIndex, RangeIndex

    HAS_ANALYTIC_COORDS = True
except ImportError:  # pragma: no cover - depends on the installed xarray
    CoordinateTransform = object  # type: ignore[assignment,misc]
    CoordinateTransformIndex = ()  # type: ignore[assignment]
    RangeIndex = None  # type: ignore[assignment]
    HAS_ANALYTIC_COORDS = False

__all__ = [
    "Affine",
    "HAS_ANALYTIC_COORDS",
    "analytic_transform",
    "coords_from_specs",
    "from_json",
    "is_analytic",
    "to_json",
]

TRANSFORM_KEY = "transform"
AFFINE = "affine"


class Affine(CoordinateTransform):  # type: ignore[misc]
    """
    The coordinate ``x(i) = translation + scale * i``.

    ``scale`` and ``translation`` are *stored*, never recovered from generated
    values. This is the whole point of the class: xarray's own
    ``RangeCoordinateTransform`` is parameterized by ``(start, stop, size)``
    and derives ``step`` as ``(stop - start) / size``, so it cannot hold an
    arbitrary ``(start, step)`` pair exactly — for roughly 8% of random
    parameters the recovered step differs from the requested one in the last
    ulp, which is enough to break a byte-exact round trip.

    The ``start`` / ``stop`` / ``step`` / ``size`` / ``slice`` surface is what
    :class:`xarray.indexes.RangeIndex` duck-types on, so instances can be
    handed to that index and inherit its slice-aware selection.
    """

    __slots__ = ()

    def __init__(
        self,
        scale: float,
        translation: float,
        size: int,
        coord_name: Hashable,
        dim: str,
        dtype: Any = None,
    ):
        if scale == 0:
            raise ValueError(
                f"an affine coordinate needs a non-zero scale, got {scale!r} "
                f"for {coord_name!r}"
            )
        super().__init__(
            [coord_name], {dim: int(size)}, dtype=dtype or np.dtype("float64")
        )
        self.scale = float(scale)
        self.translation = float(translation)

    @property
    def coord_name(self) -> Hashable:
        return self.coord_names[0]

    @property
    def dim(self) -> str:
        return self.dims[0]

    @property
    def size(self) -> int:
        return self.dim_size[self.dim]

    # the names RangeIndex expects
    @property
    def step(self) -> float:
        return self.scale

    @property
    def start(self) -> float:
        return self.translation

    @property
    def stop(self) -> float:
        return self.translation + self.size * self.scale

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        return {
            self.coord_name: self.translation + dim_positions[self.dim] * self.scale
        }

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        labels = coord_labels[self.coord_name]
        return {self.dim: (labels - self.translation) / self.scale}

    def equals(self, other: Any, exclude: frozenset[Hashable] | None = None) -> bool:
        return (
            isinstance(other, Affine)
            and self.scale == other.scale
            and self.translation == other.translation
            and self.size == other.size
        )

    def slice(self, sl: slice) -> Affine:
        """
        The transform of a sliced coordinate, still exact: a stride of ``k``
        multiplies the scale by ``k`` rather than re-deriving it from the new
        endpoints.
        """
        selected = range(self.size)[sl]
        return type(self)(
            self.scale * selected.step,
            self.translation + selected.start * self.scale,
            len(selected),
            self.coord_name,
            self.dim,
            dtype=self.dtype,
        )

    def __repr__(self) -> str:
        return (
            f"Affine(scale={self.scale!r}, translation={self.translation!r}, "
            f"size={self.size}, dim={self.dim!r})"
        )


def _index(spec: Mapping[str, Any]) -> Any:
    transform = Affine(
        spec["scale"],
        spec["translation"],
        spec["size"],
        spec.get("coord_name", spec["dim"]),
        spec["dim"],
    )
    return RangeIndex(transform)


def coords_from_specs(specs: Sequence[Mapping[str, Any]]) -> xr.Coordinates:
    """
    Build coordinates from analytic specifications.

    Each spec is ``{"scale": float, "translation": float, "size": int,
    "dim": str, "coord_name": optional}``. A spec with ``scale == 0`` or
    ``size < 1`` cannot describe a function of the index and is materialized
    instead.

    Note that neither ``Coordinates.merge`` nor item assignment preserves an
    index — both silently yield coordinates that look right and are no longer
    analytic — so variables and indexes are collected and handed to the
    ``Coordinates`` constructor together.
    """
    variables: dict[Hashable, Any] = {}
    indexes: dict[Hashable, Any] = {}
    for spec in specs:
        dim = spec["dim"]
        name = spec.get("coord_name", dim)
        scale, translation, size = spec["scale"], spec["translation"], spec["size"]
        if not HAS_ANALYTIC_COORDS or scale == 0 or size < 1:
            variables[name] = xr.Variable(
                (dim,), translation + scale * np.arange(size, dtype="float64")
            )
            continue
        coord = xr.Coordinates.from_xindex(_index(spec))
        variables.update(coord.variables)
        indexes.update(coord.xindexes)
    return xr.Coordinates(coords=variables, indexes=indexes)


def analytic_transform(ds: xr.Dataset, name: Hashable) -> Affine | None:
    """The :class:`Affine` backing ``name``, or None if it is an array coordinate."""
    if not HAS_ANALYTIC_COORDS:
        return None
    index = ds.xindexes.get(name)
    if not isinstance(index, CoordinateTransformIndex):
        return None
    transform = getattr(index, "transform", None)
    return transform if isinstance(transform, Affine) else None


def is_analytic(ds: xr.Dataset, name: Hashable) -> bool:
    """Whether ``name`` is stored as a function of the index rather than values."""
    return analytic_transform(ds, name) is not None


def to_json(ds: xr.Dataset, name: Hashable) -> dict[str, Any] | None:
    """
    The JSON serialization of an analytic coordinate, or None if it is an
    array coordinate and must be written as a zarr array instead.

    The parameters are written verbatim, which is what makes the round trip
    exact; they are information-equivalent to an OME-NGFF scale plus
    translation.
    """
    transform = analytic_transform(ds, name)
    if transform is None:
        return None
    return {
        TRANSFORM_KEY: AFFINE,
        "scale": transform.scale,
        "translation": transform.translation,
        "size": transform.size,
        "dim": str(transform.dim),
    }


def from_json(declarations: Mapping[str, Mapping[str, Any]]) -> xr.Coordinates:
    """Rebuild analytic coordinates from their JSON serialization."""
    specs = []
    for name, spec in declarations.items():
        kind = spec.get(TRANSFORM_KEY, AFFINE)
        if kind != AFFINE:
            raise ValueError(
                f"coordinate {name!r} declares an unknown transform {kind!r}; "
                f"this version understands {AFFINE!r}"
            )
        missing = {"scale", "translation", "size"} - set(spec)
        if missing:
            raise ValueError(
                f"coordinate {name!r} is missing {sorted(missing)} from its "
                "transform declaration"
            )
        specs.append({**spec, "dim": spec.get("dim", name), "coord_name": name})
    return coords_from_specs(specs)
