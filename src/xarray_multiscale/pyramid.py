"""
A native representation of multiscale data.

This module provides ``Multiscale``: an ordered collection of
``xarray.Dataset`` objects ("levels") that are samplings of one signal on a
shared continuous domain, ordered fine to coarse.

The design is documented in ``docs/design/2026-08-04-native-multiscale-design.md``.
In short:

- The *structural* invariant (same dims, same variable and coordinate names,
  monotonic coordinates in a consistent direction, levels ordered fine to
  coarse) is enforced at construction.
- The *geometric* invariant (equal domain coverage under cell semantics) is
  advisory, via :meth:`Multiscale.is_consistent`, because a large fraction of
  real-world pyramids (``::2`` subsampled images, rounded COG overviews)
  violate it slightly.
"""
from __future__ import annotations

import operator
import posixpath
import warnings
from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Any, Callable

import numpy as np
import xarray as xr

from xarray_multiscale.coordinates import (
    HAS_ANALYTIC_COORDS,
    analytic_transform,
    coords_from_specs,
)
from xarray_multiscale.coordinates import from_json as _coords_from_json
from xarray_multiscale.coordinates import to_json as _coord_to_json

__all__ = ["Multiscale", "open_multiscale"]

# key under which the manifest is stored in zarr group attributes
MULTISCALES_KEY = "multiscales"
# reserved selector kwargs for Multiscale.sel
_SELECTORS = ("level", "resolution", "shape")

_OME_AXIS_TYPES = {
    "x": "space",
    "y": "space",
    "z": "space",
    "t": "time",
    "time": "time",
    "c": "channel",
    "channel": "channel",
}


def _as_dataset(obj: xr.Dataset | xr.DataArray) -> xr.Dataset:
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name if obj.name is not None else "data")
    raise TypeError(f"Expected Dataset or DataArray, got {type(obj)}")


def _transform_index(ds: xr.Dataset, dim: Hashable) -> Any:
    """
    The analytic transform backing ``dim``, or None if the coordinate is an
    array of values. See :mod:`xarray_multiscale.coordinates` for the two
    kinds and how each is stored.
    """
    return analytic_transform(ds, dim)


def _usable_coord(ds: xr.Dataset, dim: Hashable) -> bool:
    if dim not in ds.coords or ds.sizes[dim] < 1:
        return False
    coord = ds.coords[dim]
    return coord.ndim == 1 and np.issubdtype(coord.dtype, np.number)


def _probe(ds: xr.Dataset, dim: Hashable, positions: Sequence[int]) -> np.ndarray:
    """
    Coordinate values at a few integer positions, without materializing the
    whole coordinate. Positional indexing of a functional coordinate only
    evaluates the requested points, so this stays cheap for enormous
    dimensions.
    """
    return np.asarray(ds.coords[dim][list(positions)].values, dtype="float64")


def _spacings(ds: xr.Dataset) -> dict[Hashable, float]:
    """
    Mean absolute spacing of each 1-D numeric dimension coordinate with at
    least two samples. Dimensions without a usable coordinate, or of length
    < 2, are omitted: their spacing is unknowable from the data.

    For a functional coordinate the spacing is read off the transform, which
    is *exact* — the differencing of stored float values that this replaces is
    the long-standing source of scale drift when round-tripping OME-NGFF
    transforms. Otherwise it is derived from the endpoints, which telescopes
    to the mean of the differences without reading the interior.
    """
    out: dict[Hashable, float] = {}
    for dim in ds.dims:
        transform = _transform_index(ds, dim)
        if transform is not None:
            # the scale is a parameter of the function, so it is known even
            # for a single-sample axis, where differencing has nothing to work
            # with. A size-1 z axis with a real slice thickness is ordinary in
            # OME-NGFF and its scale must not be invented.
            out[dim] = abs(transform.scale)
            continue
        if not _usable_coord(ds, dim) or ds.sizes[dim] < 2:
            continue
        first, last = _probe(ds, dim, [0, -1])
        out[dim] = abs(last - first) / (ds.sizes[dim] - 1)
    return out


def _extent(ds: xr.Dataset, dim: Hashable) -> tuple[float, float] | None:
    """
    Domain coverage of ``dim`` under cell semantics: each coordinate value is
    the center of a cell whose width is the local spacing, and the extent is
    the union of all cells. Returns None if the extent is unknowable.
    """
    if not _usable_coord(ds, dim) or ds.sizes[dim] < 2:
        return None
    first, second, penultimate, last = _probe(ds, dim, [0, 1, -2, -1])
    lo = first - (second - first) / 2
    hi = last + (last - penultimate) / 2
    return (min(lo, hi), max(lo, hi))


def _direction(ds: xr.Dataset, dim: Hashable) -> int:
    """
    +1 for increasing, -1 for decreasing, 0 for unknown. Raises if the
    coordinate is not monotonic.

    A functional coordinate is monotonic by construction, so its direction is
    the sign of its step; checking every value would defeat the point of not
    materializing it.
    """
    if not _usable_coord(ds, dim) or ds.sizes[dim] < 2:
        return 0
    transform = _transform_index(ds, dim)
    if transform is not None:
        return 1 if transform.scale > 0 else -1
    diffs = np.diff(np.asarray(ds.coords[dim].values))
    if np.all(diffs > 0):
        return 1
    if np.all(diffs < 0):
        return -1
    raise ValueError(f"coordinate {dim!r} is not monotonic")


def _is_uniform(ds: xr.Dataset, dim: Hashable, rtol: float = 1e-6) -> bool:
    """
    Whether ``dim`` is sampled at a constant interval, and so can be described
    by a scale and a translation.

    A functional coordinate is uniform by construction. An explicit one has to
    be checked against its values, which is affordable here because those
    values are already stored as an array.
    """
    if _transform_index(ds, dim) is not None:
        return True
    if not _usable_coord(ds, dim) or ds.sizes[dim] < 3:
        return True
    diffs = np.diff(np.asarray(ds.coords[dim].values, dtype="float64"))
    mean = diffs.mean()
    if mean == 0:
        return bool(np.all(diffs == 0))
    return bool(np.max(np.abs(diffs - mean)) <= abs(mean) * rtol)


def _sel_dataset(
    ds: xr.Dataset,
    indexers: Mapping[Hashable, Any],
    method: str | None = None,
    tolerance: Any = None,
) -> xr.Dataset:
    """
    Label-based selection that works uniformly over functional and explicit
    coordinates.

    Xarray applies one ``method`` to a whole ``sel`` call, but the two kinds
    of coordinate want different ones: a transform-backed index resolves
    labels by inverting its transform and accepts only ``method="nearest"``,
    while a pandas index rejects ``method`` outright when the indexer is a
    slice. A dataset holding both kinds — an OME image with functional x/y/z
    and an explicit channel or time axis — therefore cannot be selected in a
    single call. Splitting the indexers by index type makes the natural
    expression work.

    ``method`` and ``tolerance`` apply to point selections on explicit
    coordinates; for interval (slice) selection they carry no meaning.
    """
    if not indexers:
        return ds

    functional = {k: v for k, v in indexers.items() if _transform_index(ds, k) is not None}
    plain = {k: v for k, v in indexers.items() if k not in functional}

    result = ds
    if plain:
        slices = {k: v for k, v in plain.items() if isinstance(v, slice)}
        points = {k: v for k, v in plain.items() if k not in slices}
        if slices:
            result = result.sel(slices)
        if points:
            kwargs: dict[str, Any] = {}
            if method is not None:
                kwargs["method"] = method
            if tolerance is not None:
                kwargs["tolerance"] = tolerance
            result = result.sel(points, **kwargs)

    if functional:
        if method not in (None, "nearest"):
            raise ValueError(
                f"coordinates {sorted(map(str, functional))} are functional "
                f"(transform-backed) and support only method='nearest', "
                f"not {method!r}"
            )
        if tolerance is not None:
            raise ValueError(
                f"coordinates {sorted(map(str, functional))} are functional "
                "(transform-backed) and do not support a tolerance"
            )
        result = result.sel(functional, method="nearest")

    return result


def _validate_levels(levels: Mapping[str, xr.Dataset]) -> None:
    """
    Enforce the structural invariant. Raises ValueError naming the offending
    level(s) on failure.
    """
    if len(levels) == 0:
        raise ValueError("a Multiscale requires at least one level")

    names = list(levels)
    first = levels[names[0]]
    ref_dims = set(first.dims)
    ref_vars = set(first.data_vars)
    ref_coords = set(first.coords)

    for name in names[1:]:
        ds = levels[name]
        if set(ds.dims) != ref_dims:
            raise ValueError(
                f"level {name!r} has dims {sorted(map(str, ds.dims))}, "
                f"expected {sorted(map(str, ref_dims))} (from level {names[0]!r})"
            )
        if set(ds.data_vars) != ref_vars:
            raise ValueError(
                f"level {name!r} has data variables {sorted(map(str, ds.data_vars))}, "
                f"expected {sorted(map(str, ref_vars))} (from level {names[0]!r})"
            )
        if set(ds.coords) != ref_coords:
            raise ValueError(
                f"level {name!r} has coordinates {sorted(map(str, ds.coords))}, "
                f"expected {sorted(map(str, ref_coords))} (from level {names[0]!r})"
            )

    # monotonicity (raises inside _direction), consistent direction, and
    # fine -> coarse ordering of spacings
    for dim in ref_dims:
        directions = {name: _direction(levels[name], dim) for name in names}
        known = {n: d for n, d in directions.items() if d != 0}
        if len(set(known.values())) > 1:
            raise ValueError(
                f"coordinate {dim!r} does not have a consistent direction "
                f"across levels: {known}"
            )

    prev_name: str | None = None
    prev_spacing: dict[Hashable, float] = {}
    for name in names:
        spacing = _spacings(levels[name])
        for dim, s in spacing.items():
            prev = prev_spacing.get(dim)
            # allow equality: not every dimension is downsampled
            if prev is not None and s < prev * (1 - 1e-6):
                raise ValueError(
                    f"levels are not ordered fine to coarse: spacing of {dim!r} "
                    f"decreases from {prev} (level {prev_name!r}) to {s} "
                    f"(level {name!r})"
                )
        prev_spacing.update(spacing)
        prev_name = name


class Multiscale(Mapping):
    """
    An ordered, named collection of ``xarray.Dataset`` levels sampling one
    signal on a shared domain, from finest to coarsest.

    ``Multiscale`` is a ``Mapping`` from level name to ``Dataset``: iteration,
    ``.keys()``, ``.values()``, ``.items()``, and ``ms[name]`` behave like a
    dict. Integer indexing (``ms[0]``, ``ms[-1]``) selects levels by position.
    """

    def __init__(
        self,
        levels: Mapping[str, xr.Dataset | xr.DataArray],
        attrs: Mapping[str, Any] | None = None,
    ):
        normalized = {str(k): _as_dataset(v) for k, v in levels.items()}
        _validate_levels(normalized)
        self._levels: dict[str, xr.Dataset] = normalized
        # pyramid-level metadata. A dialect stores what it read here (under
        # its own key) so that it can write the same thing back out; see
        # `to_zarr`.
        self.attrs: dict[str, Any] = dict(attrs or {})

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_datasets(
        cls,
        datasets: Sequence[xr.Dataset | xr.DataArray],
        names: Sequence[str] | None = None,
    ) -> Multiscale:
        """
        Assemble a Multiscale from an explicit fine-to-coarse sequence of
        datasets (or data arrays), wherever they came from.

        This is the constructor that IO backends target: any source that can
        produce a list of per-level datasets can produce a ``Multiscale``.
        """
        if names is None:
            names = [str(i) for i in range(len(datasets))]
        if len(names) != len(datasets):
            raise ValueError(
                f"got {len(datasets)} datasets but {len(names)} names"
            )
        return cls(dict(zip(names, datasets)))

    @classmethod
    def downscale(
        cls,
        array: Any,
        reduction: Callable[..., Any],
        scale_factors: Sequence[int] | int,
        **kwargs: Any,
    ) -> Multiscale:
        """
        Derivational constructor: build a pyramid by recursively downsampling
        ``array`` (an array or DataArray) with ``reduction``. Wraps
        :func:`xarray_multiscale.multiscale`; coordinates are averaged over
        windows, preserving domain coverage under cell semantics.
        """
        from xarray_multiscale.multiscale import multiscale

        arrays = multiscale(array, reduction, scale_factors, **kwargs)
        # downscaled levels inherit dask graph names ("downscale-<hash>");
        # a pyramid is one variable, so every level gets the source's name
        name = arrays[0].name if arrays[0].name is not None else "data"
        return cls.from_datasets([a.rename(name) for a in arrays])

    # ------------------------------------------------------------------
    # Mapping protocol and access
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self) -> Iterator[str]:
        return iter(self._levels)

    def __getitem__(self, key: str | int) -> xr.Dataset:
        if isinstance(key, int):
            return self._levels[self.levels[key]]
        return self._levels[key]

    @property
    def levels(self) -> tuple[str, ...]:
        """Level names, ordered fine to coarse."""
        return tuple(self._levels)

    def level(self, key: str | int) -> xr.Dataset:
        """Project a single level out of the pyramid, as a plain Dataset."""
        return self[key]

    @property
    def finest(self) -> xr.Dataset:
        return self[0]

    @property
    def coarsest(self) -> xr.Dataset:
        return self[-1]

    @property
    def scales(self) -> dict[str, dict[Hashable, float]]:
        """Mean coordinate spacing per dimension, for each level."""
        return {name: _spacings(ds) for name, ds in self.items()}

    def transform(
        self, level: str | int = 0, precision: int | None = None
    ) -> dict[str, dict[Hashable, float]]:
        """
        The affine map from array indices to world coordinates of one level,
        derived from its coordinates: ``{"scale": {dim: spacing},
        "translate": {dim: first coordinate value}}``.

        Only dimensions with a numeric 1-D coordinate of length >= 2 appear in
        ``scale``; ``translate`` includes any dimension with a coordinate.

        Where a coordinate is *functional* (declared by a transform rather
        than stored as values), the parameters are read back exactly. Where
        it is stored as explicit values, the scale is recovered by
        differencing and is subject to float drift; ``precision`` rounds both
        to compensate.
        """
        ds = self[level]
        scale = _spacings(ds)
        translate = {}
        for dim in ds.dims:
            if not _usable_coord(ds, dim):
                continue
            transform = _transform_index(ds, dim)
            translate[dim] = (
                transform.translation
                if transform is not None
                else float(_probe(ds, dim, [0])[0])
            )
        if precision is not None:
            scale = {k: round(v, precision) for k, v in scale.items()}
            translate = {k: round(v, precision) for k, v in translate.items()}
        return {"scale": scale, "translate": translate}

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def is_consistent(self, rtol: float = 1e-3) -> bool:
        """
        Advisory check of the geometric invariant: do all levels cover the
        same domain, under cell semantics? Deviations are measured relative
        to the finest level's extent per dimension.

        This is deliberately not enforced anywhere: ``::2``-subsampled
        pyramids and rounded overview shapes fail it slightly and are still
        useful data.
        """
        finest = self.finest
        for dim in finest.dims:
            ref = _extent(finest, dim)
            if ref is None:
                continue
            span = ref[1] - ref[0]
            if span == 0:
                continue
            for ds in self.values():
                ext = _extent(ds, dim)
                if ext is None:
                    continue
                if max(abs(ext[0] - ref[0]), abs(ext[1] - ref[1])) > rtol * span:
                    return False
        return True

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    def sel(
        self,
        indexers: Mapping[str, Any] | None = None,
        *,
        level: str | int | None = None,
        resolution: float | Mapping[Hashable, float] | None = None,
        shape: Mapping[Hashable, int] | None = None,
        method: str | None = None,
        tolerance: Any = None,
        **indexers_kwargs: Any,
    ) -> Multiscale | xr.Dataset:
        """
        Select by world coordinates.

        With only coordinate indexers, the selection is applied at every
        level (each level using its own coordinates) and a ``Multiscale`` is
        returned. Passing exactly one of the selector arguments projects a
        single level instead, returning a plain ``Dataset``:

        - ``level``: that level, selected.
        - ``resolution``: the coarsest level whose spacing is <= the requested
          resolution in every constrained dimension. A scalar constrains every
          dimension whose spacing varies across levels; a mapping constrains
          exactly its keys. If even the finest level is too coarse, the
          finest level is returned (best effort).
        - ``shape``: the coarsest level that still yields at least the
          requested number of samples in each given dimension *after* the
          coordinate selection is applied (the viewer/tile use case). If no
          level is large enough, the finest is returned.
        """
        given = [k for k, v in zip(_SELECTORS, (level, resolution, shape)) if v is not None]
        if len(given) > 1:
            raise ValueError(
                f"at most one of {_SELECTORS} may be given, got {given}"
            )

        indexers = dict(indexers or {}) | dict(indexers_kwargs)

        def _apply(ds: xr.Dataset) -> xr.Dataset:
            return _sel_dataset(ds, indexers, method=method, tolerance=tolerance)

        if level is not None:
            return _apply(self[level])

        if resolution is not None:
            return _apply(self[self._pick_by_resolution(resolution)])

        if shape is not None:
            selected = {name: _apply(ds) for name, ds in self.items()}
            for name in reversed(list(selected)):
                ds = selected[name]
                if all(ds.sizes.get(dim, 0) >= n for dim, n in shape.items()):
                    return ds
            return selected[self.levels[0]]

        return type(self)({name: _apply(ds) for name, ds in self.items()}, attrs=self.attrs)

    def _pick_by_resolution(self, resolution: float | Mapping[Hashable, float]) -> str:
        all_spacings = self.scales
        if isinstance(resolution, Mapping):
            constraints: dict[Hashable, float] = dict(resolution)
        else:
            # a scalar constrains every dimension that is actually multiscale
            varying = [
                dim
                for dim in self.finest.dims
                if len({s[dim] for s in all_spacings.values() if dim in s}) > 1
            ]
            constraints = {dim: float(resolution) for dim in varying}
        if not constraints:
            return self.levels[0]

        chosen = self.levels[0]
        for name in self.levels:
            spacing = all_spacings[name]
            # a level with unknowable spacing for a constrained dim is not a candidate
            if all(
                dim in spacing and spacing[dim] <= res * (1 + 1e-6)
                for dim, res in constraints.items()
            ):
                chosen = name
            else:
                break
        return chosen

    # ------------------------------------------------------------------
    # transformation
    # ------------------------------------------------------------------
    def map(self, fn: Callable[..., xr.Dataset | xr.DataArray], *args: Any, **kwargs: Any) -> Multiscale:
        """
        Apply ``fn`` to every level and re-wrap the results. The structural
        invariant is re-validated: a function that renames coordinates or
        drops dimensions inconsistently across levels is an error.

        ``map`` is the closed algebra for geometry-preserving operations. For
        per-level reductions whose results are not a multiscale variable
        (histograms, summaries), use the mapping protocol instead::

            {name: fn(ds) for name, ds in ms.items()}
        """
        results = {}
        for name, ds in self.items():
            try:
                results[name] = _as_dataset(fn(ds, *args, **kwargs))
            except Exception as e:
                raise type(e)(f"map failed on level {name!r}: {e}") from e
        try:
            return type(self)(results, attrs=self.attrs)
        except ValueError as e:
            raise ValueError(
                f"map produced levels that violate the multiscale invariant: {e}"
            ) from e

    def _binop(self, other: Any, op: Callable[[Any, Any], Any], reflexive: bool = False):
        if isinstance(other, Multiscale):
            return NotImplemented
        if reflexive:
            return self.map(lambda ds: op(other, ds))
        return self.map(lambda ds: op(ds, other))

    def __add__(self, other): return self._binop(other, operator.add)
    def __radd__(self, other): return self._binop(other, operator.add, reflexive=True)
    def __sub__(self, other): return self._binop(other, operator.sub)
    def __rsub__(self, other): return self._binop(other, operator.sub, reflexive=True)
    def __mul__(self, other): return self._binop(other, operator.mul)
    def __rmul__(self, other): return self._binop(other, operator.mul, reflexive=True)
    def __truediv__(self, other): return self._binop(other, operator.truediv)
    def __rtruediv__(self, other): return self._binop(other, operator.truediv, reflexive=True)
    def __pow__(self, other): return self._binop(other, operator.pow)
    def __neg__(self): return self.map(operator.neg)

    # ------------------------------------------------------------------
    # level management
    # ------------------------------------------------------------------
    def add_level(
        self, dataset: xr.Dataset | xr.DataArray, name: str | None = None
    ) -> Multiscale:
        """
        Return a new Multiscale with ``dataset`` inserted at the position
        implied by its spacing (the geometric view: any sampling of the
        domain is a valid level, however it was produced).
        """
        ds = _as_dataset(dataset)
        if name is None:
            taken = set(self.levels)
            i = len(self)
            while str(i) in taken:
                i += 1
            name = str(i)
        if name in self._levels:
            raise ValueError(f"level {name!r} already exists")

        def _sort_key(item: tuple[str, xr.Dataset]) -> float:
            spacing = _spacings(item[1])
            if not spacing:
                return 0.0
            return float(np.exp(np.mean(np.log(list(spacing.values())))))

        entries = list(self._levels.items()) + [(name, ds)]
        entries.sort(key=_sort_key)
        return type(self)(dict(entries), attrs=self.attrs)

    def drop_level(self, key: str | int) -> Multiscale:
        """Return a new Multiscale without the given level."""
        name = self.levels[key] if isinstance(key, int) else key
        if name not in self._levels:
            raise KeyError(name)
        if len(self) == 1:
            raise ValueError("cannot drop the only level of a Multiscale")
        return type(self)(
            {k: v for k, v in self._levels.items() if k != name}, attrs=self.attrs
        )

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = [f"<xarray_multiscale.Multiscale ({len(self)} levels, fine to coarse)>"]
        for name, ds in self.items():
            sizes = ", ".join(f"{d}: {s}" for d, s in ds.sizes.items())
            spacing = ", ".join(
                f"{d}: {s:.6g}" for d, s in _spacings(ds).items()
            )
            lines.append(f"  {name}: ({sizes})" + (f"  spacing ({spacing})" if spacing else ""))
        lines.append(f"Data variables: {', '.join(map(str, self.finest.data_vars))}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # zarr IO
    # ------------------------------------------------------------------
    def to_zarr(
        self,
        store: Any,
        group: str | None = None,
        *,
        name: str | None = None,
        dialect: str = "xarray",
        encoding: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Write the pyramid to ``store`` in one of the supported dialects.

        A dialect fully owns the layout and metadata it produces, and is
        expected to read back what it wrote: ``open_multiscale`` on the result
        returns an equivalent ``Multiscale``. Writing is therefore *one*
        dialect, not a pile of metadata flavours over a shared layout — two
        conventions describing the same pyramid can disagree, and there is no
        way to say which one is authoritative.

        ``"xarray"`` (the default) writes the native convention: each level as
        an ordinary xarray zarr dataset, plus a manifest in the group
        attributes that references the levels by path::

            {"multiscales": [{"name": ..., "levels": [{"path": ...}, ...]}]}

        Paths resolve relative to the node carrying the manifest, so a
        manifest written by other means may reference levels anywhere in the
        store; the sibling-child-group layout this produces is a default, not
        the convention. ``encoding`` is broadcast over levels (variable names
        repeat across levels).

        ``"ome-ngff"`` writes OME-NGFF 0.4, whose own ``multiscales``
        metadata is the manifest. Coordinates are not stored as arrays: OME
        declares them with ``coordinateTransformations``, which is exactly
        what a functional coordinate is, so the declaration round-trips
        through the store rather than being materialized. Requires a single
        data variable and uniformly spaced coordinates.
        """
        writer = _WRITERS.get(dialect)
        if writer is None:
            raise ValueError(
                f"unknown dialect {dialect!r}, expected one of {sorted(_WRITERS)}"
            )
        writer(self, store, group or "", name=name, encoding=encoding, **kwargs)


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
def _resolve_path(prefix: str, path: str) -> str:
    resolved = posixpath.normpath(posixpath.join(prefix, path))
    if resolved.startswith(".."):
        raise ValueError(
            f"manifest path {path!r} escapes the store (resolved to {resolved!r})"
        )
    return "" if resolved == "." else resolved


def open_multiscale(
    store: Any,
    group: str | None = None,
    *,
    name: str | None = None,
    **kwargs: Any,
) -> Multiscale:
    """
    Open a multiscale group from a zarr store.

    Recognizes, in order:

    1. The native manifest: ``multiscales`` entries with a ``levels`` list of
       ``{"path": ...}`` references, resolved relative to ``group``. Each
       referenced node must be an ordinary xarray-readable zarr group.
    2. The OME-NGFF dialect: ``multiscales`` entries with ``axes`` and
       ``datasets`` whose paths reference zarr *arrays*; coordinates are
       generated from each dataset's scale/translation transforms.

    ``name`` selects among multiple pyramids in one manifest (by their
    ``name`` field); by default the first entry is used. Extra ``kwargs``
    are forwarded to ``xarray.open_zarr`` for native-manifest levels.
    """
    import zarr

    prefix = group or ""
    node = zarr.open_group(store, path=prefix, mode="r")
    entries = node.attrs.get(MULTISCALES_KEY)
    if not entries:
        raise ValueError(
            f"no {MULTISCALES_KEY!r} metadata found"
            + (f" in group {group!r}" if group else "")
        )

    if name is not None:
        entries = [e for e in entries if e.get("name") == name]
        if not entries:
            raise ValueError(f"no multiscale named {name!r} found")

    native = [e for e in entries if "levels" in e]
    ome = [e for e in entries if "datasets" in e and "axes" in e]

    if native:
        entry = native[0]
        datasets, names = [], []
        for i, item in enumerate(entry["levels"]):
            path = _resolve_path(prefix, item["path"])
            ds = xr.open_zarr(store, group=path, **kwargs)
            if item.get("coordinates"):
                ds = ds.assign_coords(_coords_from_json(item["coordinates"]))
            datasets.append(ds)
            names.append(str(item.get("name", posixpath.basename(item["path"]) or i)))
        return Multiscale.from_datasets(datasets, names=names)

    if ome:
        return _open_ome(store, prefix, ome[0], node)

    raise ValueError(
        f"found {MULTISCALES_KEY!r} metadata, but no recognizable dialect "
        "(expected a 'levels' list or OME-NGFF 'axes' + 'datasets')"
    )


def _write_native(
    ms: Multiscale,
    store: Any,
    prefix: str,
    name: str | None = None,
    encoding: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """
    Write each level as an ordinary xarray zarr dataset, plus a manifest
    referencing them by path.

    Each coordinate takes whichever of the two serialization paths its kind
    calls for: an array coordinate is written as a zarr array inside the level
    group, an analytic one as JSON in that level's manifest entry. An analytic
    coordinate is never evaluated into an array, because recovering the
    function from samples of it is lossy.
    """
    import zarr

    levels_manifest: list[dict[str, Any]] = []
    for level_name, ds in ms.items():
        declared: dict[str, Any] = {}
        for coord in ds.coords:
            spec = _coord_to_json(ds, coord)
            if spec is not None:
                declared[str(coord)] = spec
        to_write = ds.drop_vars(list(declared)) if declared else ds

        level_encoding = None
        if encoding is not None:
            level_encoding = {
                k: v for k, v in encoding.items() if k in to_write.variables
            }
        to_write.to_zarr(
            store,
            group=posixpath.join(prefix, level_name),
            encoding=level_encoding,
            **kwargs,
        )
        entry: dict[str, Any] = {"path": level_name}
        if declared:
            entry["coordinates"] = declared
        levels_manifest.append(entry)

    manifest = {"name": name, "levels": levels_manifest}
    root = zarr.open_group(store, path=prefix, mode="a")
    root.attrs.update({MULTISCALES_KEY: [manifest]})


def _write_ome(
    ms: Multiscale,
    store: Any,
    prefix: str,
    name: str | None = None,
    encoding: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """
    Write OME-NGFF 0.4, whose ``multiscales`` metadata is its own manifest.

    Coordinates are written as ``coordinateTransformations``, not as arrays:
    that is how OME declares them, and it is the same declaration a
    functional coordinate carries, so reading the result back reproduces the
    coordinates exactly. Axis metadata that came from a store originally
    (types, units) is preserved via ``Multiscale.attrs['ome']`` so that
    reading and re-writing does not quietly drop it.
    """
    import dask.array as da
    import zarr

    data_vars = list(ms.finest.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            "the ome-ngff dialect describes a single array per level, but the "
            f"levels have data variables {data_vars}"
        )
    (var,) = data_vars
    dims = [str(d) for d in ms.finest[var].dims]

    for level_name, ds in ms.items():
        bad = [
            d
            for d in dims
            if ds.sizes[d] > 1
            and (d not in _spacings(ds) or not _is_uniform(ds, d))
        ]
        if bad:
            raise ValueError(
                f"level {level_name!r} is not uniformly spaced along {bad}; "
                "ome-ngff declares coordinates as a scale and a translation, "
                "which cannot describe irregular sampling"
            )

    source = dict(ms.attrs.get("ome") or {})
    axes = source.get("axes")
    if not axes or [str(a.get("name")) for a in axes] != dims:
        # no source metadata, or the dimensions have changed since it was read
        axes = [
            {"name": d, "type": _OME_AXIS_TYPES.get(d.lower(), "space")} for d in dims
        ]

    root = zarr.open_group(store, path=prefix, mode="a")
    datasets = []
    for level_name in ms.levels:
        array = ms[level_name][var]
        data = array.data
        chunks = getattr(data, "chunksize", None) or data.shape
        target = root.create_array(
            level_name,
            shape=data.shape,
            dtype=data.dtype,
            chunks=chunks,
            overwrite=True,
            **(dict(encoding) if encoding else {}),
        )
        if isinstance(data, da.Array):
            da.store(data, target, lock=False)
        else:
            target[:] = np.asarray(data)

        transform = ms.transform(level_name)
        datasets.append(
            {
                "path": level_name,
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [transform["scale"].get(d, 1.0) for d in dims],
                    },
                    {
                        "type": "translation",
                        "translation": [
                            transform["translate"].get(d, 0.0) for d in dims
                        ],
                    },
                ],
            }
        )

    entry = {
        **source,
        "version": source.get("version", "0.4"),
        "axes": axes,
        "datasets": datasets,
    }
    resolved_name = name if name is not None else source.get("name")
    if resolved_name is not None:
        entry["name"] = resolved_name
    root.attrs[MULTISCALES_KEY] = [entry]


_WRITERS: dict[str, Callable[..., None]] = {
    "xarray": _write_native,
    "ome-ngff": _write_ome,
}


def _open_ome(store: Any, prefix: str, entry: dict[str, Any], node: Any) -> Multiscale:
    import dask.array as da
    import zarr

    dims = [ax["name"] for ax in entry["axes"]]
    var_name = entry.get("name") or "data"
    global_tfs = entry.get("coordinateTransformations", [])

    datasets, names = [], []
    for item in entry["datasets"]:
        path = _resolve_path(prefix, item["path"])
        arr = zarr.open_array(store, path=path, mode="r")
        data = da.from_array(arr, chunks=arr.chunks)

        scale = [1.0] * len(dims)
        translation = [0.0] * len(dims)
        for tf in list(item.get("coordinateTransformations", [])) + list(global_tfs):
            if tf["type"] == "scale":
                scale = [s * g for s, g in zip(scale, tf["scale"])]
                translation = [t * g for t, g in zip(translation, tf["scale"])]
            elif tf["type"] == "translation":
                translation = [t + g for t, g in zip(translation, tf["translation"])]

        coords = coords_from_specs(
            [
                {"dim": d, "scale": sc, "translation": tr, "size": n}
                for d, sc, tr, n in zip(dims, scale, translation, data.shape)
            ]
        )
        datasets.append(
            xr.DataArray(data, dims=dims, coords=coords, name=var_name).to_dataset()
        )
        # OME dataset paths are unique by construction; basenames need not be
        # (e.g. "0/tas", "1/tas"), so the path itself is the level name
        names.append(str(item["path"]))
    result = Multiscale.from_datasets(datasets, names=names)
    # keep the source metadata so that writing ome-ngff back out reproduces
    # the axis types, units and version rather than re-deriving them
    result.attrs["ome"] = dict(entry)
    return result
