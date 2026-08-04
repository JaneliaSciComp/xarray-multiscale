"""
The three real-world multiscale workflows from the design doc
(docs/design/2026-08-04-native-multiscale-design.md), rewritten against the
one abstraction: ``Multiscale``.

Each section names the pain it replaces:

1. Web-map pyramid (Development Seed / ndpyramid): per-level fixup loops,
   hand-assembled ``multiscales`` attrs, reopening each level by group name.
2. Overview selection (DEA / rioxarray COG): GDAL loops to discover levels,
   ``overview_level=2`` magic numbers, opening the same source four times.
3. OME-Zarr (xarray-ome-ngff / spatialdata): magic string keys
   (``['scale0']['image']``), transforms stashed in attrs, hand-derived
   ``scale = coords[1] - coords[0]`` for viewer handoff.

Run:  PYTHONPATH=src python examples/workflows.py
"""
from __future__ import annotations

import tempfile

import numpy as np
import xarray as xr
import zarr

from xarray_multiscale import Multiscale, open_multiscale
from xarray_multiscale.reducers import windowed_mean

tmp = tempfile.mkdtemp()


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------
section("1. Web-map pyramid: build, transform every level, write, reopen")
# ----------------------------------------------------------------------
# a global-ish temperature field with world coordinates
tas = xr.DataArray(
    np.random.default_rng(0).random((512, 512), dtype="float32") * 40 + 250,
    dims=("y", "x"),
    coords={"y": np.linspace(-90, 90, 512), "x": np.linspace(-180, 180, 512)},
    name="tas",
)

# derivational constructor (was: pyramid_reproject + manual loops)
pyramid = Multiscale.downscale(tas, windowed_mean, 2, depth=3)
print(pyramid)

# one operation, every level, invariant re-checked
# (was: `for child in dt.children.values(): child.ds = ...`)
pyramid = (pyramid - 273.15).map(lambda ds: ds.chunk({"y": 128, "x": 128}))

# manifest written for you (was: hand-assembled nested attrs dict)
store = f"{tmp}/tas.zarr"
pyramid.to_zarr(store, name="tas", mode="w")
print("\nmanifest:", zarr.open_group(store, mode="r").attrs["multiscales"][0])

# reopen the whole pyramid (was: xr.open_zarr(..., group=N) per level)
ms = open_multiscale(store)

# "give me a 128px thumbnail" (was: know which group number to open)
thumb = ms.sel(shape={"x": 128, "y": 128})
print("\nthumbnail level shape:", dict(thumb.sizes))

# ----------------------------------------------------------------------
section("2. Overview selection: discovery, resolution-based access, ROI")
# ----------------------------------------------------------------------
# levels assembled from anywhere -- from_datasets is the backend contract,
# so a COG reader (rioxarray) can produce a Multiscale from overviews.
# Here: levels that were written independently, like real archives.
ms = open_multiscale(store)

# discovery (was: gdal.Open + GetOverviewCount + geotransform arithmetic)
print("levels and spacings:")
for name, spacing in ms.scales.items():
    print(f"  {name}: {spacing}")

# crop once, in world coordinates, at every level
# (was: crop(open(url)), crop(open(url, overview_level=2)), ...)
roi = ms.sel(x=slice(-30, 30), y=slice(-30, 30))
print("\nROI sizes per level:", {k: dict(v.sizes) for k, v in roi.items()})

# "about 1 degree per sample" (was: overview_level=2  # a medium level)
approx = roi.sel(resolution=1.0)
print("resolution<=1.0 level shape:", dict(approx.sizes))

# compare fine vs coarse: same object, consistently cropped
print(
    "finest mean %.3f | coarsest mean %.3f"
    % (float(roi.finest.tas.mean()), float(roi.coarsest.tas.mean()))
)

# per-level reduction: mapping protocol, not map (result is not multiscale)
counts = {name: int(ds.tas.count()) for name, ds in roi.items()}
print("samples per level:", counts)

# ----------------------------------------------------------------------
section("3. OME-Zarr: open by transforms, physical selection, viewer handoff")
# ----------------------------------------------------------------------
# a store carrying ONLY OME-NGFF metadata (axes + coordinateTransformations,
# no coordinate arrays), as written by bioimaging tools
img = xr.DataArray(
    np.random.default_rng(1).integers(0, 255, (64, 256, 256), dtype="uint8"),
    dims=("z", "y", "x"),
    coords={
        "z": 0.25 + 0.5 * np.arange(64),   # 0.5 um axial
        "y": 0.09 + 0.18 * np.arange(256),  # 0.18 um lateral
        "x": 0.09 + 0.18 * np.arange(256),
    },
    name="nuclei",
)
ome_store = f"{tmp}/nuclei.zarr"
Multiscale.downscale(img, windowed_mean, (1, 2, 2), depth=2).to_zarr(
    ome_store, name="nuclei", dialect="ome-ngff"
)
print("store contains only arrays:", sorted(zarr.open_group(ome_store, mode="r").array_keys()))

ms = open_multiscale(ome_store)
print(ms)

# OME declares coordinates as a function of the index (scale + translation).
# They are carried through as functional coordinates rather than evaluated:
# nothing is allocated, and the declared parameters stay exact.
from xarray.indexes import CoordinateTransformIndex  # noqa: E402

x = ms.finest.coords["x"]
print("\nx coord is functional:", isinstance(ms.finest.xindexes["x"], CoordinateTransformIndex))
print("  materialized?       ", isinstance(x.variable._data, np.ndarray))
print("  exact scale         ", ms.scales[ms.levels[0]]["x"])
print("  vs differencing     ", float(np.diff(x.values[:2])[0]), "<- the drift this avoids")

# physical selection: one z-plane in microns, at a target resolution.
# plain xarray would demand method='nearest' here; Multiscale supplies it.
plane = ms.sel(z=16.0, resolution={"x": 0.5, "y": 0.5})
print("\n~0.5um plane shape:", dict(plane.sizes))

# viewer handoff (was: scale=[float(d[1]-d[0]) for d in dims] by hand)
print("napari kwargs:", ms.transform(0))
# e.g. viewer.add_image([lvl.nuclei.data for lvl in ms.values()], ...)

# and the dialect round-trips: read OME, write OME, metadata unchanged
again = f"{tmp}/again.zarr"
ms.to_zarr(again, dialect="ome-ngff")
before = zarr.open_group(ome_store, mode="r").attrs["multiscales"]
after = zarr.open_group(again, mode="r").attrs["multiscales"]
print("OME round-trip identical:", before == after)

print("\nall three workflows: one abstraction, no per-level loops, no magic keys")
