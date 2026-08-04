# A native multiscale representation for xarray

*Design draft, 2026-08-04. Status: exploratory — semantics and API converged in
discussion; nothing here is implemented.*

## Problem

A multiscale signal (image pyramid, COG overviews, downsampled time series) is
one logical variable sampled at several resolutions. The coordinate variables
are themselves multiscale: every level has a `time` (or `x`, `y`) coordinate
sampling the same physical domain at a different density.

xarray cannot express this today. A single `Dataset` forbids two variables
named `time` with different sizes, forcing name mangling (`time_0`, `time_1`).
`DataTree` fixes the namespace collision — each level lives in its own node —
but models the levels as unrelated datasets that happen to share a tree:
nothing states that `scale1/time` is a coarsening of `scale0/time` over the
same domain, there is no cross-level selection or mapping API, and coordinate
inheritance actively prevents nesting levels (children must align with
parents).

The ecosystem has repeatedly worked around this, incompatibly:

- **ndpyramid / carbonplan maps** (climate, web mapping): `DataTree` with
  stringified-integer children, hand-assembled `multiscales` attrs, per-level
  fixup loops.
- **rioxarray / COG overviews** (EO): one `open_rasterio(url,
  overview_level=N)` call per level; discovery of available levels requires
  dropping to GDAL; levels of the same raster are unrelated `DataArray`s.
- **OME-Zarr** (bioimaging): three active libraries return three different
  types for the same store (`dict[str, DataArray]` from xarray-ome-ngff,
  `DataTree` with `["scale0"]["image"]` paths from spatialdata /
  multiscale-spatial-image, `Dataset`-behind-a-kwarg from xarray-ome).

Every community independently reinvented: level naming conventions, manual
level-to-resolution arithmetic, per-level loops for whole-pyramid operations,
and fragile coordinate-to-affine conversion (`scale = coords[1] - coords[0]`).
These are all symptoms of the same missing object.

## Semantics

### The geometric (sampling) view

**A multiscale variable is an ordered set of samplings of one signal on a
shared continuous domain.** The relationship between levels is geometric: each
level's coordinates map into the same world domain with compatible coverage.

We explicitly do *not* define a multiscale variable by derivation ("level n+1
= downsample(level n)"). The derivational relationship is unenforceable after
any edit, and it excludes real data: multi-resolution acquisitions (the levels
were never computed from each other), pyramids built by external tools, or
levels regenerated with different methods. Derivation appears only as
*optional provenance metadata* and as a *constructor* (`coarsen`-based), never
as the invariant.

### Cell semantics and coordinate relationships

Point-sample thinking gives the wrong coordinate relationships. If level 1's
`time` is `time[::2]` of level 0, the levels do not cover the same domain:
each coarse sample represents wider support, so the coarse coverage is shifted
by half a fine cell and truncated. The correct formalization is **cell
semantics**: each coordinate value is the center of a cell whose width is the
local spacing, and two levels sample the same domain iff the union of their
cells covers the same interval, per dimension.

Consequences:

- Mean-downsampling coordinates (`coarse[i] = mean(fine[2i:2i+2])`) preserves
  coverage exactly.
- Subsample-downsampling (`fine[::2]`) does not; it implies a compensating
  half-cell translation (this is precisely why OME-NGFF grew per-level
  `translation` transforms).
- Nonuniform coordinates work under the same rule: a coarse cell is the union
  of its fine cells, its center the (weighted) mean.

### The invariant, and how strictly it is enforced

Two tiers:

1. **Structural invariant — enforced at construction.** All levels have the
   same dimension names, the same coordinate names, monotonic coordinates in
   the same direction, and levels are ordered fine to coarse. Cheap,
   unambiguous, checkable without tolerances.
2. **Geometric coverage — advisory.** Equal domain coverage under cell
   semantics, checked by `ms.is_consistent(rtol=...)` (or `check=True` flags),
   never forced. Rationale: enormous amounts of real data violate half-cell
   correctness — `::2` pyramids are ubiquitous, COG overview resolutions are
   inexact by construction (rounded overview shapes) — and refusing to
   represent data is a worse failure mode than representing it and offering a
   diagnostic. This mirrors how xarray treats CF conventions: representable
   when imperfect, verifiable when you care.

## The in-memory object: `Multiscale`

A new class layered on `DataTree` (subclass or thin wrapper): the tree shape
is constrained to a flat set of level nodes `{"0": ds0, "1": ds1, ...}`,
ordered fine → coarse, each holding the same variable and coordinate names.
DataTree supplies namespacing, IO plumbing, repr, and `map_over_datasets`;
`Multiscale` supplies the invariant and the algebra. The internal tree is an
implementation detail — in particular it is **not** a claim about storage
layout (see On-disk representation).

Levels are usually multi-variable (image + labels + masks), which is why
levels are Datasets, not bare arrays — and why a from-scratch
`MultiscaleDataArray` container independent of DataTree was rejected: it would
re-implement namespacing, IO, and mapping for the single-variable case only.
A pure convention-plus-accessor approach (no new type) was also rejected: it
is roughly what the ecosystem already has, and its limitation is exactly that
nothing can *dispatch* on multiscale-ness — `open_*` cannot hand back an
object with the algebra.

### Construction

```python
xr.Multiscale.from_datasets([ds0, ds1, ...], names=None)   # explicit assembly
ds.ms.coarsen(factors={"x": 2, "y": 2}, levels=4,
              boundary="trim", coord_func="mean")           # derivational constructor
xr.open_multiscale(store_or_path, group=None)               # recognition on read
Multiscale.from_flat(ds, pattern="{name}_{level}")          # unmangle legacy datasets
```

`from_datasets` is the **backend contract**: any source that can produce a
list of per-level Datasets can produce a `Multiscale`. The zarr manifest
(below) is just xarray's native recognizer; a rioxarray COG backend
(`open_rasterio(url, multiscale=True)`), an HDF5 pyramid reader, or a
kerchunk-assembled pyramid all target `from_datasets` directly.

### Access and projection

```python
ms.levels            # tuple of level names, fine -> coarse
ms.level(n), ms[n]   # -> Dataset (plain; projection loses nothing)
ms.finest, ms.coarsest
ms.scales            # {level: {dim: spacing}} — discovery is the repr
ms.transform(level)  # {"scale": {...}, "translation": {...}} derived from coords
                     #   under cell semantics, with explicit precision control
ms.items(), ms.values()  # mapping protocol over (name, Dataset)
```

`transform()` is deliberately public: the spacing/origin computation must
exist anyway for `is_consistent()`, and it is exactly what OME round-tripping
needs (writing `coordinateTransformations`) and what viewer handoff needs
(napari `scale=`/`translate=`). One implementation, three consumers, replacing
the fragile `coords[1] - coords[0]` idiom everywhere.

### Selection (world coordinates first)

```python
ms.sel(x=slice(a, b), y=slice(c, d))        # -> Multiscale: label selection applied
                                            #    at every level via its own coords
ms.sel(..., level=n)                        # -> Dataset: project, then select
ms.sel(..., resolution=r)                   # -> Dataset: coarsest level with
                                            #    spacing <= r in every selected dim
ms.sel(..., shape={"x": 1024, "y": 1024})   # -> Dataset: coarsest level yielding
                                            #    >= the requested samples over the
                                            #    selected extent (viewer/tile case)
```

`resolution=` and `shape=` are duals ("I need at least this sampling density"
vs "I have this many pixels"). Point selection composes with `method=`
(`ms.sel(z=60.0, method="nearest")`). Policy details (ties, dims not covered
by the request, `resolution="finest"/"coarsest"` sugar) are open questions
below.

### Transformation

```python
ms.map(fn, *args, **kwargs)   # -> Multiscale
```

Applies `fn: Dataset -> Dataset` per level, then **re-validates the
structural invariant**; a level that comes out with different dims or
coordinate names is an error naming the offending level. Arithmetic and
ufuncs forward through `map` (`ms - 273.15`, `np.log(ms)`).

`map` is intentionally restricted to geometry-compatible operations — it is
the closed algebra. Per-level *reductions* (histograms, class areas,
summaries) legitimately destroy the multiscale structure; for those, the
mapping protocol is the escape hatch:

```python
{name: fn(ds) for name, ds in ms.items()}   # plain dict, no invariant claimed
```

Operations whose *meaning* varies with resolution (a 5-pixel blur) are
allowed through `map` — the geometry is preserved; the semantics are the
caller's responsibility. The advisory check exists for exactly this boundary.

### Level management

```python
ms.add_level(ds)                              # geometric: any sampling of the domain
ms.add_level(factors={"x": 2, "y": 2}, ...)   # derivational: coarsen from an existing level
ms.drop_level(n)
```

## On-disk representation (zarr)

### Manifest, not layout

Scale levels of existing datasets are frequently *not* siblings under one
parent — anyone saving multiscale data with plain xarray today writes each
level as an independent dataset wherever it fits. The convention therefore
cannot imply containment. It is a **manifest**: a metadata document that
references levels by path.

```json
{
  "multiscales": [
    {
      "name": "signal",
      "levels": [
        {"path": "../raw/s0"},
        {"path": "derived/s1"},
        {"path": "s2"}
      ]
    }
  ]
}
```

- Paths resolve relative to the node carrying the manifest.
- Order in the list is the level order (fine → coarse).
- Each referenced node is an ordinary, independently-openable xarray zarr
  dataset with explicit 1-D coordinate arrays. Coordinates are stored per
  level: they are small, and explicit arrays are the only representation that
  handles irregular coordinates.
- v1 restriction: paths stay within a single store. Cross-store pyramids
  (level 0 on S3, overviews local) are supported in memory via explicit opens
  + `from_datasets`, but the written convention waits until zarr has a story
  for external references (URL resolution, auth, lifetime).

`ms.to_zarr(store, group=...)` writes the tidy default layout — sibling child
groups `0/, 1/, ...` plus a manifest pointing at its own children — but that
layout is a *default*, not the spec.

Properties this buys:

1. **Metadata-only migration.** Levels already scattered across a store join
   the convention by writing one attrs document. No data moves, no
   rechunking; every level remains a plain dataset for consumers that don't
   know the convention.
2. **Recognition on read**: `open_multiscale` (and `open_datatree` /
   `open_zarr` where applicable) finds a manifest, resolves references,
   validates structurally, returns a `Multiscale`.
3. The cost, stated honestly: discovery is not free-standing — a group is
   only identifiable as a pyramid level via a manifest that points at it, and
   a manifest whose targets moved is a validation error rather than an
   impossibility.

### Dialects

Other multiscale conventions map onto the manifest rather than being special
cases of a layout:

- **OME-NGFF** is a read/write dialect. Its `multiscales.datasets[].path` *is*
  a manifest (with an implicit children-only restriction we drop). Reading:
  `coordinateTransformations` (scale/translation) generate coordinate arrays —
  lazily via `CoordinateTransform` for large dims — so OME stores without
  explicit coordinate arrays still get world-coordinate `sel()`. Writing:
  `ms.to_zarr(store, dialects=("xarray", "ome-ngff"))` emits OME attrs
  alongside, computed by `ms.transform(level)`, giving napari/viv interop
  without xarray adopting OME's model wholesale.
- **Consumer-specific metadata** (e.g. carbonplan maps' `pixels_per_tile`,
  its `y`/`x` naming requirements) belongs in dialects, not in xarray's
  manifest. Producer code stops absorbing per-consumer conventions.

## Motivation: three real examples, before and after

These are real published notebooks/scripts, condensed but structurally
faithful. They span three communities that do not share code.

### 1. Building a web-map pyramid

Source: Development Seed tile-benchmarking, CMIP6 pyramid generation
(<https://developmentseed.org/tile-benchmarking/01-generate-datasets/generate-cmip6-pyramid.html>),
using ndpyramid (<https://github.com/carbonplan/ndpyramid>).

Today:

```python
pyramid = pyramid_reproject(ds, projection='equidistant-cylindrical',
                            other_chunks={'time': 1}, levels=LEVELS)

# per-level fixups, by hand, over tree children
for child in pyramid.children.values():
    child.ds = set_zarr_encoding(child.ds, codec_config={"id": "zlib", "level": 1},
                                 float_dtype="float32")
    if 'x' in child.ds and 'y' in child.ds:
        child.ds = child.ds.rename({'x': 'lon', 'y': 'lat'})
    child.ds = child.ds.chunk({"lat": -1, "lon": -1, "time": 1})
    child.ds[variable].attrs.clear()

# hand-assembled multiscales metadata
pyramid.ds.attrs['multiscales'] = [{'datasets': {}, 'metadata': {'version': 2}}]
for level in range(LEVELS):
    pyramid.ds.attrs['multiscales'][0]['datasets'][level] = {'pixels_per_tile': 128}
pyramid.to_zarr(save_path)

# consumption: each level reopened as an unrelated Dataset
root = xr.open_zarr(save_path, consolidated=True)
for group in root.multiscales[0]['datasets'].keys():
    level_ds = xr.open_zarr(save_path, consolidated=True, group=group)
    level_ds.isel(time=0).tas.plot()
```

With `Multiscale`:

```python
pyramid = pyramid_reproject(ds, ..., levels=LEVELS)          # returns xr.Multiscale

pyramid = pyramid.map(lambda ds: ds.rename({'x': 'lon', 'y': 'lat'})
                                   .chunk({'lat': -1, 'lon': -1, 'time': 1}))
pyramid.to_zarr(save_path, encoding=enc)                     # manifest written for you

ms = xr.open_multiscale(save_path)
for name, level in ms.items():
    level.isel(time=0).tas.plot()
ms.sel(shape={'lon': 256, 'lat': 256}).isel(time=0).tas.plot()   # level picked for you
```

What the example teaches:

- The `if 'x' in child.ds` guard exists because levels can silently be
  heterogeneous. `map`'s post-validation turns that into a build-time error
  naming the level, instead of a browser-side tiling failure later.
- `pixels_per_tile` and carbonplan's `y`/`x` requirement (which conflicts with
  this notebook's `lat`/`lon` renaming for a different tiler!) are the
  concrete case for dialects.
- `to_zarr(encoding=...)` needs defined broadcast-over-levels semantics,
  since variable names repeat across levels.

### 2. Picking a COG overview level

Source: Digital Earth Australia, "Handling Cloud-Optimised Geotiff overviews"
(<https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/COG_overviews/>);
also the canonical rioxarray COG example
(<https://corteva.github.io/rioxarray/stable/examples/COG.html>).

Today — discovery requires leaving xarray for GDAL, selection is a magic
number, and comparing two levels means opening the same file four times:

```python
cog = gdal.Open(f'/vsicurl/{cog_url}')
band = cog.GetRasterBand(1)
for i in range(band.GetOverviewCount()):
    ov = band.GetOverview(i)
    print(i, base_res_x * band.XSize / ov.XSize)     # index -> resolution, by hand

cog_array = rioxarray.open_rasterio(cog_url, overview_level=2)   # "a medium level"
cog_roi = crop(cog_array, roi_geom)

fine   = crop(rioxarray.open_rasterio(cog_url), roi_geom)
coarse = crop(rioxarray.open_rasterio(cog_url, overview_level=3), roi_geom)
fine.plot(ax=ax[0]); coarse.plot(ax=ax[1])
```

With `Multiscale`:

```python
ms = rioxarray.open_rasterio(cog_url, multiscale=True)   # one open; overviews are levels
ms.scales                                # {'0': {'x': 60.0, ...}, ..., '7': {'x': 3840.0, ...}}

roi = ms.sel(x=slice(x0, x1), y=slice(y0, y1))   # crop once, in world coords, all levels
roi.sel(resolution=250).plot()                   # "about 250 m" — no index arithmetic

roi.finest.plot(ax=ax[0])
roi.level(3).plot(ax=ax[1])
```

What the example teaches:

- The source is a **GeoTIFF, not zarr**: the in-memory algebra must be
  format-agnostic, with `from_datasets` as the backend contract and the
  manifest as merely one recognizer.
- The notebook's cross-level class-area comparison is a per-level reduction —
  the case for the `items()` escape hatch rather than forcing everything
  through `map`.
- Overview resolutions are inexact by construction (79.9708… m, not 80 m):
  strict geometric validation would reject virtually every real COG. The
  advisory check is the right strictness.

### 3. Reading OME-Zarr and handing it to a viewer

Sources: xarray-ome-ngff (<https://janeliascicomp.github.io/xarray-ome-ngff/>)
reading an IDR dataset; spatialdata tutorials
(<https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/transformations_advanced.html>);
napari xarray gallery
(<https://napari.org/0.6.1/gallery/xarray-latlon-timeseries.html>).

Today — three libraries, three incompatible answers, and viewer handoff
reverse-engineers the affine from coords:

```python
group = zarr.open_group("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr")
arrays = read_multiscale_group(group, array_wrapper=DaskArrayWrapper(chunks=10))
img = arrays['0']            # magic key; '0'? 's0'? 'scale0'? depends on the writer

# spatialdata's version of "give me one level":
sdata["raw_image"] = sdata["raw_image"]["scale0"]["image"]

# napari handoff: recover scale/translate by differencing coords
def get_scale_translate(dataset, name):
    dims = [getattr(dataset, d) for d in getattr(dataset, name).dims]
    return {'translate': [float(d[0]) for d in dims],
            'scale':     [float(d[1] - d[0]) for d in dims]}
```

With `Multiscale`:

```python
ms = xr.open_multiscale("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr")
# OME dialect auto-detected; micron coords generated lazily from transforms

ms.sel(z=60.0, method='nearest')                                # one slice, every level
ms.sel(z=60.0, method='nearest', resolution=1.0).image.plot()   # ~1 um/px, level chosen

viewer.add_image([lvl.image.data for lvl in ms.values()],       # napari is natively multiscale
                 **ms.transform(0))
```

What the example teaches:

- `transform(level)` must be public API: xarray-ome-ngff bolted the same
  computation on as `transform_precision`, napari users hand-write it, and
  `is_consistent()` needs it internally anyway.
- spatialdata's deepest problem — per-level transforms stored in `.attrs`, so
  levels do not share a physical coordinate system and world-coordinate
  `sel()` is impossible — is a missing invariant, not a missing feature.
  `Multiscale` makes "levels share a world domain, expressed in coords" the
  definition.
- Laziness must be the default on open (chunks from the store), not an
  opt-in wrapper.

### The pattern

| Gap | web-map pyramid | COG overviews | OME-Zarr |
|---|---|---|---|
| What levels exist, at what resolution? | parse hand-built attrs | GDAL loop | know the writer's key convention |
| Get the right level for a purpose | `group=2` | `overview_level=2` | `['scale0']['image']` |
| Do one thing to all levels | loop over children | open file 4x | loop over dict |
| coords <-> affine | broke consumer contract | rasterio-internal, inexact | `d[1]-d[0]` differencing |

Every row is the same missing object.

## Rejected alternatives

- **Derivational invariant** (levels defined as downsamples of level 0):
  unenforceable after edits; excludes multi-resolution acquisitions; the
  recipe survives as optional provenance + the `coarsen` constructor.
- **Strict geometric validation at construction**: tolerance-based equality
  is mushy and real data (`::2` pyramids, rounded COG overviews) fails it;
  advisory check instead.
- **Pure convention + accessor, no type**: cannot dispatch; is the status quo.
- **`MultiscaleDataArray` independent of DataTree**: duplicates namespacing,
  IO, repr, mapping; real pyramids are multi-variable anyway.
- **Layout-based zarr convention (levels as siblings)**: existing data is not
  laid out that way; manifest instead, sibling layout demoted to the default
  write shape.
- **Cross-store manifest references in v1**: needs zarr-level answers on URL
  resolution/auth/lifetime first; in-memory assembly covers the use case.

## Open questions

1. **Selection policy details.** Tie-breaking for `resolution=`/`shape=`;
   behavior when the request names a subset of dims; anisotropic spacing
   (level chosen per-dim can conflict); `resolution="finest"/"coarsest"`
   sugar. The `shape=` policy stretched furthest past what discussion
   settled.
2. **`map` and laziness.** `map` must not eagerly compute (levels are
   usually dask); invariant re-validation must be metadata-only.
3. **Arithmetic dunders.** How much of the Dataset API to forward through
   `map` (binary ops between two `Multiscale`s with different level sets?).
4. **Where it incubates.** `xarray.experimental`, or an external package
   targeting upstreaming (the DataTree path). This repo (xarray-multiscale)
   is a natural incubator for the object + manifest, with the `coarsen`
   constructor it already implements.
5. **DataTree coupling.** Subclass vs wrapper; how much of DataTree's public
   surface leaks through (and whether future DataTree semantics changes
   ripple in).
6. **Manifest schema details.** Versioning, multiple pyramids per manifest,
   naming, relationship to zarr conventions work and to OME-NGFF's
   collections/bioformats2raw layouts.
