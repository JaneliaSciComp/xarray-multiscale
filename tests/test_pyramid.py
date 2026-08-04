from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr

from xarray_multiscale.pyramid import Multiscale, open_multiscale
from xarray_multiscale.reducers import windowed_mean


def make_level(shape: dict[str, int], spacing: dict[str, float], value=None, var="tas"):
    """
    A level whose coords are cell centers of a domain starting at 0:
    coord[i] = spacing/2 + i*spacing, so every level with the same
    (size * spacing) product covers the same domain exactly.
    """
    coords = {
        dim: spacing[dim] / 2 + spacing[dim] * np.arange(n, dtype="float64")
        for dim, n in shape.items()
    }
    if value is None:
        data = np.random.default_rng(0).random(tuple(shape.values()))
    else:
        data = np.full(tuple(shape.values()), value, dtype="float64")
    return xr.Dataset({var: (tuple(shape), data)}, coords=coords)


def make_pyramid(n_levels=3, dims=("y", "x"), base=8, extra_var=False):
    levels = []
    for i in range(n_levels):
        factor = 2**i
        shape = {d: base // factor for d in dims}
        spacing = {d: float(factor) for d in dims}
        ds = make_level(shape, spacing, value=i)
        if extra_var:
            ds["mask"] = xr.zeros_like(ds["tas"])
        levels.append(ds)
    return Multiscale.from_datasets(levels)


# ----------------------------------------------------------------------
# construction and access
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dims", [("x",), ("y", "x"), ("t", "y", "x")])
@pytest.mark.parametrize("n_levels", [1, 3])
@pytest.mark.parametrize("extra_var", [False, True])
def test_construction_and_access(dims, n_levels, extra_var):
    ms = make_pyramid(n_levels=n_levels, dims=dims, extra_var=extra_var)

    assert ms.levels == tuple(str(i) for i in range(n_levels))
    assert len(ms) == n_levels
    # mapping protocol and integer/name access agree
    assert list(ms) == list(ms.levels)
    xr.testing.assert_identical(ms[0], ms["0"])
    xr.testing.assert_identical(ms[-1], ms[str(n_levels - 1)])
    xr.testing.assert_identical(ms.finest, ms[0])
    xr.testing.assert_identical(ms.coarsest, ms[-1])
    xr.testing.assert_identical(ms.level(0), ms[0])
    # scales report per-dim spacing, fine -> coarse
    for i, name in enumerate(ms.levels):
        for d in dims:
            assert ms.scales[name][d] == pytest.approx(2.0**i)
    # transform derives affine from coords
    tf = ms.transform(-1)
    factor = 2.0 ** (n_levels - 1)
    for d in dims:
        assert tf["scale"][d] == pytest.approx(factor)
        assert tf["translate"][d] == pytest.approx(factor / 2)
    # from_datasets on DataArrays wraps them into datasets
    ms2 = Multiscale.from_datasets([ms[i]["tas"] for i in range(n_levels)])
    assert list(ms2.finest.data_vars) == ["tas"]


def test_construction_empty_error():
    with pytest.raises(ValueError, match="at least one level"):
        Multiscale({})


def test_construction_mismatched_dims_error():
    a = make_level({"y": 8, "x": 8}, {"y": 1.0, "x": 1.0})
    b = make_level({"y": 4, "z": 4}, {"y": 2.0, "z": 2.0})
    with pytest.raises(ValueError, match="dims"):
        Multiscale.from_datasets([a, b])


def test_construction_mismatched_variables_error():
    a = make_level({"x": 8}, {"x": 1.0}, var="tas")
    b = make_level({"x": 4}, {"x": 2.0}, var="pr")
    with pytest.raises(ValueError, match="data variables"):
        Multiscale.from_datasets([a, b])


def test_construction_nonmonotonic_coord_error():
    a = make_level({"x": 8}, {"x": 1.0})
    b = make_level({"x": 4}, {"x": 2.0})
    b = b.assign_coords(x=[1.0, 5.0, 3.0, 7.0])
    with pytest.raises(ValueError, match="not monotonic"):
        Multiscale.from_datasets([a, b])


def test_construction_inconsistent_direction_error():
    a = make_level({"x": 8}, {"x": 1.0})
    b = make_level({"x": 4}, {"x": 2.0})
    b = b.assign_coords(x=b.x.values[::-1])
    with pytest.raises(ValueError, match="direction"):
        Multiscale.from_datasets([a, b])


def test_construction_wrong_order_error():
    a = make_level({"x": 8}, {"x": 1.0})
    b = make_level({"x": 4}, {"x": 2.0})
    with pytest.raises(ValueError, match="fine to coarse"):
        Multiscale.from_datasets([b, a])


def test_from_datasets_names_length_error():
    a = make_level({"x": 8}, {"x": 1.0})
    with pytest.raises(ValueError, match="names"):
        Multiscale.from_datasets([a], names=["s0", "s1"])


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def test_is_consistent():
    # cell-centered mean pyramid: exactly consistent
    assert make_pyramid().is_consistent()
    # ::2 subsampling: half-cell shift, inconsistent at tight rtol
    fine = make_level({"x": 100}, {"x": 1.0})
    sub = fine.isel(x=slice(None, None, 2))
    ms = Multiscale.from_datasets([fine, sub])
    assert not ms.is_consistent(rtol=1e-3)
    # ... but representable, and passes with a forgiving tolerance
    assert ms.is_consistent(rtol=0.1)


def test_downscale_constructor():
    data = xr.DataArray(
        np.arange(64, dtype="float64").reshape(8, 8),
        dims=("y", "x"),
        coords={"y": np.arange(8.0), "x": np.arange(8.0)},
        name="img",
    )
    ms = Multiscale.downscale(data, windowed_mean, 2)
    assert len(ms) == 4  # 8 -> 4 -> 2 -> 1
    assert ms.is_consistent()
    assert list(ms.finest.data_vars) == ["img"]


# ----------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------
def test_sel():
    ms = make_pyramid(n_levels=3, base=16)

    # world-coordinate crop applies at every level -> Multiscale
    roi = ms.sel(x=slice(0, 8), y=slice(0, 8))
    assert isinstance(roi, Multiscale)
    assert roi["0"].sizes == {"y": 8, "x": 8}
    assert roi["1"].sizes == {"y": 4, "x": 4}
    assert roi["2"].sizes == {"y": 2, "x": 2}

    # level= projects a single selected Dataset
    one = ms.sel(x=slice(0, 8), level=1)
    assert isinstance(one, xr.Dataset)
    assert one.sizes == {"y": 8, "x": 4}

    # resolution=: coarsest level with spacing <= r
    assert ms.sel(resolution=1.0).sizes["x"] == 16   # only level 0
    assert ms.sel(resolution=2.5).sizes["x"] == 8    # level 1
    assert ms.sel(resolution=100.0).sizes["x"] == 4  # level 2
    # finer than available: best effort, finest level
    assert ms.sel(resolution=0.1).sizes["x"] == 16
    # per-dim mapping form
    assert ms.sel(resolution={"x": 2.0}).sizes["x"] == 8

    # shape=: coarsest level still yielding the requested samples
    assert ms.sel(shape={"x": 4, "y": 4}).sizes["x"] == 4
    assert ms.sel(shape={"x": 5, "y": 5}).sizes["x"] == 8
    # shape interacts with the crop: half the domain at >= 4 samples
    got = ms.sel(x=slice(0, 8), y=slice(0, 8), shape={"x": 4, "y": 4})
    assert got.sizes == {"y": 4, "x": 4}
    # nothing big enough: finest wins
    assert ms.sel(shape={"x": 1000}).sizes["x"] == 16

    # scalar point selection with method, at every level
    pt = ms.sel(x=3.3, y=3.3, method="nearest")
    assert isinstance(pt, Multiscale)
    assert all("x" not in ds.dims for ds in pt.values())


def test_sel_conflicting_selectors_error():
    ms = make_pyramid()
    with pytest.raises(ValueError, match="at most one"):
        ms.sel(level=0, resolution=1.0)


def test_sel_unknown_level_error():
    ms = make_pyramid()
    with pytest.raises(KeyError):
        ms.sel(level="99")


# ----------------------------------------------------------------------
# transformation
# ----------------------------------------------------------------------
def test_map_and_arithmetic():
    ms = make_pyramid(n_levels=3)

    renamed = ms.map(lambda ds: ds.rename({"tas": "temperature"}))
    assert isinstance(renamed, Multiscale)
    assert list(renamed.finest.data_vars) == ["temperature"]
    assert renamed.levels == ms.levels

    shifted = ms - 1.0
    assert float(shifted["1"]["tas"].mean()) == pytest.approx(0.0)
    scaled = 2.0 * ms
    assert float(scaled["2"]["tas"].mean()) == pytest.approx(4.0)
    neg = -ms
    assert float(neg["1"]["tas"].mean()) == pytest.approx(-1.0)

    # per-level reductions use the mapping protocol, not map
    means = {name: float(ds["tas"].mean()) for name, ds in ms.items()}
    assert means == {"0": 0.0, "1": 1.0, "2": 2.0}


def test_map_invariant_violation_error():
    ms = make_pyramid(n_levels=2)

    def bad(ds):
        # renames a coordinate on one level only
        if ds.sizes["x"] == 4:
            return ds.rename({"x": "lon"})
        return ds

    with pytest.raises(ValueError, match="invariant"):
        ms.map(bad)


def test_map_function_failure_names_level():
    ms = make_pyramid(n_levels=2)

    def boom(ds):
        if ds.sizes["x"] == 4:
            raise RuntimeError("kaboom")
        return ds

    with pytest.raises(RuntimeError, match="level '1'"):
        ms.map(boom)


# ----------------------------------------------------------------------
# level management
# ----------------------------------------------------------------------
def test_add_and_drop_level():
    ms = make_pyramid(n_levels=3, base=16)
    # a level sampled at spacing 8, produced "elsewhere" (geometric view)
    new = make_level({"y": 2, "x": 2}, {"y": 8.0, "x": 8.0}, value=3)

    grown = ms.add_level(new)
    assert len(grown) == 4
    assert grown.coarsest.sizes == {"y": 2, "x": 2}
    # inserted in spacing order even when added out of order
    middle = make_level({"y": 16, "x": 16}, {"y": 0.5, "x": 0.5}, value=-1)
    regrown = grown.add_level(middle, name="fine")
    assert regrown.levels[0] == "fine"

    dropped = grown.drop_level("1")
    assert dropped.levels == ("0", "2", "3")


def test_add_level_duplicate_name_error():
    ms = make_pyramid()
    with pytest.raises(ValueError, match="already exists"):
        ms.add_level(ms["2"], name="2")


def test_drop_only_level_error():
    ms = make_pyramid(n_levels=1)
    with pytest.raises(ValueError, match="only level"):
        ms.drop_level("0")


# ----------------------------------------------------------------------
# zarr IO
# ----------------------------------------------------------------------
def test_zarr_roundtrip(tmp_path):
    ms = make_pyramid(n_levels=3, extra_var=True)
    store = str(tmp_path / "pyramid.zarr")
    ms.to_zarr(store, name="signal")

    # the manifest is the convention
    attrs = zarr.open_group(store, mode="r").attrs.asdict()
    assert attrs["multiscales"][0]["levels"] == [
        {"path": "0"},
        {"path": "1"},
        {"path": "2"},
    ]
    # every level remains an ordinary, independently openable dataset
    plain = xr.open_zarr(store, group="1")
    xr.testing.assert_identical(plain.load(), ms["1"])

    back = open_multiscale(store)
    assert back.levels == ms.levels
    for name in ms.levels:
        xr.testing.assert_identical(back[name].load(), ms[name])


def test_zarr_roundtrip_in_subgroup(tmp_path):
    ms = make_pyramid(n_levels=2)
    store = str(tmp_path / "root.zarr")
    ms.to_zarr(store, group="deeply/nested")
    back = open_multiscale(store, group="deeply/nested")
    xr.testing.assert_identical(back["0"].load(), ms["0"])


def test_scattered_manifest(tmp_path):
    """Levels do NOT need to be siblings: the manifest references paths."""
    ms = make_pyramid(n_levels=2)
    store = str(tmp_path / "scattered.zarr")
    # levels live in unrelated corners of the store, written independently
    ms["0"].to_zarr(store, group="raw/acquisition")
    ms["1"].to_zarr(store, group="derived/downsampled/v2")
    # joining the convention is a metadata-only operation
    root = zarr.open_group(store, path="pyramids/signal", mode="a")
    root.attrs["multiscales"] = [
        {
            "name": "signal",
            "levels": [
                {"path": "../../raw/acquisition", "name": "full"},
                {"path": "../../derived/downsampled/v2", "name": "half"},
            ],
        }
    ]

    ms2 = open_multiscale(store, group="pyramids/signal")
    assert ms2.levels == ("full", "half")
    xr.testing.assert_identical(ms2["full"].load(), ms["0"])
    xr.testing.assert_identical(ms2["half"].load(), ms["1"])


def test_manifest_path_escape_error(tmp_path):
    store = str(tmp_path / "escape.zarr")
    root = zarr.open_group(store, mode="a")
    root.attrs["multiscales"] = [{"levels": [{"path": "../outside"}]}]
    with pytest.raises(ValueError, match="escapes the store"):
        open_multiscale(store)


def test_open_no_manifest_error(tmp_path):
    store = str(tmp_path / "plain.zarr")
    make_level({"x": 4}, {"x": 1.0}).to_zarr(store)
    with pytest.raises(ValueError, match="no 'multiscales' metadata"):
        open_multiscale(store)


def test_ome_ngff_dialect_roundtrip(tmp_path):
    ms = make_pyramid(n_levels=2, dims=("y", "x"))
    store = str(tmp_path / "ome.zarr")
    ms.to_zarr(store, name="tas", dialects=("xarray", "ome-ngff"))

    attrs = zarr.open_group(store, mode="r").attrs.asdict()
    native, ome = attrs["multiscales"]
    assert ome["version"] == "0.4"
    assert [ax["name"] for ax in ome["axes"]] == ["y", "x"]
    assert ome["datasets"][0]["path"] == "0/tas"
    assert ome["datasets"][1]["coordinateTransformations"][0] == {
        "type": "scale",
        "scale": [2.0, 2.0],
    }

    # a store carrying ONLY OME metadata opens via transform-generated coords
    root = zarr.open_group(store, mode="a")
    root.attrs["multiscales"] = [ome]
    back = open_multiscale(store)
    assert back.levels == ("0/tas", "1/tas")
    assert list(back.finest.data_vars) == ["tas"]
    for name, orig_name in zip(back.levels, ms.levels):
        xr.testing.assert_allclose(
            back[name]["tas"].load(), ms[orig_name]["tas"]
        )


def test_ome_dialect_multivar_error(tmp_path):
    ms = make_pyramid(n_levels=2, extra_var=True)
    with pytest.raises(ValueError, match="exactly one data variable"):
        ms.to_zarr(str(tmp_path / "x.zarr"), dialects=("xarray", "ome-ngff"))


# ----------------------------------------------------------------------
# functional (transform-backed) coordinates
# ----------------------------------------------------------------------
pytest.importorskip("xarray.indexes", reason="needs xarray with functional coords")
from xarray.indexes import CoordinateTransformIndex, RangeIndex  # noqa: E402

from xarray_multiscale.pyramid import _functional_coords  # noqa: E402


def write_ome_store(path, sizes, scales, translations, n_levels=3, factor=2):
    """A store carrying ONLY OME-NGFF metadata: no coordinate arrays."""
    root = zarr.open_group(path, mode="w")
    datasets = []
    for i in range(n_levels):
        shape = tuple(max(1, n // factor**i) for n in sizes.values())
        arr = root.create_array(f"{i}", shape=shape, dtype="uint8", chunks=shape)
        arr[:] = i
        lvl_scale = [scales[d] * factor**i for d in sizes]
        # cell-centered: the level's origin shifts by half the extra width
        lvl_trans = [
            translations[d] + (scales[d] * (factor**i - 1)) / 2 for d in sizes
        ]
        datasets.append(
            {
                "path": f"{i}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": lvl_scale},
                    {"type": "translation", "translation": lvl_trans},
                ],
            }
        )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "nuclei",
            "axes": [{"name": d, "type": "space"} for d in sizes],
            "datasets": datasets,
        }
    ]
    return path


def test_ome_coords_are_functional(tmp_path):
    store = write_ome_store(
        str(tmp_path / "ome.zarr"),
        sizes={"z": 64, "y": 256, "x": 256},
        scales={"z": 0.5, "y": 0.18, "x": 0.18},
        translations={"z": 0.25, "y": 0.09, "x": 0.09},
    )
    ms = open_multiscale(store)

    # coordinates are declared, not materialized
    for name, ds in ms.items():
        for dim in ("z", "y", "x"):
            assert isinstance(ds.xindexes[dim], CoordinateTransformIndex), (
                f"{name}/{dim} is not functional"
            )
            assert not isinstance(ds.coords[dim].variable._data, np.ndarray)

    # ... and the declared parameters come back exactly, not by differencing
    assert ms.transform(0) == {
        "scale": {"z": 0.5, "y": 0.18, "x": 0.18},
        "translate": {"z": 0.25, "y": 0.09, "x": 0.09},
    }
    assert ms.scales["1"] == {"z": 1.0, "y": 0.36, "x": 0.36}
    assert ms.is_consistent()


def test_exact_scale_beats_differencing():
    """The transform is exact where differencing stored floats is not."""
    n, scale, trans = 256, 0.18, 0.09
    coords = _functional_coords(["x"], [n], [scale], [trans])
    ds = xr.DataArray(np.zeros(n), dims="x", coords=coords, name="v").to_dataset()
    ms = Multiscale.from_datasets([ds])

    assert ms.scales["0"]["x"] == scale  # exactly, not approximately
    differenced = float(np.diff(ds.coords["x"].values[:2])[0])
    assert differenced != scale  # the drift this avoids is real
    assert differenced == pytest.approx(scale)


def test_functional_coords_are_not_materialized():
    """An axis of any length costs nothing: 1e9 samples, no allocation."""
    n = 1_000_000_000
    ds = xr.Dataset(coords=_functional_coords(["q"], [n], [2.5], [1.0]))
    ms = Multiscale.from_datasets([ds])

    assert ms.scales["0"]["q"] == 2.5
    assert ms.transform(0)["translate"]["q"] == 1.0
    assert ms.finest.sizes["q"] == n


def test_sel_on_functional_coords(tmp_path):
    store = write_ome_store(
        str(tmp_path / "ome.zarr"),
        sizes={"z": 64, "y": 256, "x": 256},
        scales={"z": 0.5, "y": 0.18, "x": 0.18},
        translations={"z": 0.25, "y": 0.09, "x": 0.09},
    )
    ms = open_multiscale(store)

    # plain xarray requires method='nearest' on a transform-backed index;
    # Multiscale supplies it, so the natural expression works
    roi = ms.sel(y=slice(1.0, 5.0), x=slice(1.0, 5.0))
    assert isinstance(roi, Multiscale)
    assert roi["0"].sizes == {"z": 64, "y": 22, "x": 22}
    # coords stay functional through selection
    assert isinstance(roi["0"].xindexes["x"], CoordinateTransformIndex)

    # mixed point and slice selection, plus a level policy
    plane = ms.sel(z=16.0, y=slice(1.0, 5.0), resolution={"x": 0.5, "y": 0.5})
    assert isinstance(plane, xr.Dataset)
    assert "z" not in plane.dims
    assert plane.sizes["x"] == 128


def test_sel_mixed_functional_and_explicit_coords():
    """
    The case plain xarray cannot express in one call: a slice on an explicit
    coordinate together with a selection on a functional one.
    """
    n = 256
    ds = xr.Dataset(
        {"v": (("t", "x"), np.zeros((10, n)))},
        coords={"t": np.arange(10.0)},
    ).assign_coords(_functional_coords(["x"], [n], [0.18], [0.09]))
    assert isinstance(ds.xindexes["x"], CoordinateTransformIndex)
    assert not isinstance(ds.xindexes["t"], CoordinateTransformIndex)

    # plain xarray: one method for the whole call satisfies neither index
    with pytest.raises(ValueError, match="nearest"):
        ds.sel(t=slice(2.0, 5.0), x=slice(1.0, 5.0))

    ms = Multiscale.from_datasets([ds])
    got = ms.sel(t=slice(2.0, 5.0), x=slice(1.0, 5.0))
    assert got["0"].sizes == {"t": 4, "x": 22}

    # point selection with a method still reaches the explicit coordinate
    got = ms.sel(t=2.2, x=1.0, method="nearest")
    assert got["0"].sizes == {}


def test_sel_functional_unsupported_method_error():
    ds = xr.Dataset(
        {"v": ("x", np.zeros(16))},
        coords=_functional_coords(["x"], [16], [0.5], [0.25]),
    )
    ms = Multiscale.from_datasets([ds])
    with pytest.raises(ValueError, match="only method='nearest'"):
        ms.sel(x=1.0, method="pad")


def test_sel_functional_tolerance_error():
    ds = xr.Dataset(
        {"v": ("x", np.zeros(16))},
        coords=_functional_coords(["x"], [16], [0.5], [0.25]),
    )
    ms = Multiscale.from_datasets([ds])
    with pytest.raises(ValueError, match="tolerance"):
        ms.sel(x=1.0, tolerance=0.1)


def test_ome_transform_roundtrip_is_exact(tmp_path):
    """OME transforms -> functional coords -> OME transforms, unchanged."""
    scales = {"z": 0.5, "y": 0.18, "x": 0.18}
    translations = {"z": 0.25, "y": 0.09, "x": 0.09}
    store = write_ome_store(
        str(tmp_path / "in.zarr"),
        sizes={"z": 8, "y": 64, "x": 64},
        scales=scales,
        translations=translations,
        n_levels=2,
    )
    ms = open_multiscale(store)
    emitted = ms._ome_manifest(name="nuclei")

    original = zarr.open_group(store, mode="r").attrs["multiscales"][0]["datasets"]
    for orig, new in zip(original, emitted["datasets"]):
        o_scale, o_trans = orig["coordinateTransformations"]
        n_scale, n_trans = new["coordinateTransformations"]
        assert n_scale["scale"] == o_scale["scale"]
        assert n_trans["translation"] == o_trans["translation"]
