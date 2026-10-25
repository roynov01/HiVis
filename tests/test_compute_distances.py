# -*- coding: utf-8 -*-
"""
Synthetic tests for AnalysisAgg.compute_distances and AnalysisVisium.compute_distances.
Compares the current implementation with the one from HiVis 1.0.5.

Run with pytest, or directly: python tests/test_compute_distances.py
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from shapely import wkt, from_wkt
from shapely.geometry import Point, box
from shapely.strtree import STRtree

from HiVis import HiVis

MPP = 0.5 # microns per pixel
TMP_DIRS = []


# ---------- HiVis 1.0.5 implementations, kept verbatim for comparison ----------

def orig_agg_compute_distances(self, target_agg, dist_col_name=None, nearest_col_name=None):
    '''AnalysisAgg.compute_distances from HiVis 1.0.5. self is an AnalysisAgg'''
    if isinstance(target_agg, str):
        target_agg = self.main.viz.agg[target_agg]
    if target_agg is self:
        raise ValueError("target_agg should be either another Agg object or key in HiVis.agg")

    target_name = target_agg.name
    dist_col_name = dist_col_name    or f"dist_to_{target_name}"
    nearest_col_name = nearest_col_name or f"nearest_{target_name}"

    def to_geoms(series):
        if series.dtype == "object" and isinstance(series.iloc[0], str):
            return series.map(from_wkt)
        return series

    cur_geoms = to_geoms(self.main.adata.obs.geometry)
    target_geoms = to_geoms(target_agg.adata.obs.geometry)

    target_list = target_geoms.tolist()
    tree  = STRtree(target_list)
    distances = [0.0]  * len(cur_geoms)
    nearest_ids = [None] * len(cur_geoms)

    for i, poly in enumerate(cur_geoms):
        nearest_idx   = int(tree.nearest(poly))
        nearest_geom  = target_list[nearest_idx]
        distances[i]  = poly.distance(nearest_geom)
        nearest_ids[i]= target_agg.adata.obs.index[nearest_idx]

    self.main.adata.obs[dist_col_name] = distances
    self.main.adata.obs[nearest_col_name] = nearest_ids


def orig_visium_compute_distances(self, agg_name, dist_col_name=None, nearest_col_name=None):
    '''AnalysisVisium.compute_distances from HiVis 1.0.5. self is an AnalysisVisium'''
    if agg_name not in self.main.agg:
        raise ValueError(f"{agg_name} is not a valid aggregation in {self.main.name}")

    dist_col_name = dist_col_name if dist_col_name is not None else f"dist_to_{agg_name}"
    nearest_col_name = nearest_col_name if nearest_col_name is not None else f"nearest_{agg_name}"

    target_geoms = self.main.agg[agg_name].adata.obs.geometry.apply(wkt.loads).tolist()
    tree = STRtree(target_geoms)

    x_coords = self.main["um_x"]
    y_coords = self.main["um_y"]
    n_pts = len(x_coords)
    distances = [0.0] * n_pts
    nearest_obs_ids = [None] * n_pts

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        pt = Point(x, y)
        nearest_index = tree.nearest(pt)
        nearest_geom = target_geoms[nearest_index]
        distances[i] = pt.distance(nearest_geom)
        nearest_obs_ids[i] = self.main.agg[agg_name].adata.obs.index[nearest_index]

    self.main.adata.obs[dist_col_name] = distances
    self.main.adata.obs[nearest_col_name] = nearest_obs_ids


# ---------- synthetic data ----------

def _squares(centers_um, side_um):
    return [box(x - side_um / 2, y - side_um / 2, x + side_um / 2, y + side_um / 2) for x, y in centers_um]


def _agg_adata(centers_um, prefix):
    obs = pd.DataFrame({"pxl_col_in_fullres": [x / MPP for x, _ in centers_um],
                        "pxl_row_in_fullres": [y / MPP for _, y in centers_um]},
                       index=[f"{prefix}{i}" for i in range(len(centers_um))])
    X = sparse.csr_matrix(np.ones((len(centers_um), 3), dtype=np.float32))
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"]))


def make_viz(geometry="str"):
    '''
    Builds a small HiVis with two aggregations, "SC" (cells) and "colonies", with square polygons in obs["geometry"].
    Integer coordinates on a grid make some cells equidistant to two colonies (tests tie-breaking).
    geometry - "str" (pandas 3 default), "object" (pandas < 3) or "shapely"
    '''
    path_output = tempfile.mkdtemp(prefix="hivis_test_")
    TMP_DIRS.append(path_output)
    n = 20
    xs, ys = np.meshgrid(np.arange(n) * 10.0, np.arange(n) * 10.0)
    obs = pd.DataFrame({"pxl_col_in_fullres": xs.ravel(), "pxl_row_in_fullres": ys.ravel()},
                       index=[f"bin{i}" for i in range(n * n)])
    adata = ad.AnnData(X=sparse.csr_matrix(np.ones((n * n, 3), dtype=np.float32)), obs=obs,
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    json = {"microns_per_pixel": MPP, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
            "spot_diameter_fullres": 4}
    viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name="viz",
                      path_output=path_output, plot_qc=False)

    rng = np.random.default_rng(0)
    cell_centers = [(float(x), float(y)) for x, y in rng.integers(2, 98, size=(60, 2))]
    cell_centers += [(50.0, 20.0), (20.0, 50.0)] # equidistant to two colonies
    colony_centers = [(20.0, 20.0), (80.0, 20.0), (20.0, 80.0), (80.0, 80.0), (50.0, 50.0)]

    viz.add_agg(_agg_adata(cell_centers, "cell"), "SC")
    viz.add_agg(_agg_adata(colony_centers, "colony"), "colonies")
    set_geometry(viz.agg["SC"], _squares(cell_centers, 2), geometry)
    set_geometry(viz.agg["colonies"], _squares(colony_centers, 10), geometry)
    return viz


def set_geometry(agg, geoms, kind):
    '''Sets obs["geometry"]. geoms can contain None / np.nan for missing values'''
    obs = agg.adata.obs
    if kind == "shapely":
        obs["geometry"] = pd.Series(geoms, index=obs.index, dtype=object)
        return
    values = [g.wkt if hasattr(g, "wkt") else g for g in geoms]
    if kind == "object":
        obs["geometry"] = pd.Series(values, index=obs.index, dtype=object)
    elif kind == "str":
        obs["geometry"] = pd.Series(values, index=obs.index, dtype="str")
    else:
        raise ValueError(kind)


def geoms_of(agg):
    return [from_wkt(v) if isinstance(v, str) else v for v in agg.adata.obs["geometry"]]


def result(agg, target_name="colonies"):
    name = f"viz_{target_name}"
    obs = agg.adata.obs
    return obs[f"dist_to_{name}"].copy(), obs[f"nearest_{name}"].copy()


def assert_same_result(res1, res2, check_dtype=True):
    (d1, n1), (d2, n2) = res1, res2
    assert np.array_equal(d1.to_numpy(), d2.to_numpy(), equal_nan=True), "distances differ"
    assert list(n1.index) == list(n2.index)
    assert [None if pd.isna(x) else x for x in n1] == [None if pd.isna(x) else x for x in n2], "nearest ids differ"
    if check_dtype:
        assert d1.dtype == d2.dtype, (d1.dtype, d2.dtype)
        assert n1.dtype == n2.dtype, (n1.dtype, n2.dtype)


def run_orig(viz):
    orig_agg_compute_distances(viz.agg["SC"].analysis, "colonies")
    return result(viz.agg["SC"])


def run_fix(viz):
    viz.agg["SC"].analysis.compute_distances("colonies")
    return result(viz.agg["SC"])


def reference_result():
    '''Result of the 1.0.5 implementation on object dtype WKT (what pandas < 3 produced)'''
    return run_orig(make_viz("object"))


# ---------- tests ----------

def test_A_str_dtype_bug_reproduced_and_fixed():
    viz = make_viz("str")
    assert viz.agg["SC"].adata.obs["geometry"].dtype != object
    assert pd.api.types.is_string_dtype(viz.agg["SC"].adata.obs["geometry"])
    try:
        run_orig(viz)
    except TypeError:
        pass
    else:
        raise AssertionError("1.0.5 implementation was expected to raise TypeError on str dtype")
    dist, nearest = run_fix(viz)
    assert dist.notna().all() and nearest.notna().all()


def test_B_object_dtype_identical_to_orig():
    ref = reference_result()
    fix = run_fix(make_viz("object"))
    assert_same_result(fix, ref)
    assert fix[0].dtype == np.float64
    assert_same_result(run_fix(make_viz("str")), ref)


def test_C_shapely_geometries_identical_to_orig():
    assert_same_result(run_fix(make_viz("shapely")), run_orig(make_viz("shapely")))
    assert_same_result(run_fix(make_viz("shapely")), reference_result())


def test_D_missing_current_geometry():
    ref = reference_result()
    missing = [0, 5, 61]
    for kind in ["object", "str"]:
        viz = make_viz(kind)
        geoms = geoms_of(viz.agg["SC"])
        geoms[0], geoms[5], geoms[61] = np.nan, None, np.nan
        set_geometry(viz.agg["SC"], geoms, kind)
        try:
            run_orig(viz)
        except (TypeError, AttributeError):
            pass
        else:
            raise AssertionError(f"1.0.5 implementation was expected to crash on missing geometry ({kind})")
        dist, nearest = run_fix(viz)
        assert dist.dtype == np.float64
        assert dist.iloc[missing].isna().all() and nearest.iloc[missing].isna().all()
        keep = [i for i in range(len(dist)) if i not in missing]
        assert_same_result((dist.iloc[keep], nearest.iloc[keep]), (ref[0].iloc[keep], ref[1].iloc[keep]), check_dtype=False)


def test_D_empty_current_geometry():
    viz = make_viz("str")
    geoms = geoms_of(viz.agg["SC"])
    geoms[3] = "POLYGON EMPTY"
    geoms[4] = ""
    set_geometry(viz.agg["SC"], geoms, "str")
    dist, nearest = run_fix(viz)
    assert dist.iloc[[3, 4]].isna().all() and nearest.iloc[[3, 4]].isna().all()
    assert dist.drop(dist.index[[3, 4]]).notna().all()


def test_E_missing_target_geometry():
    for kind in ["object", "str"]:
        viz = make_viz(kind)
        targets = geoms_of(viz.agg["colonies"])
        targets[4] = np.nan # central colony
        set_geometry(viz.agg["colonies"], targets, kind)
        dist, nearest = run_fix(viz)
        assert "colony4" not in set(nearest)
        target_ids = list(viz.agg["colonies"].adata.obs.index)
        for cell, d, n in zip(geoms_of(viz.agg["SC"]), dist, nearest):
            expected = min(cell.distance(t) for t in targets if not pd.isna(t))
            assert d == expected
            assert cell.distance(targets[target_ids.index(n)]) == d


def test_E_target_without_any_geometry_raises():
    for kind in ["object", "str"]:
        viz = make_viz(kind)
        set_geometry(viz.agg["colonies"], [np.nan] * viz.agg["colonies"].shape[0], kind)
        try:
            run_fix(viz)
        except ValueError as e:
            assert "No geometry" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_F_call_variants():
    ref = reference_result()
    viz = make_viz("str")
    sc_agg = viz.agg["SC"]
    sc_agg.analysis.compute_distances("colonies")
    assert {"dist_to_viz_colonies", "nearest_viz_colonies"} <= set(sc_agg.adata.obs.columns)
    assert_same_result(result(sc_agg), ref)
    for call in [lambda: sc_agg.analysis.compute_distances(viz.agg["colonies"]),
                 lambda: sc_agg.analysis.compute_distances(target_agg="colonies"),
                 lambda: sc_agg.analysis.compute_distances("colonies", None, None)]:
        del sc_agg.adata.obs["dist_to_viz_colonies"], sc_agg.adata.obs["nearest_viz_colonies"]
        assert call() is None
        assert_same_result(result(sc_agg), ref)
    sc_agg.analysis.compute_distances("colonies", "d", "n")
    sc_agg.analysis.compute_distances("colonies", dist_col_name="d2", nearest_col_name="n2")
    obs = sc_agg.adata.obs
    assert_same_result((obs["d"], obs["n"]), ref)
    assert_same_result((obs["d2"], obs["n2"]), ref)


def test_F_self_target_same_as_orig():
    viz_orig, viz_fix = make_viz("object"), make_viz("object")
    orig_agg_compute_distances(viz_orig.agg["SC"].analysis, "SC")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        viz_fix.agg["SC"].analysis.compute_distances("SC")
    assert any("same Aggregation" in str(w.message) for w in caught)
    assert_same_result(result(viz_fix.agg["SC"], "SC"), result(viz_orig.agg["SC"], "SC"))
    assert (result(viz_fix.agg["SC"], "SC")[0] == 0).all()


def test_G_stored_geometry_untouched():
    for kind in ["object", "str", "shapely"]:
        viz = make_viz(kind)
        before = {name: viz.agg[name].adata.obs["geometry"].copy() for name in ["SC", "colonies"]}
        run_fix(viz)
        for name, series in before.items():
            after = viz.agg[name].adata.obs["geometry"]
            assert after.dtype == series.dtype
            assert after.equals(series)


def test_H_dill_round_trip():
    ref = reference_result()
    for kind in ["object", "str"]:
        viz = make_viz(kind)
        path = viz.save(os.path.join(viz.path_output, "viz.pkl"))
        loaded = HiVis.load(path)
        assert loaded.agg["SC"].adata.obs["geometry"].dtype == viz.agg["SC"].adata.obs["geometry"].dtype
        assert_same_result(run_fix(loaded), ref)


def test_I_infer_string_disabled():
    ref = reference_result()
    with pd.option_context("future.infer_string", False):
        viz = make_viz("object")
        geoms = [g.wkt for g in geoms_of(viz.agg["SC"])]
        viz.agg["SC"].adata.obs["geometry"] = geoms # inferred dtype under the option
        assert viz.agg["SC"].adata.obs["geometry"].dtype == object
        fix = run_fix(viz)
    assert_same_result(fix, ref, check_dtype=False)


def test_J_write_h5ad():
    for kind in ["object", "str"]:
        viz = make_viz(kind)
        geoms = geoms_of(viz.agg["SC"])
        geoms[0] = np.nan
        set_geometry(viz.agg["SC"], geoms, kind)
        run_fix(viz)
        path = os.path.join(viz.path_output, "sc.h5ad")
        viz.agg["SC"].adata.write_h5ad(path)
        back = ad.read_h5ad(path)
        assert np.array_equal(back.obs["dist_to_viz_colonies"].to_numpy(),
                              viz.agg["SC"].adata.obs["dist_to_viz_colonies"].to_numpy(), equal_nan=True)


def test_K_visium_compute_distances():
    viz_ref = make_viz("object")
    orig_visium_compute_distances(viz_ref.analysis, "colonies")
    ref = (viz_ref.adata.obs["dist_to_colonies"], viz_ref.adata.obs["nearest_colonies"])
    for kind in ["object", "str", "shapely"]:
        viz = make_viz(kind)
        viz.analysis.compute_distances("colonies")
        assert_same_result((viz.adata.obs["dist_to_colonies"], viz.adata.obs["nearest_colonies"]), ref)
    # 1.0.5 also worked with str dtype
    viz = make_viz("str")
    orig_visium_compute_distances(viz.analysis, "colonies")
    assert_same_result((viz.adata.obs["dist_to_colonies"], viz.adata.obs["nearest_colonies"]), ref)


def test_K_visium_missing_target_geometry():
    viz = make_viz("str")
    targets = geoms_of(viz.agg["colonies"])
    targets[4] = np.nan
    set_geometry(viz.agg["colonies"], targets, "str")
    viz.analysis.compute_distances("colonies")
    obs = viz.adata.obs
    assert "colony4" not in set(obs["nearest_colonies"])
    valid = [t for t in targets if not pd.isna(t)]
    for x, y, d in zip(obs["um_x"], obs["um_y"], obs["dist_to_colonies"]):
        assert d == min(Point(x, y).distance(t) for t in valid)
    set_geometry(viz.agg["colonies"], [np.nan] * len(targets), "str")
    try:
        viz.analysis.compute_distances("colonies")
    except ValueError as e:
        assert "No geometry" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ---------- runner ----------

def main():
    import traceback
    tests = [(name, func) for name, func in globals().items() if name.startswith("test_") and callable(func)]
    failed = 0
    try:
        for name, func in tests:
            try:
                func()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    finally:
        for path in TMP_DIRS:
            shutil.rmtree(path, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
