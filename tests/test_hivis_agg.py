# -*- coding: utf-8 -*-
"""
Synthetic tests for the 1.1.0 fixes in HiVis and Aggregation. Compares the current implementation with the one
from HiVis 1.0.6:
    * HiVis.subset named a subset of a subset "" (empty name)
    * Aggregation.copy deep-copied twice, and both copies included the whole HiVis object (images, adata)
    * agg_from_annotations / new_adata added keys to the caller's obs2agg list or dict
    * agg_cells could not import cells exported as QuPath detections, and found out only after the slow aggregation
    * update() didn't reload HiVis_utils, Aggregation_utils and other_utils
    * update_meta no longer uses is_categorical_dtype (deprecated in pandas)

Run with pytest, or directly: python tests/test_hivis_agg.py
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import shutil
import tempfile
import warnings
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import geopandas as gpd
from scipy import sparse
from shapely.geometry import box

from HiVis import HiVis, Aggregation_utils, HiVis_utils, other_utils
from HiVis.Aggregation import Aggregation

TMP_DIRS = []
N = 20 # bins per side
# 4 cells, each covering a block of 4x4 bins. Bins are 10 pixels apart
CELLS = {"cA": (0, 0), "cB": (0, 8), "cC": (8, 4), "cD": (12, 12)} # (first bin row, first bin col)


# ---------- HiVis 1.0.6 implementations, kept verbatim for comparison ----------

def orig_copy(self, new_name=None, new_out_path=False, full=False):
    '''Aggregation.copy from HiVis 1.0.6'''
    new = deepcopy(self)
    new.viz = self.viz
    new = deepcopy(self)
    if new_name:
        new.rename(new_name, new_out_path=new_out_path, full=full)
    return new


def orig_new_adata(adata, aggregate_by, aggregation_func, obs2agg=None, **aggregation_kwargs):
    '''Aggregation_utils.new_adata from HiVis 1.0.6'''
    if obs2agg is None:
        obs2agg = {}
    if isinstance(obs2agg, list):
        obs2agg = {k:None for k in set(obs2agg)}
    obs2agg["pxl_row_in_fullres"] = np.mean
    obs2agg["pxl_col_in_fullres"] = np.mean
    obs2agg["nUMI"] = np.sum

    meta_df, meta_ids = Aggregation_utils._aggregate_meta(adata=adata,aggregate_by=aggregate_by,custom_agg=obs2agg)

    expr_data, expr_ids, layers, other_results = aggregation_func(adata, group_col=aggregate_by,**aggregation_kwargs)

    meta_df = meta_df.reindex(expr_ids)

    adata_agg = ad.AnnData(X=expr_data,obs=meta_df,var=pd.DataFrame(index=adata.var_names))

    if layers:
        for layer_name, layer_data in layers.items():
            adata_agg.layers[layer_name] = layer_data

    return adata_agg, other_results


def orig_agg_from_annotations(self, annotation_id_col, name="SC", obs2agg=None, geojson_path=None):
    '''HiVis.agg_from_annotations from HiVis 1.0.6'''
    aggregation_func = Aggregation_utils._aggregate_data_annotations

    annotation_col = annotation_id_col.replace("_id","")
    if annotation_col != annotation_id_col and annotation_col in self.adata.obs.columns: # Add annotation class to each object
        if obs2agg is not None:
            if annotation_col not in obs2agg:
                if isinstance(obs2agg, list):
                    obs2agg += [annotation_col]
                else:
                    obs2agg[annotation_col] = None
        else:
            obs2agg = [annotation_col]

    adata_agg, _ = orig_new_adata(self.adata, annotation_id_col, aggregation_func,obs2agg=obs2agg)

    self.add_agg(adata_agg, name=name)
    self.agg[name].adata.obs[f"{annotation_col}_col"] = self.agg[name].adata.obs.index
    if geojson_path:
        self.agg[name].import_geometry(geojson_path, object_type="annotation")


def orig_agg_cells(self, input_df, name="SC", obs2add=None, obs2agg=None, geojson_path=None):
    '''HiVis.agg_cells from HiVis 1.0.6'''
    id_col = f"Cell_ID_{name}"
    input_df.rename(columns={"Object ID":"Cell_ID"}, inplace=True)
    spots_only, cells_only = Aggregation_utils.split_spots_cells(input_df)

    spots_only = spots_only.rename(columns={"Cell_ID": id_col})
    cells_only.index.name = id_col

    self.adata.obs = self.adata.obs.drop(columns=[id_col], errors='ignore')
    overlap = self.adata.obs.columns.intersection(spots_only.columns)
    overlap = overlap.difference([id_col])
    if len(overlap) > 0:
        self.adata.obs = self.adata.obs.drop(columns=list(overlap))
        print(f"Dropping overlapping columns from obs: {list(overlap)}")
    self.adata.obs = self.adata.obs.join(spots_only,how="left")

    aggregation_func = Aggregation_utils._aggregate_data_cells

    adata_agg, _ = orig_new_adata(self.adata, id_col, aggregation_func,obs2agg=obs2agg)

    if obs2add:
        obs2add = [col for col in cells_only.columns if col in obs2add]
        Aggregation_utils.merge_cells(cells_only, adata_agg, obs2add, id_col=id_col)

    self.add_agg(adata_agg, name=name)
    if geojson_path:
        self.agg[name].import_geometry(geojson_path)


# ---------- helpers ----------

def bin_index(r, c):
    return r * N + c


def make_viz(name="viz"):
    '''Small HiVis with two annotation sets on the bins (vessels and colonies) and one aggregation'''
    path_output = tempfile.mkdtemp(prefix="hivis_test_")
    TMP_DIRS.append(path_output)
    xs, ys = np.meshgrid(np.arange(N) * 10.0, np.arange(N) * 10.0)
    rng = np.random.default_rng(0)
    n = N * N
    rows, cols = np.divmod(np.arange(n), N)
    vessel_id = np.where(cols < 10, np.where(rows < 10, "v1", "v2"), None)
    colony_id = np.where(rows % 5 == 0, np.where(cols < 10, "k1", "k2"), None)
    obs = pd.DataFrame({"pxl_col_in_fullres": xs.ravel(), "pxl_row_in_fullres": ys.ravel(),
                        "nUMI": rng.poisson(5, n).astype(float),
                        "value": rng.uniform(0, 1, n),
                        "tissue": pd.Categorical(np.where(rows < 10, "top", "bottom")),
                        "vessel_id": vessel_id, "vessel": np.where(vessel_id == "v1", "artery", np.where(vessel_id == "v2", "vein", None)),
                        "colony_id": colony_id, "colony": np.where(colony_id == None, None, "colony"),
                        "agg_id": np.where(rows < 10, "a1", np.where(cols < 10, "a2", "a3"))},
                       index=[f"bin{i}" for i in range(n)])
    adata = ad.AnnData(X=sparse.csr_matrix(rng.poisson(1, (n, 3)).astype(np.float32)), obs=obs,
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    json = {"microns_per_pixel": 0.5, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
            "spot_diameter_fullres": 4, "bin_size_um": 5}
    viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name=name,
                      path_output=path_output, plot_qc=False)
    agg_obs = pd.DataFrame({"pxl_col_in_fullres": [10.0, 50.0, 120.0], "pxl_row_in_fullres": [10.0, 50.0, 150.0],
                            "score": [1.0, 2.0, 3.0], "cls": pd.Categorical(["a", "b", "a"])},
                           index=pd.Index(["a1", "a2", "a3"], name="agg_id"))
    agg_obs["geometry"] = [box(x * 0.5 - 2, y * 0.5 - 2, x * 0.5 + 2, y * 0.5 + 2).wkt
                           for x, y in zip(agg_obs["pxl_col_in_fullres"], agg_obs["pxl_row_in_fullres"])]
    viz.add_agg(ad.AnnData(X=sparse.csr_matrix(rng.poisson(3, (3, 3)).astype(np.float32)), obs=agg_obs,
                           var=pd.DataFrame(index=["g0", "g1", "g2"])), "SC")
    return viz


def qupath_csv():
    '''Like the QuPath csv export: one row per tile (bin), Name "<bin>__<cell>", then one row per cell'''
    cell_of_bin = {}
    for cell, (r0, c0) in CELLS.items():
        for r in range(r0, r0 + 4):
            for c in range(c0, c0 + 4):
                cell_of_bin[f"bin{bin_index(r, c)}"] = cell
    tiles = pd.DataFrame({"Name": [f"bin{i}__{cell_of_bin[f'bin{i}']}" if f"bin{i}" in cell_of_bin else f"bin{i}"
                                   for i in range(N * N)],
                          "Object type": "Tile", "Object ID": [f"t{i}" for i in range(N * N)],
                          "InNuc": [i % 2 == 0 for i in range(N * N)], "Area µm^2": np.nan})
    cells = pd.DataFrame({"Name": list(CELLS), "Object type": "Cell", "Object ID": list(CELLS),
                          "InNuc": np.nan, "Area µm^2": [10.0, 20.0, 30.0, 40.0]})
    return pd.concat([tiles, cells], ignore_index=True)


def qupath_geojson(object_type):
    '''The cells' outlines (in fullres pixels), as exported from QuPath'''
    path = os.path.join(tempfile.mkdtemp(prefix="hivis_test_"), f"cells_{object_type}.geojson")
    TMP_DIRS.append(os.path.dirname(path))
    geoms = [box(c0 * 10 - 5, r0 * 10 - 5, (c0 + 4) * 10 - 5, (r0 + 4) * 10 - 5) for r0, c0 in CELLS.values()]
    gdf = gpd.GeoDataFrame({"id": list(CELLS), "objectType": object_type}, geometry=geoms)
    tissue = gpd.GeoDataFrame({"id": ["tissue1"], "objectType": ["annotation"]}, geometry=[box(0, 0, 200, 200)])
    pd.concat([gdf, tissue], ignore_index=True).to_file(path, driver="GeoJSON")
    return path


def assert_agg_equal(a, b):
    pd.testing.assert_frame_equal(a.adata.obs, b.adata.obs)
    pd.testing.assert_frame_equal(a.adata.var, b.adata.var)
    assert np.array_equal(a.adata.X.toarray(), b.adata.X.toarray())
    assert a.name == b.name
    assert os.path.relpath(a.path_output, a.viz.path_output) == os.path.relpath(b.path_output, b.viz.path_output)


# ---------- O3: subset name ----------

def test_A_subset_names():
    viz = make_viz()
    sub = viz.subset((viz["vessel_id"] == "v1", slice(None)))
    assert sub.name == "viz_subset" # unchanged
    subsub = sub.subset((sub["value"] > 0.5, slice(None)))
    assert subsub.name == "viz_subset", repr(subsub.name) # was ""
    assert subsub.agg["SC"].name == "viz_subset_SC"
    named = make_viz("liver_subset")
    assert named.subset((named["value"] > 0.5, slice(None))).name == "liver_subset"


# ---------- O4: Aggregation.copy ----------

def test_B_copy_shares_hivis():
    viz = make_viz()
    agg = viz.agg["SC"]
    new = agg.copy()
    assert new.viz is viz, "the copy must stay linked to the same HiVis"
    assert new.viz.image_fullres is viz.image_fullres
    assert new.plot.main is new and new.analysis.main is new
    assert new.adata is not agg.adata
    assert viz.agg["SC"] is agg, "the HiVis must still hold the original"


def test_C_copy_same_data_as_orig():
    viz = make_viz()
    agg = viz.agg["SC"]
    agg.plot.cells("score") # also copies plot state (geometry, adata_cropped)
    plt.close("all")
    for kwargs in [{}, {"new_name": "copy2"}, {"new_name": "copy3", "new_out_path": True},
                   {"new_name": "full_name", "full": True}]:
        o = orig_copy(agg, **kwargs)
        n = agg.copy(**kwargs)
        assert_agg_equal(o, n)
        pd.testing.assert_frame_equal(pd.DataFrame(o.adata_cropped.obs), pd.DataFrame(n.adata_cropped.obs))
        assert list(o.plot.geometry.index) == list(n.plot.geometry.index)


def test_D_copy_is_independent():
    viz = make_viz()
    agg = viz.agg["SC"]
    before = agg.adata.obs.copy()
    new = agg.copy("other")
    new.adata.obs["score"] = 0.0
    new.adata.X[0, 0] = 99
    new["added"] = [1, 2, 3]
    pd.testing.assert_frame_equal(agg.adata.obs, before)
    assert agg.adata.X[0, 0] != 99
    assert agg.name == "viz_SC" and new.name == "viz_other"


def test_E_copy_plots_and_syncs_to_parent():
    viz = make_viz()
    new = viz.agg["SC"].copy()
    new.plot.cells("score")
    new.plot.spatial("cls", xlim=[0, 60], ylim=[0, 60])
    plt.close("all")
    new.adata.obs["label"] = ["x", "y", "z"]
    new.sync("label") # was written into a hidden copy of the HiVis in 1.0.6
    assert set(viz.adata.obs["label"]) == {"x", "y", "z"}


# ---------- O8: obs2agg is not modified ----------

def test_F_obs2agg_not_modified():
    for obs2agg in [["value"], {"value": np.median}, {"value": [np.mean, np.max], "tissue": None}]:
        viz = make_viz()
        before = deepcopy(obs2agg)
        viz.agg_from_annotations("vessel_id", "vessels", obs2agg=obs2agg)
        assert obs2agg == before, obs2agg
        viz.agg_cells(qupath_csv(), name="cells", obs2agg=obs2agg)
        assert obs2agg == before, obs2agg


def test_G_same_aggregation_as_orig():
    '''A single call gives the same aggregation as 1.0.6'''
    for obs2agg in [None, ["value", "tissue"], {"value": np.median}, {"value": [np.mean, np.max], "vessel": None}]:
        for id_col in ["vessel_id", "colony_id"]:
            viz_o, viz_n = make_viz(), make_viz()
            orig_agg_from_annotations(viz_o, id_col, "ann", obs2agg=deepcopy(obs2agg))
            viz_n.agg_from_annotations(id_col, "ann", obs2agg=deepcopy(obs2agg))
            o, n = viz_o.agg["ann"], viz_n.agg["ann"]
            pd.testing.assert_frame_equal(o.adata.obs, n.adata.obs)
            assert np.array_equal(o.adata.X.toarray(), n.adata.X.toarray())


def test_H_reused_obs2agg_no_longer_accumulates():
    '''1.0.6: a dict reused for two annotation sets also aggregated the first set's class column into the second'''
    shared = {"value": np.mean}
    viz_o = make_viz()
    orig_agg_from_annotations(viz_o, "vessel_id", "vessels", obs2agg=shared)
    orig_agg_from_annotations(viz_o, "colony_id", "colonies", obs2agg=shared)
    assert "vessel" in viz_o.agg["colonies"].adata.obs.columns # the leak

    shared = {"value": np.mean}
    viz_n = make_viz()
    viz_n.agg_from_annotations("vessel_id", "vessels", obs2agg=shared)
    viz_n.agg_from_annotations("colony_id", "colonies", obs2agg=shared)
    assert "vessel" not in viz_n.agg["colonies"].adata.obs.columns
    common = [c for c in viz_n.agg["colonies"].adata.obs.columns]
    pd.testing.assert_frame_equal(viz_o.agg["colonies"].adata.obs[common], viz_n.agg["colonies"].adata.obs)


# ---------- O19: agg_cells with detections ----------

def test_I_agg_cells_same_as_orig():
    '''Without object_type, agg_cells gives the same aggregation and geometry as 1.0.6'''
    path = qupath_geojson("cell")
    for geojson_path in [None, path]:
        viz_o, viz_n = make_viz(), make_viz()
        csv_o, csv_n = qupath_csv(), qupath_csv()
        orig_agg_cells(viz_o, csv_o, name="cells", obs2add=["Area µm^2"], obs2agg=["value"], geojson_path=geojson_path)
        viz_n.agg_cells(csv_n, name="cells", obs2add=["Area µm^2"], obs2agg=["value"], geojson_path=geojson_path)
        assert_agg_equal(viz_o.agg["cells"], viz_n.agg["cells"])
        pd.testing.assert_frame_equal(viz_o.adata.obs, viz_n.adata.obs)
        pd.testing.assert_frame_equal(csv_o, csv_n)
        if geojson_path:
            assert viz_n.agg["cells"].adata.obs["geometry"].notna().all()


def test_J_orig_detections_fail_after_aggregation():
    '''Documents 1.0.6: the error came after the slow aggregation, leaving an aggregation without geometry'''
    viz = make_viz()
    try:
        orig_agg_cells(viz, qupath_csv(), name="cells", geojson_path=qupath_geojson("detection"))
        raised = False
    except ValueError:
        raised = True
    assert raised and "cells" in viz.agg and "geometry" not in viz.agg["cells"].adata.obs


def test_K_detections_fail_before_aggregation():
    viz = make_viz()
    obs_before, csv = viz.adata.obs.copy(), qupath_csv()
    csv_before = csv.copy()
    try:
        viz.agg_cells(csv, name="cells", geojson_path=qupath_geojson("detection"))
        message = None
    except ValueError as e:
        message = str(e)
    assert message and "detection" in message, message # tells which objectType values exist
    assert "cells" not in viz.agg
    pd.testing.assert_frame_equal(viz.adata.obs, obs_before)
    pd.testing.assert_frame_equal(csv, csv_before)


def test_L_detections_with_object_type():
    '''object_type="detection" gives the same result as 1.0.6 with the same shapes exported as cells'''
    viz_o, viz_n = make_viz(), make_viz()
    orig_agg_cells(viz_o, qupath_csv(), name="cells", geojson_path=qupath_geojson("cell"))
    viz_n.agg_cells(qupath_csv(), name="cells", geojson_path=qupath_geojson("detection"), object_type="detection")
    assert_agg_equal(viz_o.agg["cells"], viz_n.agg["cells"])


def test_M_import_geometry_message():
    viz = make_viz()
    try:
        viz.agg["SC"].import_geometry(qupath_geojson("detection"))
        message = None
    except ValueError as e:
        message = str(e)
    assert message.startswith("No entries with object_type=cell") and "detection" in message, message


# ---------- O15: update_meta keeps categorical columns ----------

def test_N_update_meta_categorical():
    viz = make_viz()
    agg = viz.agg["SC"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        viz.update_meta("tissue", {"top": "upper"})
        agg.update_meta("cls", {"a": "A"})
        agg.update_meta("score", {1.0: 10.0})
    assert not [x for x in w if "is_categorical_dtype" in str(x.message)]
    assert isinstance(viz.adata.obs["tissue"].dtype, pd.CategoricalDtype)
    assert set(viz.adata.obs["tissue"]) == {"upper", "bottom"}
    assert isinstance(agg.adata.obs["cls"].dtype, pd.CategoricalDtype)
    assert list(agg.adata.obs["cls"]) == ["A", "b", "A"]
    assert not isinstance(agg.adata.obs["score"].dtype, pd.CategoricalDtype)
    assert list(agg.adata.obs["score"]) == [10.0, 2.0, 3.0]


# ---------- O17: update() reloads the utils modules ----------

def test_O_update_reloads_utils():
    viz = make_viz()
    HiVis_utils.matnorm = Aggregation_utils.new_adata = other_utils.matnorm = "edited"
    viz.update()
    assert callable(HiVis_utils.matnorm) and callable(Aggregation_utils.new_adata) and callable(other_utils.matnorm)
    HiVis_utils.matnorm = "edited"
    viz.agg["SC"].update()
    assert callable(HiVis_utils.matnorm)
    # still works after the reload
    viz.update(agg=True)
    viz.agg["SC"].plot.cells("score")
    viz.plot.spatial("value")
    plt.close("all")
    viz.agg_from_annotations("vessel_id", "vessels")
    assert viz.agg["vessels"].shape[0] == 2


def teardown_module(module=None):
    for d in TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
    teardown_module()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
