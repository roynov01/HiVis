# -*- coding: utf-8 -*-
"""
Synthetic tests for the axis labels of PlotAgg.cells. geopandas >= 1.2 labels the axes "x"/"y" whenever it draws
shapes, so plot.cells(axis_labels=False) showed "x"/"y", and axis_labels=True showed "x"/"y" instead of
"Spatial 1 (µm)". The labels after plot.cells must be the ones plot.spatial sets, as with geopandas < 1.2.

Run with pytest, or directly: python tests/test_plot_cells_labels.py
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import geopandas as gpd
from packaging.version import Version
from scipy import sparse
from shapely.geometry import box

from HiVis import HiVis

MPP = 0.5 # microns per pixel
TMP_DIRS = []


def make_viz():
    '''Small HiVis with an aggregation "objects" that has square geometries (WKT, in µm) and two obs to color by'''
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
            "spot_diameter_fullres": 4, "bin_size_um": 2.0}
    viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name="viz",
                      path_output=path_output, plot_qc=False)
    centers_pxl = [(30.0, 30.0), (120.0, 40.0), (60.0, 150.0)]
    agg_obs = pd.DataFrame({"pxl_col_in_fullres": [c[0] for c in centers_pxl],
                            "pxl_row_in_fullres": [c[1] for c in centers_pxl],
                            "group": ["a", "b", "a"], "score": [1.0, 2.0, 3.0],
                            "geometry": [box((x - 10) * MPP, (y - 10) * MPP, (x + 10) * MPP, (y + 10) * MPP).wkt
                                         for x, y in centers_pxl]},
                           index=pd.Index(["o1", "o2", "o3"], name="object_id"))
    viz.add_agg(ad.AnnData(X=sparse.csr_matrix(np.ones((3, 3), dtype=np.float32)), obs=agg_obs,
                           var=pd.DataFrame(index=["g0", "g1", "g2"])), "objects")
    return viz


def labels(ax):
    return ax.get_xlabel(), ax.get_ylabel()


def spatial_labels(viz, axis_labels):
    fig, ax = plt.subplots()
    viz.plot.spatial(ax=ax, axis_labels=axis_labels)
    out = labels(ax)
    plt.close(fig)
    return out


def cells_labels(viz, what, axis_labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    viz.agg["objects"].plot.cells(what, ax=ax, axis_labels=axis_labels)
    out = labels(ax)
    plt.close(ax.get_figure())
    return out


def test_A_geopandas_labels_axes():
    '''Documents the cause: geopandas >= 1.2 sets "x"/"y" axis labels when plotting'''
    fig, ax = plt.subplots()
    gpd.GeoSeries([box(0, 0, 1, 1)]).boundary.plot(ax=ax)
    if Version(gpd.__version__) >= Version("1.2"):
        assert labels(ax) == ("x", "y")
    plt.close(fig)


def test_B_no_labels_when_axis_labels_false():
    viz = make_viz()
    for what in [None, "group", "score"]:
        assert cells_labels(viz, what, axis_labels=False) == spatial_labels(viz, False) == ("", ""), what


def test_C_spatial_labels_when_axis_labels_true():
    viz = make_viz()
    for what in [None, "group", "score"]:
        assert cells_labels(viz, what, axis_labels=True) == spatial_labels(viz, True) == ("Spatial 1 (µm)", "Spatial 2 (µm)"), what


def test_D_labels_of_a_passed_ax_are_kept():
    '''As with geopandas < 1.2: plot.cells(axis_labels=False) does not touch labels already on the ax'''
    viz = make_viz()
    fig, ax = plt.subplots()
    ax.set_xlabel("mine x")
    ax.set_ylabel("mine y")
    assert cells_labels(viz, "group", axis_labels=False, ax=ax) == ("mine x", "mine y")


def test_E_plot_otherwise_unchanged():
    '''Same artists as before the fix: outlines + one patch collection per category, legend, no ticks'''
    viz = make_viz()
    fig, ax = plt.subplots()
    viz.agg["objects"].plot.cells("group", ax=ax, legend="upper right", legend_title="")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]
    assert len(ax.collections) == 3 # boundary + "a" + "b"
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []
    plt.close(fig)


def teardown_module(module=None):
    for d in TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    print("geopandas", gpd.__version__)
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
