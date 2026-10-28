# -*- coding: utf-8 -*-
"""
Synthetic tests for where HiVis 1.1.0 writes files:
    * an Aggregation no longer makes an empty folder in path_output when it is created (also by subset, crop, copy).
      The folder (same path as in 1.0.6) is made when a plot or data is saved into it
    * every save (plots, tables, h5ad, mat, pickle) makes the folder it writes into, if missing
    * a new HiVis object saves its QC plot as PNG only. qc(save=True) still writes pdf, svg and png

Run with pytest, or directly: python tests/test_output_folders.py
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import inspect
import shutil
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from shapely.geometry import box

from HiVis import HiVis

TMP_DIRS = []


# ---------- helpers ----------

def make_viz(plot_qc=False):
    '''Small HiVis with an annotation column on the bins and one aggregation with geometry'''
    path_output = tempfile.mkdtemp(prefix="hivis_test_")
    TMP_DIRS.append(path_output)
    n = 20
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(n) * 10.0, np.arange(n) * 10.0)
    rows = np.repeat(np.arange(n), n)
    obs = pd.DataFrame({"pxl_col_in_fullres": xs.ravel(), "pxl_row_in_fullres": ys.ravel(),
                        "nUMI": rng.poisson(5, n * n).astype(float),
                        "vessel_id": np.where(rows < 10, "v1", "v2"),
                        "agg_id": np.where(rows < 10, "a1", "a2")}, index=[f"bin{i}" for i in range(n * n)])
    adata = ad.AnnData(X=sparse.csr_matrix(rng.poisson(1, (n * n, 3)).astype(np.float32)), obs=obs,
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    json = {"microns_per_pixel": 0.5, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
            "spot_diameter_fullres": 4, "bin_size_um": 5}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # plt.show() with the Agg backend
        viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name="viz",
                          path_output=path_output, plot_qc=plot_qc)
    plt.close("all")
    agg_obs = pd.DataFrame({"pxl_col_in_fullres": [10.0, 150.0], "pxl_row_in_fullres": [10.0, 150.0],
                            "score": [1.0, 2.0]}, index=pd.Index(["a1", "a2"], name="agg_id"))
    agg_obs["geometry"] = [box(x * 0.5 - 2, y * 0.5 - 2, x * 0.5 + 2, y * 0.5 + 2).wkt
                           for x, y in zip(agg_obs["pxl_col_in_fullres"], agg_obs["pxl_row_in_fullres"])]
    viz.add_agg(ad.AnnData(X=sparse.csr_matrix(np.ones((2, 3), dtype=np.float32)), obs=agg_obs,
                           var=pd.DataFrame(index=["g0", "g1", "g2"])), "SC")
    return viz


def dirs_in(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def files_in(path):
    return sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))


# ---------- O26: no empty folders ----------

def test_A_no_folders_when_creating_aggregations():
    viz = make_viz()
    agg = viz.agg["SC"]
    viz.agg_from_annotations("vessel_id", "vessels")
    sub = viz.subset((viz["vessel_id"] == "v1", slice(None)))
    viz.crop(xlim=[0, 60], ylim=[0, 60])
    agg.copy("copied")
    agg[agg.adata.obs.index == "a1", :]
    agg + viz.agg["vessels"]
    sub.agg["SC"].plot.spatial("score") # plotting without saving writes nothing
    plt.close("all")
    assert dirs_in(viz.path_output) == [], dirs_in(viz.path_output) # 1.0.6: viz_SC, viz_vessels, viz_subset_SC, ...
    assert agg.path_output == f"{viz.path_output}/viz_SC" # the same path as in 1.0.6


def test_B_plot_save_makes_the_folder():
    viz = make_viz()
    agg = viz.agg["SC"]
    agg.plot.cells("score")
    path = agg.plot.save("cells")
    plt.close("all")
    assert path == f"{viz.path_output}/viz_SC/viz_SC_cells"
    assert files_in(agg.path_output) == ["viz_SC_cells.pdf", "viz_SC_cells.png", "viz_SC_cells.svg"]
    agg.plot.save("table", fig=agg.adata.obs) # a DataFrame is saved as csv
    assert "viz_SC_table.csv" in files_in(agg.path_output)
    agg.plot.hist("score", save=True)
    plt.close("all")
    assert "viz_SC_score_HIST.png" in files_in(agg.path_output)


def test_C_exports_make_the_folder():
    viz = make_viz()
    agg = viz.agg["SC"]
    path = agg.export_h5()
    assert os.path.isfile(path) and os.path.dirname(path) == agg.path_output
    path = agg.export_to_matlab()
    assert os.path.isfile(path)
    other = os.path.join(viz.path_output, "new", "deeper", "agg.h5ad")
    assert agg.export_h5(other) == other and os.path.isfile(other)


def test_D_hivis_saves_make_the_folder():
    viz = make_viz()
    shutil.rmtree(viz.path_output) # e.g. a pickle loaded on another computer
    viz.plot.spatial()
    viz.plot.save("image", formats="png")
    plt.close("all")
    assert files_in(viz.path_output) == ["viz_image.png"]
    path = viz.save()
    assert os.path.isfile(path) and HiVis.load(path).name == "viz"
    assert os.path.isfile(viz.export_h5())
    viz.rename("renamed", new_out_path=True, full=True) # path_output becomes a subfolder
    viz.plot.spatial()
    viz.plot.save("image", formats="png")
    plt.close("all")
    assert files_in(viz.path_output) == ["renamed_image.png"]


def test_E_existing_folder_is_kept():
    '''Folders that already exist (made by 1.0.6, or by an earlier save) are used as they are'''
    viz = make_viz()
    agg = viz.agg["SC"]
    os.makedirs(agg.path_output)
    with open(os.path.join(agg.path_output, "mine.txt"), "w") as f:
        f.write("keep")
    agg.plot.cells("score")
    agg.plot.save("cells", formats="png")
    plt.close("all")
    assert files_in(agg.path_output) == ["mine.txt", "viz_SC_cells.png"]


# ---------- QC plot of a new object ----------

def test_F_new_object_saves_qc_png_only():
    viz = make_viz(plot_qc=True)
    assert files_in(viz.path_output) == ["viz_QC.png"] # 1.0.6: viz_QC.pdf, viz_QC.png, viz_QC.svg


def test_G_qc_saves_as_before():
    assert "formats" not in inspect.signature(HiVis.HiVis_analysis.AnalysisVisium.qc).parameters
    viz = make_viz()
    viz.analysis.qc()
    plt.close("all")
    assert files_in(viz.path_output) == []
    viz.analysis.qc(save=True)
    plt.close("all")
    assert files_in(viz.path_output) == ["viz_QC.pdf", "viz_QC.png", "viz_QC.svg"]


def test_H_qc_png_same_as_qc_save():
    '''The PNG of a new object is the same image that qc(save=True) writes'''
    viz = make_viz(plot_qc=True)
    with open(os.path.join(viz.path_output, "viz_QC.png"), "rb") as f:
        at_creation = f.read()
    viz.analysis.qc(save=True)
    plt.close("all")
    with open(os.path.join(viz.path_output, "viz_QC.png"), "rb") as f:
        assert f.read() == at_creation


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
