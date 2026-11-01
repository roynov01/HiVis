# -*- coding: utf-8 -*-
"""
Synthetic tests for the plotting fixes after 1.1.0. Compares the current implementation with the one from HiVis 1.1.0:
    * plot.hist of a string or categorical column raised TypeError under pandas 3 (str dtype)
    * a matplotlib Colormap (e.g. ListedColormap) as cmap of numeric values raised UnboundLocalError. Now it is used
      as it is, and draws the same figure as its name
    * scalebar: the documented key length_microns didn't exist (now an alias of length), and a dict passed as scalebar
      got the plot's color added, which then stayed for the next plots that reused it

Run with pytest, or directly: python tests/test_plot_options.py
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
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from shapely.geometry import box

from HiVis import HiVis, HiVis_plot
from HiVis.HiVis_plot import get_colors, DEFAULT_COLOR, plot_scatter_signif

TMP_DIRS = []
PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3


# ---------- HiVis 1.1.0 implementation, kept verbatim for comparison ----------

def orig_plot_histogram(values, bins=10, show_zeroes=False, xlim=None, title=None, figsize=(7,7),
              cmap=None, color="blue", ylab="Count",xlab=None,ax=None):
    '''
    HiVis_plot.plot_histogram from HiVis 1.1.0. Plots histogram from numeric values or barplot for categorical values.
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    if np.issubdtype(values.dtype, np.number):
        if not show_zeroes:
            values = values[values > 0]
        counts, edges, patches = ax.hist(values,bins=bins,color=None if cmap else color)

        if cmap is not None:
            colormap = plt.get_cmap(cmap, len(patches))  # Generate enough colors for all bins (plt.cm.get_cmap was removed in matplotlib>=3.9)
            for i, patch in enumerate(patches):
                patch.set_facecolor(colormap(i))

        if xlim:
            ax.set_xlim(xlim)
        lower, upper = ax.get_xlim()
        relevant_counts = counts[(edges[:-1] >= lower) & (edges[:-1] <= upper)]
        max_count = relevant_counts.max() if len(relevant_counts) > 0 else counts.max()

    # Set ylim a little above the maximum count
        ax.set_ylim([0, max_count * 1.1])
    else: # Categorical case
        unique_vals = pd.Series(values.unique()).sort_values()
        unique_vals = unique_vals.dropna()
        value_counts = values.value_counts().reindex(unique_vals)
        if isinstance(cmap, str):
            colors = get_colors(unique_vals.index, cmap) if cmap else color
        elif isinstance(cmap, list):
            colors = LinearSegmentedColormap.from_list("custom_cmap", cmap)
            colors = [colors(i / (len(unique_vals) - 1)) for i in range(len(unique_vals))]
        else:
            if cmap:
                colors = [cmap.get(val, DEFAULT_COLOR) for val in value_counts.index]
            else:
                colors = color
        value_counts.plot(kind='bar',color=colors, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    return ax


# ---------- helpers ----------

def make_viz(fluorescence=False):
    '''Small HiVis (20x20 bins on a 200x200 pixel image) with an aggregation of 60 cells that have geometry'''
    path_output = tempfile.mkdtemp(prefix="hivis_test_")
    TMP_DIRS.append(path_output)
    n = 20
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(n) * 10.0, np.arange(n) * 10.0)
    rows = np.repeat(np.arange(n), n)
    zone = np.where(rows < 7, "portal", np.where(rows < 14, "mid", "central")).astype(object)
    zone[::9] = np.nan
    obs = pd.DataFrame({"pxl_col_in_fullres": xs.ravel(), "pxl_row_in_fullres": ys.ravel(),
                        "value": rng.uniform(0, 10, n * n), "zone": zone,
                        "tissue": pd.Categorical(np.where(rows < 10, "top", "bottom"))},
                       index=[f"bin{i}" for i in range(n * n)])
    adata = ad.AnnData(X=sparse.csr_matrix(rng.poisson(1, (n * n, 3)).astype(np.float32)), obs=obs,
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    json = {"microns_per_pixel": 0.5, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
            "spot_diameter_fullres": 4, "bin_size_um": 5}
    fluorescence = {"DAPI": "blue", "GFP": "green", "RFP": "red"} if fluorescence else False
    viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name="viz",
                      path_output=path_output, plot_qc=False, fluorescence=fluorescence)
    # cells with expression of 25 genes: g0 drives g1..g4 up and g5..g9 down
    m, n_genes = 60, 25
    base = rng.poisson(20, m).astype(float)
    X = rng.poisson(3, (m, n_genes)).astype(float)
    X[:, 0] += base
    X[:, 1:5] += base[:, None] * rng.uniform(0.5, 1, 4)
    X[:, 5:10] += (base.max() - base)[:, None] * rng.uniform(0.5, 1, 5)
    px, py = rng.uniform(10, 190, m), rng.uniform(10, 190, m)
    agg_obs = pd.DataFrame({"pxl_col_in_fullres": px, "pxl_row_in_fullres": py,
                            "score": rng.uniform(0, 5, m), "count": rng.integers(1, 4, m).astype(float),
                            "cls": pd.Categorical(rng.choice(["a", "b", "c"], m))},
                           index=pd.Index([f"c{i}" for i in range(m)], name="Cell_ID"))
    agg_obs["geometry"] = [box(x * 0.5 - 2, y * 0.5 - 2, x * 0.5 + 2, y * 0.5 + 2).wkt for x, y in zip(px, py)]
    viz.add_agg(ad.AnnData(X=sparse.csr_matrix(X.astype(np.float32)), obs=agg_obs,
                           var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])), "SC")
    return viz


def image(ax):
    '''The rendered figure of ax, as an array'''
    fig = ax.get_figure()
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return pixels


def scalebar_color(ax):
    lines = [l for l in ax.lines if l.get_linewidth() == 4 and l.get_zorder() == 1000]
    assert len(lines) == 1
    return to_rgba(lines[0].get_color())


# ---------- pandas 3: plot.hist of string / categorical columns ----------

def test_A_hist_of_string_and_categorical_columns():
    viz = make_viz()
    agg = viz.agg["SC"]
    for obj, what in [(viz, "zone"), (viz, "tissue"), (agg, "cls")]:
        values = obj.get(what) # an object array, as in every pandas version
        new = image(obj.plot.hist(what))
        fig, ax = plt.subplots(figsize=(7, 7)) # as plot.hist makes it
        old = image(orig_plot_histogram(pd.Series(values, dtype=object), bins=20, title=what, ylab=None, ax=ax)) # pandas 2: object dtype
        assert np.array_equal(new, old), what
    if PANDAS_3:
        try:
            orig_plot_histogram(pd.Series(viz.get("zone"))) # str dtype
            raised = False
        except TypeError:
            raised = True
        plt.close("all")
        assert raised


def test_B_hist_same_as_orig_for_other_dtypes():
    rng = np.random.default_rng(1)
    for values in [pd.Series(rng.uniform(0, 5, 200)), pd.Series(rng.integers(0, 6, 200)),
                   pd.Series(rng.uniform(0, 5, 200) > 2), pd.Series(rng.choice(["x", "y"], 200).astype(object), dtype=object)]:
        for cmap in [None, "viridis"]:
            old = image(orig_plot_histogram(values, cmap=cmap))
            new = image(HiVis_plot.plot_histogram(values, cmap=cmap))
            assert np.array_equal(new, old), (values.dtype, cmap)
    # a categorical Series (1.1.0 raised TypeError in every pandas version) is drawn as its values
    values = pd.Series(rng.choice(["x", "y"], 200))
    assert np.array_equal(image(HiVis_plot.plot_histogram(values.astype("category"))),
                          image(orig_plot_histogram(values.astype(object))))
    # nullable numeric dtypes (numpy can't interpret them) are numeric
    values = pd.Series([1, 2, 2, 3, None], dtype="Int64")
    assert HiVis_plot._is_numeric(values.dtype) and not HiVis_plot._is_numeric(pd.Series([True, False]).dtype)


# ---------- O37: a Colormap as cmap ----------

def test_C_colormap_same_as_its_name():
    viz = make_viz()
    agg = viz.agg["SC"]
    plots = [lambda cmap: viz.plot.spatial("value", cmap=cmap),
             lambda cmap: viz.plot.spatial("value", cmap=cmap, xlim=[0, 40], ylim=[0, 40]), # exact squares
             lambda cmap: viz.plot.spatial("zone", cmap=cmap),
             lambda cmap: viz.plot.spatial("zone", cmap=cmap, xlim=[0, 40], ylim=[0, 40]),
             lambda cmap: agg.plot.spatial("score", cmap=cmap),
             lambda cmap: agg.plot.cells("score", cmap=cmap),
             lambda cmap: agg.plot.cells("cls", cmap=cmap)]
    for k, plot in enumerate(plots):
        for name in ["viridis", "tab10"]:
            assert np.array_equal(image(plot(colormaps[name])), image(plot(name))), (k, name)


def test_D_listed_colormap_and_bad_cmap():
    viz = make_viz()
    agg = viz.agg["SC"]
    zones = ListedColormap(["red", "green", "blue"])
    ax = agg.plot.cells("count", cmap=zones, line_color=None) # 1, 2, 3
    colors = {tuple(np.round(c, 3)) for coll in ax.collections for c in coll.get_facecolor()}
    plt.close("all")
    assert colors == {(1.0, 0.0, 0.0, 0.7), (0.0, 0.502, 0.0, 0.7), (0.0, 0.0, 1.0, 0.7)}, colors
    for bad in [None, {"a": "red"}, 3]: # UnboundLocalError in 1.1.0
        for plot in [lambda: agg.plot.cells("score", cmap=bad), lambda: viz.plot.spatial("value", cmap=bad)]:
            try:
                plot()
                error = None
            except Exception as e:
                error = e
            plt.close("all")
            assert isinstance(error, ValueError), repr(error)


# ---------- O31: scalebar ----------

def test_E_scalebar_docs_and_length_microns():
    for func in [HiVis_plot.PlotVisium.spatial, HiVis_plot.PlotAgg.spatial, HiVis_plot.PlotAgg.cells]:
        assert "length_microns" not in func.__doc__ and "{length=None, text=True" in func.__doc__
    params = inspect.signature(HiVis_plot.add_scalebar).parameters
    assert list(params)[:8] == ["ax", "microns_per_pixel", "length", "text", "bar_offset", "line_width", "color", "text_offset"]
    viz = make_viz()
    expected = image(viz.plot.spatial(scalebar=20))
    assert np.array_equal(image(viz.plot.spatial(scalebar={"length": 20})), expected)
    assert np.array_equal(image(viz.plot.spatial(scalebar={"length_microns": 20})), expected) # TypeError in 1.1.0
    assert not np.array_equal(image(viz.plot.spatial(scalebar=50)), expected)


def test_F_scalebar_dict_not_modified():
    white, black = to_rgba("white"), to_rgba("black")
    viz = make_viz(fluorescence=True)
    # the default colors, as before
    assert scalebar_color(viz.plot.spatial()) == white
    assert scalebar_color(viz.plot.spatial(image=False)) == black
    assert scalebar_color(make_viz().plot.spatial()) == black
    plt.close("all")
    scalebar = {"length": 20}
    assert scalebar_color(viz.plot.spatial(scalebar=scalebar)) == white
    assert scalebar == {"length": 20} # 1.1.0 added "color": "white"
    assert scalebar_color(viz.plot.spatial(image=False, scalebar=scalebar)) == black # white in 1.1.0
    assert scalebar_color(viz.agg["SC"].plot.cells(image=False, scalebar=scalebar)) == black
    assert scalebar == {"length": 20}
    assert scalebar_color(viz.plot.spatial(image=False, scalebar={"color": "red"})) == to_rgba("red")
    plt.close("all")


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
