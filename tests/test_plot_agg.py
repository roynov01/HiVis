# -*- coding: utf-8 -*-
"""
Synthetic tests for the 1.1.0 plotting fixes. Compares the current implementation with the one from HiVis 1.0.6:
    * PlotAgg._crop shifted the objects by the unclamped xlim/ylim, while the image is cropped from pixel 0,
      so objects were misaligned with the image when the limits went past the image edge
    * PlotAgg._crop copied the whole adata (X, layers and the images in uns) on every plot. Now a view
    * the legend title is the raw column name (the capitalization branch was dead code)
    * the colorbar tick code warned "invalid value encountered in cast" when the values had NaN
    * is_categorical_dtype (deprecated in pandas) is no longer used
    * spatial / cells take vmin / vmax (the color range of the objects and the colorbar). Without them, as before
    * plot_MA takes genes (to color and name instead of the significant genes), color and color_genes

Run with pytest, or directly: python tests/test_plot_agg.py
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import shutil
import tempfile
import types
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from shapely import wkt
from shapely.geometry import box

from HiVis import HiVis, HiVis_plot
from HiVis.HiVis_plot import FULLRES_THRESH, HIGHRES_THRESH, plot_scatter_signif

MPP = 0.5 # microns per pixel. The 200x200 image covers 100x100 microns
TMP_DIRS = []
# objects (cells) sit exactly on these bins (fullres pixels), so they can be compared with the bins' positions
CELL_PXL = {"c0": (10, 10), "c1": (50, 20), "c2": (100, 100), "c3": (150, 60), "c4": (190, 190)}


# ---------- HiVis 1.0.6 implementation, kept verbatim for comparison ----------

def orig_crop(self, xlim=None, ylim=None, resolution=None, geometry=False):
    '''PlotAgg._crop from HiVis 1.0.6. self is a PlotAgg'''
    # If xlim or ylim is None, set to the full range of the data
    if xlim is None:
        xlim = self.xlim_max
    if ylim is None:
        ylim = self.ylim_max

    x_mask = (self.main.adata.obs['um_x'] >= xlim[0]) & (self.main.adata.obs['um_x'] <= xlim[1])
    y_mask = (self.main.adata.obs['um_y'] >= ylim[0]) & (self.main.adata.obs['um_y'] <= ylim[1])
    mask = x_mask & y_mask

    # Crop the adata
    self.main.adata_cropped = self.main.adata[mask].copy()

    # Adjust adata coordinates relative to the cropped image
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    lim_size = max(x_range, y_range)

    if resolution == "full":
        pxl_col, pxl_row, scalef = 'pxl_col_in_fullres', 'pxl_row_in_fullres', 1
    elif resolution == "high":
        pxl_col, pxl_row, scalef = 'pxl_col_in_highres', 'pxl_row_in_highres', self.main.viz.json['tissue_hires_scalef']
    elif resolution == "low":
        pxl_col, pxl_row, scalef = 'pxl_col_in_lowres', 'pxl_row_in_lowres', self.main.viz.json['tissue_lowres_scalef']
    elif lim_size <= FULLRES_THRESH:
        pxl_col, pxl_row, scalef = 'pxl_col_in_fullres', 'pxl_row_in_fullres', 1
    elif lim_size <= HIGHRES_THRESH:
        pxl_col, pxl_row, scalef = 'pxl_col_in_highres', 'pxl_row_in_highres', self.main.viz.json['tissue_hires_scalef']
    else:
        pxl_col, pxl_row, scalef = 'pxl_col_in_lowres', 'pxl_row_in_lowres', self.main.viz.json['tissue_lowres_scalef']
    microns_per_pixel = self.main.viz.json['microns_per_pixel']
    adjusted_microns_per_pixel = microns_per_pixel / scalef
    xlim_pxl = [int(lim/ adjusted_microns_per_pixel) for lim in xlim]
    ylim_pxl = [int(lim/ adjusted_microns_per_pixel) for lim in ylim]

    self.pixel_x = self.main.adata_cropped.obs[pxl_col] - xlim_pxl[0]
    self.pixel_y = self.main.adata_cropped.obs[pxl_row] - ylim_pxl[0]

    if geometry:
        self._init_geometry(adjusted_microns_per_pixel, xlim_pxl, ylim_pxl)


class use_orig_crop:
    '''Within the block, PlotAgg._crop of the given aggregation is the 1.0.6 one'''
    def __init__(self, agg):
        self.agg = agg
    def __enter__(self):
        self.agg.plot._crop = types.MethodType(orig_crop, self.agg.plot)
    def __exit__(self, *exc):
        del self.agg.plot._crop


# ---------- helpers ----------

def make_viz():
    '''Small HiVis (20x20 bins on a 200x200 pixel image) with one aggregation of 5 cells that have geometry'''
    path_output = tempfile.mkdtemp(prefix="hivis_test_")
    TMP_DIRS.append(path_output)
    n = 20
    xs, ys = np.meshgrid(np.arange(n) * 10.0, np.arange(n) * 10.0)
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 20, n * n)
    dist[::7] = np.nan
    obs = pd.DataFrame({"pxl_col_in_fullres": xs.ravel(), "pxl_row_in_fullres": ys.ravel(),
                        "dist_bins": dist}, index=[f"bin{i}" for i in range(n * n)])
    adata = ad.AnnData(X=sparse.csr_matrix(rng.poisson(1, (n * n, 3)).astype(np.float32)), obs=obs,
                       var=pd.DataFrame(index=["g0", "g1", "g2"]))
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    json = {"microns_per_pixel": MPP, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
            "spot_diameter_fullres": 4, "bin_size_um": 5}
    viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name="viz",
                      path_output=path_output, plot_qc=False)

    ids = list(CELL_PXL)
    px = np.array([CELL_PXL[c][0] for c in ids], dtype=float)
    py = np.array([CELL_PXL[c][1] for c in ids], dtype=float)
    agg_obs = pd.DataFrame({"pxl_col_in_fullres": px, "pxl_row_in_fullres": py,
                            "score": [0.5, 1.5, 2.5, 3.5, 4.5],
                            "count": [0, 1, 2, 0, 5],
                            "dist": [1.0, np.nan, 3.0, 4.5, np.nan],
                            "cls": pd.Categorical(["a", "b", "a", "c", "b"], categories=["c", "b", "a"], ordered=True)},
                           index=pd.Index(ids, name="Cell_ID"))
    # squares of 4 microns around each cell, in microns, like import_geometry stores them
    agg_obs["geometry"] = [box(x * MPP - 2, y * MPP - 2, x * MPP + 2, y * MPP + 2).wkt for x, y in zip(px, py)]
    X = sparse.csr_matrix(rng.poisson(3, (len(ids), 3)).astype(np.float32))
    viz.add_agg(ad.AnnData(X=X, obs=agg_obs, var=pd.DataFrame(index=["g0", "g1", "g2"])), "SC")
    return viz


def bin_of(cell):
    x, y = CELL_PXL[cell]
    return f"bin{(y // 10) * 20 + x // 10}"


# limits that go past the image edge (negative), for each resolution
OUTSIDE = [([-20, 60], [-30, 60], "full"), ([-20, 60], [10, 60], "high"), ([10, 60], [-25, 60], "low"),
           ([-20, 60], [-30, 60], None)]
# limits inside the image, and the full extent
INSIDE = [(None, None, None), ([10, 60], [5, 60], "full"), ([10, 60], [5, 60], "high"),
          ([10, 99], [5, 99], "low"), ([12.3, 77.7], [3.1, 55.5], None)]


def close():
    plt.close("all")


# ---------- O1: alignment when the limits go past the image edge ----------

def test_A_orig_misaligned_past_edge():
    '''Documents the 1.0.6 bug: objects are shifted against the bins (and the image) when xlim/ylim < 0'''
    viz = make_viz()
    agg = viz.agg["SC"]
    with use_orig_crop(agg):
        agg.plot.spatial(xlim=[-20, 60], ylim=[-30, 60], img_resolution="full")
    close()
    # the bins and the image are cropped from pixel 0, the objects were shifted by 40 and 60 pixels
    assert agg.plot.pixel_x["c0"] - viz.plot.pixel_x[bin_of("c0")] == 40
    assert agg.plot.pixel_y["c0"] - viz.plot.pixel_y[bin_of("c0")] == 60


def test_B_aligned_past_edge():
    viz = make_viz()
    agg = viz.agg["SC"]
    for xlim, ylim, res in OUTSIDE:
        agg.plot.spatial(xlim=xlim, ylim=ylim, img_resolution=res)
        close()
        cells = agg.plot.pixel_x.index
        assert len(cells) > 0
        for c in cells: # the bins are drawn where the image is (PlotVisium._crop), so the objects must be at the same place
            assert agg.plot.pixel_x[c] == viz.plot.pixel_x[bin_of(c)], (xlim, ylim, res, c)
            assert agg.plot.pixel_y[c] == viz.plot.pixel_y[bin_of(c)], (xlim, ylim, res, c)


def test_C_geometry_aligned_past_edge():
    '''plot.cells: the shapes are centered on the object's position in the cropped image'''
    viz = make_viz()
    agg = viz.agg["SC"]
    for xlim, ylim, res in OUTSIDE:
        agg.plot.cells(xlim=xlim, ylim=ylim, img_resolution=res)
        close()
        geometry = agg.plot.geometry
        assert len(geometry) > 0
        for c, geom in geometry.geometry.items():
            assert np.isclose(geom.centroid.x, viz.plot.pixel_x[bin_of(c)]), (xlim, ylim, res, c)
            assert np.isclose(geom.centroid.y, viz.plot.pixel_y[bin_of(c)]), (xlim, ylim, res, c)


def test_D_same_as_orig_inside_image():
    '''Limits inside the image (and no limits): the same positions, data and shapes as 1.0.6'''
    viz = make_viz()
    agg = viz.agg["SC"]
    for xlim, ylim, res in INSIDE:
        with use_orig_crop(agg):
            agg.plot._crop(xlim, ylim, resolution=res, geometry=True)
        o_x, o_y, o_geom = agg.plot.pixel_x.copy(), agg.plot.pixel_y.copy(), agg.plot.geometry.copy()
        o_obs, o_X = agg.adata_cropped.obs.copy(), agg.adata_cropped.X.toarray()
        agg.plot._crop(xlim, ylim, resolution=res, geometry=True)
        pd.testing.assert_series_equal(agg.plot.pixel_x, o_x)
        pd.testing.assert_series_equal(agg.plot.pixel_y, o_y)
        pd.testing.assert_frame_equal(pd.DataFrame(agg.adata_cropped.obs), o_obs)
        assert np.array_equal(agg.adata_cropped.X.toarray(), o_X)
        assert list(agg.plot.geometry.index) == list(o_geom.index)
        assert all(a.equals(b) for a, b in zip(agg.plot.geometry.geometry, o_geom.geometry))
        pd.testing.assert_frame_equal(pd.DataFrame(agg.plot.geometry.drop(columns="geometry")),
                                      pd.DataFrame(o_geom.drop(columns="geometry")))


def test_E_same_figure_as_orig_inside_image():
    '''The plotted offsets of spatial() and cells() are unchanged inside the image'''
    viz = make_viz()
    agg = viz.agg["SC"]
    for xlim, ylim, res in INSIDE:
        for what in ["score", "cls"]:
            with use_orig_crop(agg):
                ax = agg.plot.spatial(what, xlim=xlim, ylim=ylim, img_resolution=res)
            o = [c.get_offsets().copy() for c in ax.collections]
            close()
            ax = agg.plot.spatial(what, xlim=xlim, ylim=ylim, img_resolution=res)
            n = [c.get_offsets().copy() for c in ax.collections]
            close()
            assert len(o) == len(n) and all(np.array_equal(a, b) for a, b in zip(o, n))
            if xlim is not None: # the 4 micron shapes are within the limits
                with use_orig_crop(agg):
                    ax = agg.plot.cells(what, xlim=xlim, ylim=ylim, img_resolution=res)
                o = [p.vertices.copy() for c in ax.collections for p in c.get_paths()]
                close()
                ax = agg.plot.cells(what, xlim=xlim, ylim=ylim, img_resolution=res)
                n = [p.vertices.copy() for c in ax.collections for p in c.get_paths()]
                close()
                assert len(o) == len(n) and all(np.array_equal(a, b) for a, b in zip(o, n))


# ---------- O13: adata_cropped is a view ----------

def test_F_cropped_is_view_with_same_values():
    viz = make_viz()
    agg = viz.agg["SC"]
    for xlim, ylim, res in INSIDE + OUTSIDE:
        expected = {}
        with use_orig_crop(agg):
            agg.plot._crop(xlim, ylim, resolution=res, geometry=True)
            for what in ["score", "cls", "g1", "count"]:
                for geometry in [False, True]:
                    expected[what, geometry] = agg.get(what, cropped=True, geometry=geometry)
        agg.plot._crop(xlim, ylim, resolution=res, geometry=True)
        assert agg.adata_cropped.is_view
        for (what, geometry), values in expected.items():
            assert np.array_equal(agg.get(what, cropped=True, geometry=geometry), values), (what, geometry)
        assert not agg.adata.is_view, "the aggregation's own adata must not change"


def test_G_hist_and_plots_without_anndata_warnings():
    viz = make_viz()
    agg = viz.agg["SC"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agg.plot.hist("score")
        agg.plot.spatial("g1", xlim=[10, 60], ylim=[5, 60])
        agg.plot.cells("score", xlim=[10, 60], ylim=[5, 60])
        agg.plot.cells("cls")
        close()
    bad = [str(x.message) for x in w if "view" in str(x.message).lower() or x.category.__name__ == "ImplicitModificationWarning"]
    assert not bad, bad


def test_H_pickle_round_trip_and_smaller():
    '''After plotting, the pickle no longer stores a second copy of the aggregation (and its images in uns)'''
    viz = make_viz()
    agg = viz.agg["SC"]
    d = tempfile.mkdtemp(prefix="hivis_test_"); TMP_DIRS.append(d)
    with use_orig_crop(agg):
        agg.plot.spatial("score")
    close()
    p_orig = viz.save(os.path.join(d, "orig.pkl"))
    agg.plot.spatial("score")
    close()
    p_new = viz.save(os.path.join(d, "new.pkl"))
    assert os.path.getsize(p_new) < os.path.getsize(p_orig)
    for path in [p_orig, p_new]: # pickles written with the 1.0.6 copy load and plot as well
        loaded = HiVis.load(path)
        la = loaded.agg["SC"]
        assert np.array_equal(la.get("score", cropped=True), agg.get("score"))
        la.plot.cells("score", xlim=[-20, 60], ylim=[-30, 60])
        close()
        assert la.adata_cropped.is_view
        assert la.adata_cropped._adata_ref is la.adata


# ---------- O2: legend title ----------

def test_I_legend_title_is_raw_column_name():
    '''1.0.6 never capitalized the default legend title (what==what.lower compared to the method). Still raw'''
    viz = make_viz()
    agg = viz.agg["SC"]
    for ax in [agg.plot.spatial("score"), agg.plot.cells("score"), viz.plot.spatial("dist_bins")]:
        labels = [a.get_ylabel() for a in ax.get_figure().axes if a is not ax]
        close()
        assert labels == ["score"] or labels == ["dist_bins"], labels
    ax = agg.plot.spatial("score", legend_title="My title")
    labels = [a.get_ylabel() for a in ax.get_figure().axes if a is not ax]
    close()
    assert labels == ["My title"]


# ---------- O5: colorbar ticks with NaN ----------

def cast_warnings(w):
    return [str(x.message) for x in w if "invalid value encountered in cast" in str(x.message)]


def colorbar(ax):
    cbars = [a for a in ax.get_figure().axes if a is not ax]
    assert len(cbars) == 1
    return cbars[0]


def test_J_no_cast_warning_with_nan():
    viz = make_viz()
    agg = viz.agg["SC"]
    plots = [lambda: agg.plot.spatial("dist"),                          # plot_scatter
             lambda: agg.plot.cells("count"),                           # zeros become NaN
             lambda: viz.plot.spatial("dist_bins", exact=True),         # _plot_squares_exact
             lambda: viz.plot.spatial("dist_bins", exact=False)]        # plot_scatter
    for plot in plots:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot()
        cb = colorbar(ax)
        close()
        assert not cast_warnings(w), cast_warnings(w)
        # as in 1.0.6: NaN made is_int False, so the ticks come from MaxNLocator, not one tick per integer
        assert isinstance(cb.yaxis.get_major_locator(), ticker.MaxNLocator)


def test_K_integer_ticks_unchanged():
    '''Finite integer values with a range <= 10 still get one tick per integer'''
    viz = make_viz()
    agg = viz.agg["SC"]
    ax = agg.plot.cells("count", show_zeros=True)
    ticks = list(colorbar(ax).get_yticks())
    close()
    assert ticks == [0, 1, 2, 3, 4, 5]


# ---------- O15: categorical columns ----------

def test_L_categorical_without_deprecation_warning():
    viz = make_viz()
    agg = viz.agg["SC"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ax = agg.plot.cells("cls")
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        close()
    assert not [x for x in w if "is_categorical_dtype" in str(x.message)]
    assert legend == ["c", "b", "a"], legend # the category order is kept


# ---------- O30: color range (vmin / vmax) ----------

def colorbar_clim(ax):
    return colorbar(ax)._colorbar.mappable.get_clim()


def colored_collection(ax):
    collections = [c for c in ax.collections if c.get_array() is not None and len(c.get_array())]
    assert len(collections) == 1
    return collections[0]


def square_colors(ax):
    return np.array([p.get_facecolor() for p in ax.patches])


def plotted_bin_values(viz):
    '''The values viz.plot.spatial drew as squares: non-zero, sorted, NaN skipped'''
    v = viz.get("dist_bins", cropped=True)
    v = np.sort(v[v != 0])
    return v[~np.isnan(v)]


def test_M_color_range_default():
    '''Without vmin/vmax: the range of the plotted values, for the objects and the colorbar (as in 1.0.6)'''
    viz = make_viz()
    agg = viz.agg["SC"]
    ax = agg.plot.spatial("score")
    assert colored_collection(ax).get_clim() == colorbar_clim(ax) == (0.5, 4.5)
    close()
    ax = agg.plot.cells("score")
    assert colored_collection(ax).get_clim() == colorbar_clim(ax) == (0.5, 4.5)
    close()
    ax = viz.plot.spatial("dist_bins", exact=True, xlim=[0, 40], ylim=[0, 40])
    v = plotted_bin_values(viz)
    assert colorbar_clim(ax) == (v.min(), v.max())
    cmap = matplotlib.colormaps.get_cmap("winter")
    assert np.allclose(square_colors(ax), cmap(matplotlib.colors.Normalize(v.min(), v.max())(v)))
    close()


def test_N_color_range_vmin_vmax():
    viz = make_viz()
    agg = viz.agg["SC"]
    ax = agg.plot.spatial("score", vmin=1, vmax=3)
    assert colored_collection(ax).get_clim() == colorbar_clim(ax) == (1, 3)
    close()
    ax = agg.plot.cells("score", vmin=1, vmax=3) # the colorbar is a separate mappable: it must get the range too
    assert colored_collection(ax).get_clim() == colorbar_clim(ax) == (1, 3)
    close()
    ax = viz.plot.spatial("dist_bins", exact=True, xlim=[0, 40], ylim=[0, 40], vmin=5, vmax=10)
    v = plotted_bin_values(viz)
    assert colorbar_clim(ax) == (5, 10)
    cmap = matplotlib.colormaps.get_cmap("winter")
    assert np.allclose(square_colors(ax), cmap(matplotlib.colors.Normalize(5, 10)(v)))
    close()
    ax = viz.plot.spatial("dist_bins", exact=False, vmin=5, vmax=10)
    assert colored_collection(ax).get_clim() == colorbar_clim(ax) == (5, 10)
    close()
    # the integer ticks follow the color range
    ax = agg.plot.cells("count", show_zeros=True, vmin=0, vmax=8)
    assert list(colorbar(ax).get_yticks()) == list(range(9))
    close()
    ax = agg.plot.cells("count", show_zeros=True, vmin=0, vmax=50)
    assert isinstance(colorbar(ax).yaxis.get_major_locator(), ticker.MaxNLocator)
    close()


# ---------- O27: plot_MA genes and colors ----------

def orig_plot_MA(df, qval_thresh=0.25, exp_thresh=0, fc_thresh=0 ,figsize=(7,7), ax=None, title=None,
            size=10, colname_exp="expression_mean",colname_qval="qval",
            colname_fc="log2fc", n_texts=130, ylab="log2(ratio)",repel=False):
    '''HiVis_plot.plot_MA from HiVis 1.0.6'''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    plot = df.loc[df[colname_exp] >= exp_thresh].copy()
    plot["exp"] = np.log10(plot[colname_exp])
    plot["signif"] = (plot[colname_qval] <= qval_thresh) & (abs(plot[colname_fc]) >= fc_thresh)
    if not "gene" in plot.columns:
        plot["gene"] = plot.index
    signif_genes = plot.loc[plot["signif"]==True,"gene"].tolist()
    text = True if len(signif_genes) < n_texts else False
    ax = plot_scatter_signif(plot, "exp", colname_fc, genes=signif_genes,
                             text=text, title=title,ax=ax,size=size,repel=repel,
                             xlab=f"log10({colname_exp.replace('_',' ')})",
                             ylab=ylab,y_line=0,color_genes="red", color="gray")
    return ax


def dge_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"expression_mean": rng.uniform(1e-5, 1e-2, n), "log2fc": rng.normal(0, 1, n),
                         "qval": rng.uniform(0, 1, n)}, index=[f"g{i}" for i in range(n)])


def ma_summary(ax):
    '''Everything plot_MA draws: points (positions, colors) and texts (name, position, color)'''
    points = [(np.round(c.get_offsets().data, 12).tolist(), np.round(c.get_facecolors(), 6).tolist()) for c in ax.collections]
    texts = [(t.get_text(), t.get_position(), t.get_color()) for t in ax.texts]
    return points, texts, ax.get_xlabel(), ax.get_ylabel()


def test_O_plot_MA_default_same_as_orig():
    for kwargs in [{}, {"n_texts": 5}, {"qval_thresh": 0.05, "fc_thresh": 1}, {"exp_thresh": 1e-3, "title": "t"}]:
        o = ma_summary(orig_plot_MA(dge_table(), **kwargs))
        close()
        n = ma_summary(HiVis_plot.plot_MA(dge_table(), **kwargs))
        close()
        assert o == n, kwargs


def test_P_plot_MA_genes_and_colors():
    df = dge_table()
    ax = HiVis_plot.plot_MA(df, genes=["g3", "g10", "g150"], n_texts=0, color="black", color_genes="blue")
    texts = sorted(t.get_text() for t in ax.texts)
    assert texts == ["g10", "g150", "g3"] # named even though n_texts=0
    assert all(t.get_color() == "blue" for t in ax.texts)
    background, highlighted = ax.collections[0], ax.collections[1]
    assert len(highlighted.get_offsets()) == 3 and len(background.get_offsets()) == len(df) - 3
    assert np.allclose(highlighted.get_facecolors()[:, :3], matplotlib.colors.to_rgb("blue"))
    assert np.allclose(background.get_facecolors()[:, :3], matplotlib.colors.to_rgb("black"))
    close()
    ax = HiVis_plot.plot_MA(df, genes=[]) # nothing named or highlighted
    assert not ax.texts and len(ax.collections) == 1
    close()


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
