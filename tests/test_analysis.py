# -*- coding: utf-8 -*-
"""
Synthetic tests for the 1.1.0 analysis fixes. Compares the current implementation with the one from HiVis 1.0.6:
    * dge sliced the normalized matrix per gene even for fisher_exact, which doesn't use it (slow on CSR)
    * score_genes has a layer argument. Without it, it scores X as before
    * docstrings with "\\_" and "\\**" no longer raise SyntaxWarning, and their text is unchanged
    * matnorm(DataFrame, axis="row") failed under pandas 3 (read-only array)
    * dge with wilcox / t_test slices the genes from CSC copies instead of CSR (same p-values, faster)

Run with pytest, or directly: python tests/test_analysis.py
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import glob
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.stats import mannwhitneyu, ttest_ind, fisher_exact
from tqdm import tqdm

from HiVis import HiVis, HiVis_utils, HiVis_plot, HiVis_analysis, other_utils
from HiVis.HiVis_utils import matnorm

TMP_DIRS = []
PACKAGE_DIR = os.path.dirname(HiVis_utils.__file__)


# ---------- HiVis 1.0.6 implementation, kept verbatim for comparison ----------

def orig_dge(adata, column, group1, group2=None, umi_thresh=0, layer=None,
        method="fisher_exact", alternative="two-sided", inplace=False):
    '''
    HiVis_utils.dge from HiVis 1.0.6. Runs differential gene expression analysis between two groups.
    Values will be saved in self.var: expression_mean, log2fc, pval
    parameters:
        * column - which column in obs has the groups classification
        * group1 - specific value in the "column"
        * group2 - specific value in the "column". 
                   if None, will run against all other values, and will be called "rest"
        * layer - which layer to get the data from (if None will get from adata.X)
        * method - one of ["fisher_exact", "wilcox", "t_test"]
        * alternative - {"two-sided", "less", "greater"}
        * umi_thresh - use only cells with more UMIs than this number
        * inplace - modify the adata.var with log2fc, pval and expression columns?
    '''

    def get_data(ann, lyr):
        return ann.X if lyr is None else ann.layers[lyr]
    df = adata.var.copy()

    # Group1 prep
    group1_adata = adata[adata.obs[column] == group1].copy()
    group1_data = get_data(group1_adata, layer)
    if umi_thresh:
        mask1 = group1_data.sum(axis=1) > umi_thresh
        if sum(mask1) == 0:
            raise ValueError(f"No cells in group '{group1}' pass umi_thresh={umi_thresh}")
        group1_adata = group1_adata[mask1].copy()
        group1_data = get_data(group1_adata, layer)
    group1_adata_raw = group1_data.copy()
    total1 = group1_adata_raw.sum()

    print(f'Normalizing "{group1}" spots')
    if layer is None:
        group1_adata.X = matnorm(group1_data, axis="row")


    # Group2 prep 
    if group2 is None:
        group2_adata = adata[(adata.obs[column] != group1) & ~adata.obs[column].isna()].copy()
        group2 = "rest"
    else:
        group2_adata = adata[adata.obs[column] == group2].copy()
    group2_data = get_data(group2_adata, layer)
    if umi_thresh:
        mask2 = group2_data.sum(axis=1) > umi_thresh
        if sum(mask2) == 0:
            raise ValueError(f"No cells in group '{group2}' pass umi_thresh={umi_thresh}")
        group2_adata = group2_adata[mask2].copy()
        group2_data = get_data(group2_adata, layer)
    
    group2_adata_raw = group2_data.copy()
    total2 = group2_adata_raw.sum()
    print(f'Normalizing "{group2}" spots')
    if layer is None:
        group2_adata.X = matnorm(group2_data, axis="row")

    if layer is None:
        group1_norm = group1_adata.X
        group2_norm = group2_adata.X
    else:
        group1_norm = group1_adata.layers[layer]
        group2_norm = group2_adata.layers[layer]

    # pre-alloc result columns
    df[group2], df[group1] = np.nan, np.nan
    df[f"expression_mean_{column}"] = np.nan
    df[f"log2fc_{column}"] = np.nan
    df[f"pval_{column}"] = np.nan

    sum1 = np.asarray(group1_data.sum(axis=0)).ravel()
    sum2 = np.asarray(group2_data.sum(axis=0)).ravel()
    mean1 = sum1 / sum1.sum() if sum1.sum() > 0 else np.zeros_like(sum1, dtype=float)
    mean2 = sum2 / sum2.sum() if sum2.sum() > 0 else np.zeros_like(sum2, dtype=float)
    # mean1 = sum1 / sum1.sum()
    # mean2 = sum2 / sum2.sum()
    
    df[group1] = mean1
    df[group2] = mean2
    df[f"expression_mean_{column}"] = np.nanmean([mean1,mean2],axis=0)
    # df[f"expression_mean_{column}"] = (mean1 + mean2) / 2

    # smallest non-zero mean for pseudocount
    pn = df.loc[df[f"expression_mean_{column}"] > 0, f"expression_mean_{column}"].min()

    for j, gene in enumerate(tqdm(df.index, desc=f"Running {method} on [{column}]")):
        if (mean1[j] == 0) and (mean2[j] == 0):
            df.at[gene, f"log2fc_{column}"] = 0.0
            df.at[gene, f"pval_{column}"] = np.nan
            continue

        cur_norm1 = group1_norm[:, j].toarray().ravel()
        cur_norm2 = group2_norm[:, j].toarray().ravel()

        # Calculate log2 fold-change
        df.at[gene, f"log2fc_{column}"] = np.log2((mean1[j] + pn) / (mean2[j] + pn))

        # Calculate Pval
        if method == "fisher_exact":
            g1_counts = group1_adata_raw[:, j].sum()
            g2_counts = group2_adata_raw[:, j].sum()

            table = [
                [g1_counts,                 g2_counts],
                [total1 - g1_counts, total2 - g2_counts]
            ]
            _, p = fisher_exact(table, alternative=alternative)                

        elif method == "wilcox":
            _, p = mannwhitneyu(cur_norm1, cur_norm2, alternative=alternative)

        elif method == "t_test":
            _, p = ttest_ind(cur_norm1, cur_norm2, alternative=alternative)

        else:
            p = np.nan

        df.at[gene, f"pval_{column}"] = p

    # Add the results to adata.Var
    if inplace:
        columns_to_drop = [col for col in df.columns if col in adata.var.columns]
        adata.var.drop(columns=columns_to_drop, inplace=True)
        adata.var = adata.var.join(df, how="left")

    return df


def orig_dge2(adata, column, group1, group2=None, umi_thresh=0, layer=None,
        method="fisher_exact", alternative="two-sided", inplace=False):
    '''
    HiVis_utils.dge2 from HiVis 1.0.6. Runs differential gene expression analysis between two groups.
    Values will be saved in self.var: expression_mean, log2fc, pval
    parameters:
        * column - which column in obs has the groups classification
        * group1 - specific value in the "column"
        * group2 - specific value in the "column". 
                   if None, will run against all other values, and will be called "rest"
        * layer - which layer to get the data from (if None will get from adata.X)
        * method - one of ["fisher_exact", "wilcox", "t_test"]
        * alternative - {"two-sided", "less", "greater"}
        * umi_thresh - use only cells with more UMIs than this number
        * inplace - modify the adata.var with log2fc, pval and expression columns?
    '''

    def get_data(ann, lyr):
        return ann.X if lyr is None else ann.layers[lyr]
    df = adata.var.copy()

    # Group1 prep
    group1_adata = adata[adata.obs[column] == group1].copy()
    group1_data = get_data(group1_adata, layer)
    if umi_thresh:
        mask1 = group1_data.sum(axis=1) > umi_thresh
        if sum(mask1) == 0:
            raise ValueError(f"No cells in group '{group1}' pass umi_thresh={umi_thresh}")
        group1_adata = group1_adata[mask1].copy()
        group1_data = get_data(group1_adata, layer)
    group1_adata_raw = group1_data.copy()
    total1 = group1_adata_raw.sum()

    print(f'Normalizing "{group1}" spots')
    if layer is None:
        group1_adata.X = matnorm(group1_data, axis="row")


    # Group2 prep 
    if group2 is None:
        group2_adata = adata[(adata.obs[column] != group1) & ~adata.obs[column].isna()].copy()
        group2 = "rest"
    else:
        group2_adata = adata[adata.obs[column] == group2].copy()
    group2_data = get_data(group2_adata, layer)
    if umi_thresh:
        mask2 = group2_data.sum(axis=1) > umi_thresh
        if sum(mask2) == 0:
            raise ValueError(f"No cells in group '{group2}' pass umi_thresh={umi_thresh}")
        group2_adata = group2_adata[mask2].copy()
        group2_data = get_data(group2_adata, layer)
    
    group2_adata_raw = group2_data.copy()
    total2 = group2_adata_raw.sum()
    print(f'Normalizing "{group2}" spots')
    if layer is None:
        group2_adata.X = matnorm(group2_data, axis="row")

    if layer is None:
        group1_norm = group1_adata.X
        group2_norm = group2_adata.X
    else:
        group1_norm = group1_adata.layers[layer]
        group2_norm = group2_adata.layers[layer]

    # pre-alloc result columns
    df[group2], df[group1] = np.nan, np.nan
    df[f"expression_mean_{column}"] = np.nan
    df[f"log2fc_{column}"] = np.nan
    df[f"pval_{column}"] = np.nan

    sum1 = np.asarray(group1_data.sum(axis=0)).ravel()
    sum2 = np.asarray(group2_data.sum(axis=0)).ravel()
    mean1 = sum1 / sum1.sum() if sum1.sum() > 0 else np.zeros_like(sum1, dtype=float)
    mean2 = sum2 / sum2.sum() if sum2.sum() > 0 else np.zeros_like(sum2, dtype=float)
    
    
    df[group1] = mean1
    df[group2] = mean2
    df[f"expression_mean_{column}"] = np.nanmean([mean1,mean2])

    # smallest non-zero mean for pseudocount
    pn = df.loc[df[f"expression_mean_{column}"] > 0, f"expression_mean_{column}"].min()

    for j, gene in enumerate(tqdm(df.index, desc=f"Running {method} on [{column}]")):
        if (mean1[j] == 0) and (mean2[j] == 0):
            df.at[gene, f"log2fc_{column}"] = 0.0
            df.at[gene, f"pval_{column}"] = np.nan
            continue

        cur_norm1 = group1_norm[:, j].toarray().ravel()
        cur_norm2 = group2_norm[:, j].toarray().ravel()

        # Calculate log2 fold-change
        df.at[gene, f"log2fc_{column}"] = np.log2((mean1[j] + pn) / (mean2[j] + pn))

        # Calculate Pval
        if method == "fisher_exact":
            g1_counts = group1_adata_raw[:, j].sum()
            g2_counts = group2_adata_raw[:, j].sum()

            table = [
                [g1_counts,                 g2_counts],
                [total1 - g1_counts, total2 - g2_counts]
            ]
            _, p = fisher_exact(table, alternative=alternative)                

        elif method == "wilcox":
            _, p = mannwhitneyu(cur_norm1, cur_norm2, alternative=alternative)

        elif method == "t_test":
            _, p = ttest_ind(cur_norm1, cur_norm2, alternative=alternative)

        else:
            p = np.nan

        df.at[gene, f"pval_{column}"] = p

    # Add the results to adata.Var
    if inplace:
        columns_to_drop = [col for col in df.columns if col in adata.var.columns]
        adata.var.drop(columns=columns_to_drop, inplace=True)
        adata.var = adata.var.join(df, how="left")

    return df


# ---------- helpers ----------

def make_adata(n=300, n_genes=60, dense=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.uniform(0.05, 3, n_genes), (n, n_genes)).astype(np.float32)
    X[:, :3] = 0 # genes without counts
    X[:40, 5] += 20 # a gene up in group A
    obs = pd.DataFrame({"group": np.repeat(["A", "B", "C"], n // 3)}, index=[f"bin{i}" for i in range(n)])
    obs.loc[obs.index[::25], "group"] = np.nan
    adata = ad.AnnData(X=X if dense else sparse.csr_matrix(X), obs=obs,
                       var=pd.DataFrame({"chrom": "1"}, index=[f"g{i}" for i in range(n_genes)]))
    adata.layers["counts"] = sparse.csr_matrix(X)
    adata.layers["norm"] = sparse.csr_matrix(matnorm(X, axis="row"))
    return adata


def make_viz(n_genes=300):
    path_output = tempfile.mkdtemp(prefix="hivis_test_")
    TMP_DIRS.append(path_output)
    n = 20
    rng = np.random.default_rng(1)
    xs, ys = np.meshgrid(np.arange(n) * 10.0, np.arange(n) * 10.0)
    obs = pd.DataFrame({"pxl_col_in_fullres": xs.ravel(), "pxl_row_in_fullres": ys.ravel()},
                       index=[f"bin{i}" for i in range(n * n)])
    X = rng.poisson(rng.uniform(0.05, 3, n_genes), (n * n, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    adata.layers["norm"] = sparse.csr_matrix(np.log1p(matnorm(X, axis="row") * 1e4))
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    json = {"microns_per_pixel": 0.5, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
            "spot_diameter_fullres": 4, "bin_size_um": 5}
    viz = HiVis.HiVis(adata, img, img[::2, ::2], img[::10, ::10], json, name="viz",
                      path_output=path_output, plot_qc=False)
    m = 120
    agg_X = rng.poisson(rng.uniform(0.5, 10, n_genes), (m, n_genes)).astype(np.float32)
    agg = ad.AnnData(X=sparse.csr_matrix(agg_X),
                     obs=pd.DataFrame({"pxl_col_in_fullres": rng.uniform(0, 190, m), "pxl_row_in_fullres": rng.uniform(0, 190, m)},
                                      index=pd.Index([f"c{i}" for i in range(m)], name="Cell_ID")),
                     var=pd.DataFrame(index=adata.var_names))
    agg.layers["norm"] = sparse.csr_matrix(np.log1p(matnorm(agg_X, axis="row") * 1e4))
    viz.add_agg(agg, "SC")
    return viz


GENES = ["g1", "g5", "g7", "g10", "g20", "g33"]


# ---------- O11: dge ----------

def test_A_dge_same_as_orig():
    '''Same table (exactly) for every method, with and without group2, umi_thresh and layer'''
    for method in ["fisher_exact", "wilcox", "t_test"]:
        for group2 in [None, "B"]:
            for umi_thresh in [0, 40]:
                for layer in [None, "counts", "norm"]:
                    if umi_thresh and layer == "norm": # normalized rows sum to 1
                        continue
                    for alternative in ["two-sided", "greater"]:
                        args = dict(column="group", group1="A", group2=group2, umi_thresh=umi_thresh, layer=layer,
                                    method=method, alternative=alternative)
                        o = orig_dge(make_adata(), **args)
                        n = HiVis_utils.dge(make_adata(), **args)
                        pd.testing.assert_frame_equal(n, o, check_exact=True, obj=str(args))


def test_B_dge_inplace_same_as_orig():
    for method in ["fisher_exact", "wilcox"]:
        a_o, a_n = make_adata(), make_adata()
        orig_dge(a_o, "group", "A", method=method, inplace=True)
        HiVis_utils.dge(a_n, "group", "A", method=method, inplace=True)
        pd.testing.assert_frame_equal(a_n.var, a_o.var, check_exact=True)


def test_C_dge2_same_as_orig():
    '''dge2 (not used by HiVis) had the same per-gene slicing'''
    for method in ["fisher_exact", "wilcox", "t_test"]:
        for layer in [None, "counts"]:
            o = orig_dge2(make_adata(), "group", "A", method=method, layer=layer)
            n = HiVis_utils.dge2(make_adata(), "group", "A", method=method, layer=layer)
            pd.testing.assert_frame_equal(n, o, check_exact=True)


def test_D_dge_fisher_dense():
    '''1.0.6 crashed with dense X (no .toarray()) even for fisher_exact, which never used the slice'''
    try:
        orig_dge(make_adata(dense=True), "group", "A", method="fisher_exact")
        crashed = False
    except AttributeError:
        crashed = True
    assert crashed
    n = HiVis_utils.dge(make_adata(dense=True), "group", "A", method="fisher_exact")
    s = HiVis_utils.dge(make_adata(), "group", "A", method="fisher_exact")
    pd.testing.assert_frame_equal(n, s, check_exact=True)


def test_E_dge_through_analysis_same_as_orig():
    viz = make_viz()
    viz.adata.obs["half"] = np.where(viz.adata.obs["pxl_col_in_fullres"] < 100, "left", "right")
    for method in ["fisher_exact", "wilcox"]:
        o = orig_dge(viz.adata.copy(), "half", "left", method=method, alternative="greater") # two_sided=False
        n = viz.analysis.dge("half", "left", method=method)
        assert np.array_equal(n["pval_left"], o["pval_half"], equal_nan=True)
        assert np.array_equal(n["log2fc"], o["log2fc_half"])
        assert np.array_equal(n[["left", "rest"]], o[["left", "rest"]])


# ---------- O14: score_genes layer ----------

def test_F_score_genes_default_unchanged():
    viz = make_viz()
    for analysis, adata in [(viz.analysis, viz.adata), (viz.agg["SC"].analysis, viz.agg["SC"].adata)]:
        expected = adata.copy()
        sc.tl.score_genes(expected, gene_list=GENES, score_name="s") # 1.0.6 called exactly this
        scores = analysis.score_genes(GENES, "s")
        assert np.array_equal(scores.to_numpy(), expected.obs["s"].to_numpy())
        z = analysis.score_genes(GENES, "z", z_normalize=True)
        assert np.allclose(z.mean(), 0) and not np.array_equal(z.to_numpy(), scores.to_numpy())


def test_G_score_genes_layer():
    viz = make_viz()
    for analysis, adata in [(viz.analysis, viz.adata), (viz.agg["SC"].analysis, viz.agg["SC"].adata)]:
        on_layer = ad.AnnData(X=adata.layers["norm"].copy(), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
        sc.tl.score_genes(on_layer, gene_list=GENES, score_name="s")
        X_before = adata.X.copy()
        scores = analysis.score_genes(GENES, "s_norm", layer="norm")
        assert np.array_equal(scores.to_numpy(), on_layer.obs["s"].to_numpy())
        assert not np.array_equal(scores.to_numpy(), analysis.score_genes(GENES, "s_X").to_numpy())
        assert (adata.X != X_before).nnz == 0


# ---------- O16: docstrings ----------

def test_H_no_syntax_warnings():
    for path in sorted(glob.glob(os.path.join(PACKAGE_DIR, "*.py"))):
        with open(path, encoding="utf8") as f:
            source = f.read()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", SyntaxWarning)
            compile(source, path, "exec")
        assert not [x for x in w if issubclass(x.category, SyntaxWarning)], (path, [str(x.message) for x in w])


def test_I_docstrings_unchanged():
    '''The text is the same as 1.0.6 gave (an invalid escape keeps its backslash)'''
    bs = chr(92)
    for doc in [HiVis_plot.PlotVisium.cor.__doc__, HiVis_plot.PlotAgg.cor.__doc__]:
        assert f"* print{bs}_ (bool) - print most correlated genes" in doc
        assert f"{bs}{bs}" not in doc
    for doc in [HiVis_analysis.AnalysisAgg.smooth.__doc__, HiVis_analysis.AnalysisVisium.smooth.__doc__]:
        assert f"* {bs}**kwargs - Additional Parameters" in doc
        assert f"{bs}{bs}" not in doc


# ---------- O28: matnorm of a DataFrame by rows ----------

def orig_matnorm_rows(df):
    '''The DataFrame/axis="row" branch of matnorm from HiVis 1.0.6'''
    row_sums = df.sum(axis=1).to_numpy().reshape(-1, 1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return df.div(row_sums, axis=0).astype(np.float32)


def test_J_matnorm_dataframe_rows():
    df = pd.DataFrame([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 4.0]], index=["a", "b", "c"], columns=["x", "y", "z"])
    expected = pd.DataFrame([[0.25, 0.75, 0.0], [0.0, 0.0, 0.0], [0.25, 0.25, 0.5]], index=df.index,
                            columns=df.columns, dtype=np.float32)
    if int(pd.__version__.split(".")[0]) >= 3: # documents the bug: read-only array under copy-on-write
        try:
            orig_matnorm_rows(df)
            raised = False
        except ValueError:
            raised = True
        assert raised
    for matnorm_func in [HiVis_utils.matnorm, other_utils.matnorm]:
        before = df.copy()
        result = matnorm_func(df, axis="row")
        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(df, before) # the input is not modified
        # the other branches are unchanged
        pd.testing.assert_frame_equal(matnorm_func(df, axis="col"),
                                      df.div(df.sum(axis=0).replace(0, 1), axis=1).astype(np.float32))
        assert np.allclose(matnorm_func(df.to_numpy(), axis="row"), expected.to_numpy())


# ---------- O29: dge with wilcox / t_test slices CSC columns ----------

def test_K_dge_wilcox_keeps_input_and_matches_orig():
    '''The CSC copies give exactly the 1.0.6 p-values (also covered by test_A), and the input adata is not changed'''
    for method in ["wilcox", "t_test"]:
        for layer in [None, "counts"]:
            adata = make_adata(seed=3)
            X, layers = adata.X.copy(), {k: v.copy() for k, v in adata.layers.items()}
            obs, var = adata.obs.copy(), adata.var.copy()
            n = HiVis_utils.dge(adata, "group", "B", group2="C", method=method, layer=layer)
            o = orig_dge(make_adata(seed=3), "group", "B", group2="C", method=method, layer=layer)
            pd.testing.assert_frame_equal(n, o, check_exact=True)
            assert (adata.X != X).nnz == 0 and adata.X.format == "csr"
            assert all((adata.layers[k] != v).nnz == 0 for k, v in layers.items())
            pd.testing.assert_frame_equal(adata.obs, obs)
            pd.testing.assert_frame_equal(adata.var, var)


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
