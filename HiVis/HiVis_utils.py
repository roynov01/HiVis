# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
import sys
import importlib
import os
from tqdm import tqdm
import math
import warnings
import scanpy as sc
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, ttest_ind, spearmanr, fisher_exact
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import anndata as ad
import json

# import squidpy


MAX_RAM = 16 # maximum GB RAM to use for a variable

def update_instance_methods(instance):
    '''reloads the methods in an instance'''
    DUNDER_METHODS = ["__str__","__getitem__","__len__"]
    module_name = instance.__class__.__module__
    module = sys.modules[module_name]
    module = importlib.reload(module)
    class_name = instance.__class__.__name__
    updated_class = getattr(module, class_name)

    # Update methods of the class in the instance
    for attr_name in dir(updated_class):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue # Skip special attributes like __class__, __init__, etc.
        attr = getattr(updated_class, attr_name)
        if callable(attr):
            setattr(instance, attr_name, attr.__get__(instance, updated_class))
    for dunder_method in DUNDER_METHODS:
        if hasattr(updated_class, dunder_method):
            attr = getattr(updated_class, dunder_method)
            if callable(attr):
                setattr(instance.__class__, dunder_method, attr)


def p_adjust(pvals, method="fdr_bh"):
    '''
    Adjusts Pvalues, return array of q-values.
    pvals - list / array / pd.Series
    method is passed to statsmodels.stats.multitest.multipletests
    '''
    # Track original type to return in same format
    if isinstance(pvals, pd.Series):
        orig = "series"
        ser = pvals.copy()
    elif isinstance(pvals, np.ndarray):
        orig = "ndarray"
        ser = pd.Series(pvals)
    elif isinstance(pvals, list):
        orig = "list"
        ser = pd.Series(pvals)
    else:
        raise TypeError("Input should be a list, numpy array, or pandas Series.")

    # Identify non-NaN positions
    non_nan_mask = ser.notna()
    ntests = int(non_nan_mask.sum())

    # Prepare output series
    qser = pd.Series(np.nan, index=ser.index)

    if ntests == 0:
        # All NaNs (or empty) → return all NaNs, same shape
        pass
    else:
        # Apply correction only to non-NaN values
        _, qvals_corrected, _, _ = multipletests(ser[non_nan_mask].to_numpy(), method=method)
        qser.loc[non_nan_mask] = qvals_corrected

    # Return in original type
    if orig == "series":
        return qser
    elif orig == "ndarray":
        return qser.to_numpy()
    else:  # list
        return qser.tolist()
    

def matnorm(df, axis="col"):
    '''
    Normalizes a dataframe or matrix by the sum of columns or rows.
    
    Parameters:
    - df: np.ndarray, sparse matrix, or pandas DataFrame
    - axis: "col" for column-wise normalization, "row" for row-wise normalization
    
    Returns:
    - Normalized matrix of the same type as input
    '''
    if isinstance(df, pd.Series):
        return df.div(df.sum())

    if isinstance(df, (np.ndarray, np.matrix)):
        df = np.asarray(df)  # Convert matrix to array if needed
        axis_num = 1 if axis == "row" else 0
        sums = df.sum(axis=axis_num, keepdims=True)
        sums[sums == 0] = 1  # Avoid division by zero
        return df / sums

    if isinstance(df, pd.DataFrame):
        if axis == "row":
            row_sums = df.sum(axis=1).to_numpy().reshape(-1, 1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return df.div(row_sums, axis=0).astype(np.float32)
        else:
            col_sums = df.sum(axis=0)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            return df.div(col_sums, axis=1).astype(np.float32)

    if sp.isspmatrix_csr(df):
        if axis == "row":
            row_sums = np.array(df.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            diag_inv = sp.diags(1 / row_sums)
            return diag_inv.dot(df)  # Normalize rows
        else:
            col_sums = np.array(df.sum(axis=0)).ravel()
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            diag_inv = sp.diags(1 / col_sums)
            return df.dot(diag_inv)  # Normalize columns

    raise ValueError("df is not a supported type (list, numpy array, sparse matrix, or dataframe)")
        

def validate_exists(file_path):
    '''Validates if a file exists'''
    if isinstance(file_path, (list, tuple)):
        for path in file_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No such file or directory:\n\t{path}")
    else:
         if not os.path.exists(file_path):
             raise FileNotFoundError(f"No such file or directory:\n\t{file_path}")    


def dge2(adata, column, group1, group2=None, umi_thresh=0, layer=None,
        method="fisher_exact", alternative="two-sided", inplace=False):
    '''
    Runs differential gene expression analysis between two groups.
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

def dge(adata, column, group1, group2=None, umi_thresh=0, layer=None,
        method="fisher_exact", alternative="two-sided", inplace=False):
    '''
    Runs differential gene expression analysis between two groups.
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

def add_spatial_keys(hivis_obj, adata, name):
    """
    Adds spatial keys to the AnnData object to make it Scanpy/Squidpy spatial plot compatible.
    
    Parameters:
        * hivis_obj (HiVis) - that has images and scalefactors json
        * adata (AnnData) - AnnData object to which spatial keys will be added.
        * name (str) - name of adata, will be concatinated to hivis_obj.name
    
    """
    required_cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    if not all(col in adata.obs.columns for col in required_cols):
        raise ValueError("Missing required spatial coordinate columns in adata.obs")
    
    adata.obsm["spatial"] = adata.obs[required_cols].to_numpy()
        
    adata.uns["spatial"] = {
        name: {"images": {"hires": hivis_obj.image_highres,
                          "lowres": hivis_obj.image_lowres},
                           "scalefactors": hivis_obj.json,
                           "metadata": hivis_obj.properties}}
    

def load_images(path_image_fullres, path_image_highres, path_image_lowres):
    '''
    Loads images.
    '''
    print("[Loading images]")
    image_fullres = tifffile.imread(path_image_fullres)
    rgb_dim = image_fullres.shape.index(min(image_fullres.shape)) # Find color channel
    if rgb_dim != 2:  # If the color dimension is not already last
        axes_order = list(range(image_fullres.ndim))  # Default axes order
        axes_order.append(axes_order.pop(rgb_dim))  # Move the RGB dim to the last position
        image_fullres = image_fullres.transpose(axes_order)
    if path_image_highres.endswith(".png"):
        image_highres = plt.imread(path_image_highres)
    else:
        image_highres = tifffile.imread(path_image_highres)
    if path_image_lowres.endswith(".png"):
        image_lowres = plt.imread(path_image_lowres)
    else:
        image_lowres = tifffile.imread(path_image_lowres)
    if len(image_highres.shape) == 2: # convert grayscale to RGB
        image_lowres = _normalize_channel(image_lowres)
        image_lowres = np.stack((image_lowres,)*3,axis=-1)
        image_highres = _normalize_channel(image_highres)
        image_highres = np.stack((image_highres,)*3,axis=-1)
    return image_fullres, image_highres, image_lowres


def _crop_images_permenent(adata, image_fullres, image_highres, image_lowres, scalefactor_json):
    '''
    crops the images, based on the coordinates from the metadata. 
    shifts the metadata to start at x=0, y=0.
    at first run, will save the cropped images.
    '''
    pxl_col_in_fullres = adata.obs["pxl_col_in_fullres"].values
    pxl_row_in_fullres = adata.obs["pxl_row_in_fullres"].values
    
    xlim_pixels_fullres = [math.floor(pxl_col_in_fullres.min()), math.ceil(pxl_col_in_fullres.max())]
    ylim_pixels_fullres = [math.floor(pxl_row_in_fullres.min()), math.ceil(pxl_row_in_fullres.max())]
    # Ensure the limits are within the image boundaries
    xlim_pixels_fullres = [max(0, xlim_pixels_fullres[0]), min(image_fullres.shape[1], xlim_pixels_fullres[1])]
    ylim_pixels_fullres = [max(0, ylim_pixels_fullres[0]), min(image_fullres.shape[0], ylim_pixels_fullres[1])]

    # Crop the full-resolution image
    image_fullres = image_fullres[ylim_pixels_fullres[0]:ylim_pixels_fullres[1],
                                           xlim_pixels_fullres[0]:xlim_pixels_fullres[1],:]
    
    if len(image_highres.shape) == 2:
        image_highres = np.repeat(image_highres[:, :, np.newaxis], 3, axis=2)
    if len(image_lowres.shape) == 2:
        image_lowres = np.repeat(image_lowres[:, :, np.newaxis], 3, axis=2)
    
    # Adjust limits for high-resolution image and crop
    scaling_factor_hires = scalefactor_json["tissue_hires_scalef"]
    xlim_pixels_highres = [x*scaling_factor_hires for x in xlim_pixels_fullres]
    ylim_pixels_highres = [y*scaling_factor_hires for y in ylim_pixels_fullres]
    xlim_pixels_highres[0], xlim_pixels_highres[1] = math.floor(xlim_pixels_highres[0]), math.ceil(xlim_pixels_highres[1])
    ylim_pixels_highres[0], ylim_pixels_highres[1] = math.floor(ylim_pixels_highres[0]), math.ceil(ylim_pixels_highres[1])
    image_highres = image_highres[ylim_pixels_highres[0]:ylim_pixels_highres[1],
                                           xlim_pixels_highres[0]:xlim_pixels_highres[1],:]

    # Adjust limits for low-resolution image and crop
    scaling_factor_lowres = scalefactor_json["tissue_lowres_scalef"]
    xlim_pixels_lowres = [x*scaling_factor_lowres for x in xlim_pixels_fullres]
    ylim_pixels_lowres = [y*scaling_factor_lowres for y in ylim_pixels_fullres]
    xlim_pixels_lowres[0], xlim_pixels_lowres[1] = math.floor(xlim_pixels_lowres[0]), math.ceil(xlim_pixels_lowres[1])
    ylim_pixels_lowres[0], ylim_pixels_lowres[1] = math.floor(ylim_pixels_lowres[0]), math.ceil(ylim_pixels_lowres[1])
    image_lowres = image_lowres[ylim_pixels_lowres[0]:ylim_pixels_lowres[1],
                                         xlim_pixels_lowres[0]:xlim_pixels_lowres[1],:]
    
    # Shift the metadata to the new poisition
    adata.obs["pxl_col_in_fullres"] = adata.obs["pxl_col_in_fullres"] - xlim_pixels_fullres[0]
    adata.obs["pxl_row_in_fullres"] = adata.obs["pxl_row_in_fullres"] - ylim_pixels_fullres[0]
    
    return adata, image_fullres, image_highres, image_lowres
    

def _export_images(path_image_fullres, path_image_highres, path_image_lowres,
                    image_fullres, image_highres, image_lowres, force=False,um_per_pxl=None):
    '''Saves cropped images. force - overrite existing files?'''
    def _export_image(img, path):
        nonlocal printed_message
        if not os.path.exists(path) or force:
            if not printed_message:
                print(f"[Saving cropped images] {path_image_fullres}")
                printed_message = True
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            if um_per_pxl:
                pixels_per_cm = 1 / (um_per_pxl * 1e-4) 
                tifffile.imwrite(path, img, resolution=(pixels_per_cm, pixels_per_cm), resolutionunit='CENTIMETER')
            else:
                tifffile.imwrite(path, img)
            # image = Image.fromarray(img)
            # image.save(save_path, format='TIFF')
            
    printed_message = False
    images = [image_fullres, image_highres, image_lowres]
    paths = [path_image_fullres, path_image_highres, path_image_lowres]
    for img, path in zip(images, paths):
        _export_image(img, path)
        um_per_pxl = None # We want this value only for the fullres image
    return images
              

def _edit_adata(adata, scalefactor_json, mito_name_prefix):
    '''
    Adds QC (nUMI, mito %) and unit transformation to anndata.
    '''
    adata.obs["pxl_col_in_lowres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
    adata.obs["pxl_row_in_lowres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
    adata.obs["pxl_col_in_highres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
    adata.obs["pxl_row_in_highres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
    adata.obs["um_x"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["microns_per_pixel"]
    adata.obs["um_y"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["microns_per_pixel"]

    # Quality control - number of UMIs and mitochondrial %
    adata.obs["nUMI"] = np.array(adata.X.sum(axis=1).flatten())[0]
    adata.var["nUMI_gene"] = np.array(adata.X.sum(axis=0).flatten())[0]
    mito_genes = adata.var_names[adata.var_names.str.startswith(mito_name_prefix)].values
    adata.obs["mito_sum"] = adata[:,adata.var.index.isin(mito_genes)].X.sum(axis=1).A1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        adata.obs["nUMI_log10"] = np.log10(adata.obs["nUMI"])
        adata.var["nUMI_gene_log10"] = np.log10(adata.var["nUMI_gene"])
        adata.obs["mito_percent_log10"] = np.log10((adata.obs["mito_sum"] / adata.obs["nUMI"]) * 100)
    for col in ['in_tissue', 'array_row', 'array_col']:
        if col not in adata.obs.columns:
            adata.obs[col] = 1
    return adata


def _measure_fluorescence(adata, image_fullres, fluorescence, spot_diameter_fullres):
    '''
    Adds measurements of each fluorescence channel into the adata.
    '''
    num_channels = image_fullres.shape[2]
    if len(fluorescence) != num_channels:
        raise ValueError(f"Length of 'fluorescence' should be number of channels in image ({num_channels})")
    
    half_size = int(spot_diameter_fullres / 2)
    
    # Extract the coordinates of the spot centers
    centers_x = adata.obs['pxl_col_in_fullres'].values.astype(int)
    centers_y = adata.obs['pxl_row_in_fullres'].values.astype(int)
    
    # Loop over each channel
    for idx, channel in enumerate(fluorescence):
        # Initialize an array to hold the fluorescence sums for this channel
        if channel in adata.obs.columns:
            continue
        fluorescence_sums = np.zeros(len(centers_x))
        
        # Calculate fluorescence sums per spot for this channel
        for j, (cx, cy) in enumerate(tqdm(zip(centers_x, centers_y), total=len(centers_x),
                                           desc=f"Calculating intensity per spot: {channel}")):
            # Define the square bounding box
            x_min, x_max = max(cx - half_size, 0), min(cx + half_size + 1, image_fullres.shape[1])
            y_min, y_max = max(cy - half_size, 0), min(cy + half_size + 1, image_fullres.shape[0])
    
            # Sum the pixels in this region for the current channel
            fluorescence_sums[j] = image_fullres[y_min:y_max, x_min:x_max, idx].sum()
    
        # Assign the sums to adata.obs for this channel
        adata.obs[channel] = fluorescence_sums
    
def fluorescence_to_RGB(image, colors:list, normalization_method=None):
    '''
    Creates RGB image from a multichannel.
    parameters:
        * image - np.array of shape (y,x,c)
        * colors - list of colors, some can be None
        * normalization_method - {"percentile", "histogram","clahe","sqrt" or None for minmax}
    '''
    # Initialize an empty RGB image with the same spatial dimensions
    image_shape = image.shape[:2]
    image_rgb = np.zeros((*image_shape, 3))
    
    # Loop over the channels and apply the specified colors
    for idx, color in tqdm(enumerate(colors),total=len(colors),desc="Normilizing channels"):
        if color is None:
            continue  # Ignore this channel
        if idx >= image.shape[-1]:
            break  # Prevent index errors if there are fewer channels than expected

        # Get the fluorescence channel data
        channel_data = image[..., idx]

        # Normalize the channel data for visualization
        normalized_channel = _normalize_channel(channel_data, normalization_method)
       
        # Convert color name or hex to RGB values
        color_rgb = np.array(to_rgba(color)[:3])  # Extract RGB components

        # Add the weighted channel to the RGB image
        for i in range(3):  # For each RGB component
            image_rgb[..., i] += normalized_channel * color_rgb[i]

    # Clip the RGB values to be between 0 and 1
    image_rgb = np.clip(image_rgb, 0, 1)

    return image_rgb
 

def _normalize_channel(channel_data, method="percentile"):
    '''Normilizes one image channel based on the given method'''
    if method == "percentile":
        p_min, p_max = np.percentile(channel_data, (1, 99))
        if p_max > p_min:
            normalized = (channel_data - p_min) / (p_max - p_min)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = channel_data.copy()
    elif method == "histogram":
        from skimage import exposure
        normalized = exposure.equalize_hist(channel_data)
    elif method == "clahe":
        from skimage import exposure
        normalized = exposure.equalize_adapthist(channel_data, clip_limit=0.03)
    elif method == "sqrt":
        ch_min = channel_data.min()
        shifted = channel_data - ch_min
        max_val = shifted.max()
        if max_val > 0:
            normalized = np.sqrt(shifted) / np.sqrt(max_val)
        else:
            normalized = channel_data.copy()
    else: # Min-max scaling
        ch_min = channel_data.min()
        ch_max = channel_data.max()
        diff = ch_max - ch_min
        if diff > 0:
            normalized = (channel_data - ch_min) / diff
        else:
            normalized = channel_data.copy()
    return normalized
   

def _import_data(metadata_path, path_input_data, path_image_fullres, on_tissue_only):
    '''Imports data, metadata and image
        parameters:
            * paths - metadata_path is parquet file, path_input_data is folder, 
                      such as square_002um. path_image_fullres is tif file.
            * on_tissue_only - filter spots that are classified to be under tissue?
    '''
    # load metadata (and save as CSV)
    print("[Loading metadata]")        
    metadata = pd.read_parquet(metadata_path)
    if not os.path.isfile(metadata_path.replace(".parquet",".csv")):
        print("[Writing metadata to CSV]")  
        metadata.to_csv(metadata_path.replace(".parquet",".csv"),index=False)
    del metadata["array_row"]
    del metadata["array_col"]
    
    # load data
    print("[Loading data]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique")
        adata = sc.read_visium(path_input_data, source_image_path=path_image_fullres)
        # adata = squidpy.read.visium(path_input_data, source_image_path=path_image_fullres)
    adata.var_names_make_unique()
    # del adata.uns["spatial"]
    
    # filter spots that are classified to be under tissue
    if ('in_tissue' in adata.obs.columns) and on_tissue_only: 
        metadata = metadata.loc[metadata['in_tissue'] == 1,]
        adata = adata[adata.obs['in_tissue'] == 1]
        del metadata["in_tissue"] 
    
    # merge data and metadata
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    metadata.set_index('barcode', inplace=True)
    adata.obs = adata.obs.join(metadata, how='left')
    return adata


def noise_mean_curve(adata,layer=None,inplace=False,poly_deg=4):
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers.keys():
            raise KeyError(f"Layer {layer} doesn't exist in the adata")
        X = adata.layers[layer]
    
    if sp.issparse(X):
        # Compute mean per gene over cells
        mean_expression = np.array(X.mean(axis=0)).ravel()
        # Compute mean of squares for each gene using the .power(2) method
        mean_square = np.array(X.power(2).mean(axis=0)).ravel()
        # Standard deviation computed using the formula: sqrt(E[x^2] - (E[x])^2)
        sd_expression = np.sqrt(np.maximum(mean_square - mean_expression**2, 0))
    else:
        # For dense matrices, standard operations work
        mean_expression = np.mean(X, axis=0)
        sd_expression = np.std(X, axis=0)
    pn = mean_expression[mean_expression > 0].min()
    cv = sd_expression / (mean_expression + pn)
    
    valid_genes = mean_expression > 0
    cv_pn = cv[cv > 0].min()
    cv_log = np.log10(cv[valid_genes] + cv_pn)
    exp_log = np.log10(mean_expression[valid_genes])

    # Fit an Ordinary Least Squares regression model
    poly_feat = PolynomialFeatures(poly_deg, include_bias=False)
    X_poly = poly_feat.fit_transform(exp_log.reshape(-1, 1))
    
    poly_model = LinearRegression()
    poly_model.fit(X_poly, cv_log)
    cv_log_pred = poly_model.predict(X_poly)
    residuals = cv_log - cv_log_pred
    
    df = pd.DataFrame({
        "gene": np.array(adata.var_names)[valid_genes],
        "expression_mean": mean_expression[valid_genes],
        "mean_log": exp_log,
        "cv": cv[valid_genes],
        "cv_log10": cv_log,
        "residual": residuals
    })
    
    adata.uns["noise_mean_curve"] = { # save the model
        "poly_deg"   : poly_deg,
        "coef"       : poly_model.coef_.tolist(), 
        "intercept"  : float(poly_model.intercept_)
    }
    
    
    if inplace:
        cols = ["cv", "expression_mean", "residual", "cv_log10", "mean_log"]
        adata.var.loc[df["gene"], cols] = df[cols].values
    
    return df

def cor_gene(adata, vec, gene_name, self_corr_value=None, normalize=True,  layer: str = None, inplace=False):
    '''
    Computes Spearman correlation of a given gene (represented by vec) with all genes.
    Parameters:
        * adata - AnnData object containing the data.
        * vec - Expression vector for the gene of interest.
        * gene_name - Identifier (name) of the gene.
        * normalize - normilize data and vector (matnorm)?
        * self_corr_value - Replace self-correlation with this value if provided.
                         If False, no replacement is done.
        * layer - Layer in adata to compute the correlation from (default uses adata.X).
        * inplace - If True, add the computed values to adata.var; otherwise return the DataFrame.
    '''

    # Check if the gene is expressed
    if vec.sum() == 0:
        print("Gene is not expressed!")
        return None

    # Check if the vector length matches the number of observations
    if len(vec) != adata.shape[0]:
        raise ValueError(f"{gene_name} isn't a valid gene or obs")

    if layer is not None:
        matrix = adata.layers[layer]
    else:
        matrix = adata.X

    # Normalize
    matrix = matnorm(matrix, "row")
    if normalize:
        vec = vec / vec.sum()

    # Calculate mean expression of each gene 
    gene_means = np.asarray(matrix.mean(axis=0)).ravel()

    corrs = np.zeros(adata.n_vars, dtype=np.float64)
    pvals = np.zeros(adata.n_vars, dtype=np.float64)
    if hasattr(matrix, "toarray"):
        estimated_memory = estimate_dense_memory(matrix)
        if estimated_memory < MAX_RAM:
            matrix = matrix.toarray()

    # Compute Spearman correlation for each gene
    for i in tqdm(range(adata.n_vars), desc=f"Computing correlation with {gene_name}"):
        y = matrix[:, i]
        if hasattr(y, "toarray"):
            y = y.toarray().ravel() 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            r, p = spearmanr(vec, y)
        corrs[i] = r
        pvals[i] = p
    qvals = p_adjust(pvals)
    
    df = pd.DataFrame({"r": corrs,"expression_mean": gene_means,
        "gene": adata.var_names,"pval": pvals,"qval": qvals})

    # Replace the self-correlation value if specified
    if self_corr_value is not None:
        df.loc[df["gene"] == gene_name, "r"] = self_corr_value

    # If inplace, add the results to adata.var
    if inplace:
        adata.var[f"cor_{gene_name}"] = df["r"].values
        adata.var[f"exp_{gene_name}"] = df["expression_mean"].values
        adata.var[f"cor_qval_{gene_name}"] = df["qval"].values

    return df


def cor_genes(adata,gene_list,self_corr_value=None, normalize=True, layer=None):
    """
    Compute a pairwise correlation matrix among all genes in gene_list.
    Returns a DataFrame of correlation, and q-value matrices.
    """

    for g in gene_list:
        if g not in adata.var_names:
            raise ValueError(f"Gene {g} not found in adata.var_names.")
    
    gene_indices = [adata.var_names.get_loc(g) for g in gene_list]
    if layer is not None:
        matrix = adata.layers[layer]
    else:
        matrix = adata.X
    sub_matrix = matrix[:, gene_indices] 
    if sp.issparse(sub_matrix):
        sub_matrix = sub_matrix.toarray() 
    
    if normalize:
        sub_matrix = matnorm(sub_matrix, "row")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat, pval_mat = spearmanr(sub_matrix, axis=0)
    
    # Adjust p-values
    qvals_flat = p_adjust(pval_mat.flatten())
    qval_mat = np.array(qvals_flat).reshape(pval_mat.shape)

    if self_corr_value is not None:
        np.fill_diagonal(corr_mat, self_corr_value)
        np.fill_diagonal(qval_mat, np.nan)
    
    corr_df = pd.DataFrame(corr_mat, index=gene_list, columns=gene_list)
    #qval_df = pd.DataFrame(qval_mat, index=gene_list, columns=gene_list)
    
    return corr_df


def cluster_df(df, correlation=False, cluster_rows=True,
               cluster_cols=True, method="average", metric="euclidean"):
    '''
    Clusters a DataFrame by rows and/or columns using hierarchical clustering.
    
    Parameters:
        * df - If correlation=True, df must be a square, symmetric correlation matrix.
        * correlation - If True, interpret df as a correlation matrix and transform it via 
            distance = (1 - correlation) before clustering.
        * cluster_rows - Whether to cluster (reorder) the rows.
        * cluster_cols - Whether to cluster (reorder) the columns.
        * method - Linkage method for hierarchical clustering(e.g. "single", "complete", "average", "ward", ...).
        * metric - Distance metric for `pdist` or `linkage`. Ignored if `correlation=True`, 
            because we simply do `distance = 1 - df` and feed it to `linkage(...)` via `squareform()`.
    
    Returns a new DataFrame, reordered according to the clustering of rows and/or columns.
    '''
    def _get_dendrogram_order(data, method="average", metric="euclidean", correlation=False):
        if correlation:
            dist_mat = 1 - data
            dist_condensed = squareform(dist_mat.to_numpy(), checks=False)
        else:
            dist_condensed = pdist(data, metric=metric)
        Z = linkage(dist_condensed, method=method)
        dend = dendrogram(Z, no_plot=True)
        return dend["leaves"]

    df_out = df.copy()

    if correlation:
        if cluster_rows or cluster_cols:
            order = _get_dendrogram_order(df_out, method=method, metric=metric, correlation=True)
            df_out = df_out.iloc[order, order]
    else:
        if cluster_rows:
            row_order = _get_dendrogram_order(df_out, method=method, metric=metric, correlation=False)
            df_out = df_out.iloc[row_order, :]
        if cluster_cols:
            col_order = _get_dendrogram_order(df_out.T, method=method, metric=metric, correlation=False)
            df_out = df_out.iloc[:, col_order]

    return df_out

def estimate_dense_memory(matrix):
    '''return size (in GB) of a sparse matrix upon convertion'''
    n_rows, n_cols = matrix.shape
    element_size = matrix.dtype.itemsize  # e.g., 8 for float64
    total_bytes = n_rows * n_cols * element_size
    total_gb = total_bytes / (1024 ** 3)
    return total_gb


def _convert_bool_columns_to_float(df):
    """
    For columns with bool dtype or object columns that only contain
    True/False/NaN, convert to float. (True=1.0, False=0.0, NaN stays NaN)
    """
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            # Pure bool column -> convert directly
            df[col] = df[col].astype(float)
        elif pd.api.types.is_object_dtype(df[col]):
            # a mix of True/False/NaN
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({True, False}):
                df[col] = df[col].astype(float)


def combine_dges(dges_list, group_names, pval_reducer, log2fc_reducer=np.nanmedian, expression_reducer=np.nanmean, exp_thresh=0):
    reducers = {
        "log2fc": log2fc_reducer,
        "expression_min": expression_reducer,
        "expression_max": expression_reducer,
        "expression_mean": expression_reducer,
    }
    for g in group_names:
        reducers[f"pval_{g}"] = pval_reducer
        reducers[g] = expression_reducer

    # merge all dfs on "gene"
    needed_cols = ["gene"] + list(reducers.keys())
    per = []
    for i, df in enumerate(dges_list, start=1):
        cols = [c for c in needed_cols if c in df.columns]
        sub = df[cols].copy()
        sub = sub.rename(columns={c: f"{c}_{i}" for c in cols if c != "gene"})
        per.append(sub.set_index("gene"))

    merged = pd.concat(per, axis=1, join="inner").reset_index() 

    final_df = merged[["gene"]].copy()

    def _apply_reducer(matrix2d, func):
        """Apply reducer, row-wise if axis not supported. Handle tuple return (combine_pvalues)."""
        try:
            return np.asarray(func(matrix2d, axis=1))
        except TypeError:
            pass

        out = []
        for row in matrix2d:
            val = func(row)
            if isinstance(val, (tuple, list)) and len(val) > 1 and np.isscalar(val[1]):
                val = val[1]
            out.append(val)
        return np.asarray(out)

    # reduce each family of columns
    for base_col, func in reducers.items():
        colnames = [c for c in merged.columns if c.startswith(base_col)]
        if not colnames:
            continue
        values = merged[colnames].to_numpy()
        if base_col == "log2fc":
            final_df[f"count_{group_names[0]}"] = np.nansum(values > 0, axis=1).astype(int)
            final_df[f"count_{group_names[1]}"] = np.nansum(values < 0, axis=1).astype(int)
        final_df[base_col] = _apply_reducer(values, func)

    # Filter low expressed genes        
    final_df = final_df[final_df["expression_max"] >= exp_thresh].reset_index(drop=True)
    
    # FDR correction
    pval_cols = [c for c in final_df.columns if c.startswith("pval_")]
    pvals = final_df[pval_cols].to_numpy().ravel()
    qvals = np.array(p_adjust(pvals))
    qvals_matrix = qvals.reshape(final_df[pval_cols].shape)
    qval_cols = [c.replace("pval_", "qval_") for c in pval_cols]
    final_df[qval_cols] = qvals_matrix
    final_df["qval"] = final_df[qval_cols].min(axis=1)
    final_df.index = final_df["gene"]
    final_df.index.name = None

    return final_df


def create_rescaled_images(full_res_image, high_res_scale=0.5, low_res_scale=0.1):
    from skimage.measure import block_reduce
    down_factor_highres = int(1/high_res_scale)
    down_factor_lowres = int(1/low_res_scale)
    high_res_image = block_reduce(full_res_image, block_size=(down_factor_highres, down_factor_highres, 1), func=np.mean)
    low_res_image = block_reduce(full_res_image, block_size=(down_factor_lowres, down_factor_lowres, 1), func=np.mean)
    return high_res_image, low_res_image, high_res_scale, low_res_scale


def _load_spatial_data(matrix_file, gene_file, barcode_file, coord_file, image_file, geojson_file):
   # Load expression matrix
   X = sp.load_npz(matrix_file)  # or read from CSV/MTX depending on format

   # Load gene names and barcodes
   genes = pd.read_csv(gene_file, header=None)[0].values
   barcodes = pd.read_csv(barcode_file, header=None)[0].values

   # Load coordinates
   coords = pd.read_csv(coord_file, index_col=0)
   coords.index = barcodes  # align coordinates with barcodes

   # Create AnnData object
   adata = ad.AnnData(X=X)
   adata.var_names = genes
   adata.obs_names = barcodes
   adata.obsm["spatial"] = coords.values
   coords.columns = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
   adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]] = coords.loc[adata.obs_names]

   # Load scalefactors. make sure "bin_size_um" and "microns_per_pixel" are present
   with open(geojson_file) as f:
       geojson = json.load(f)

   return adata, geojson   

def load_and_prepare_data(matrix_file, gene_file, barcode_file, coord_file, image_file, geojson_file):
    """
    Load and prepare spatial transcriptomics data from non-VisiumHD sources.

    Parameters:
    * matrix_file (str) - Path to the expression matrix file (e.g., .npz, .mtx, or .csv).
    * gene_file (str) - Path to the file containing gene names.
    * barcode_file (str) - Path to the file containing barcodes corresponding to each spot or cell.
    * coord_file (str) - Path to the CSV file containing spatial coordinates aligned with barcodes.
    * image_file (str) - Path to the associated tissue image file.
    * geojson_file (str) - Path to the GeoJSON file containing spatial metadata, including "bin_size_um" and "microns_per_pixel".

    Returns:
    * adata - AnnData object containing expression data, spatial coordinates, and metadata.
    * geojson (dict) - Parsed GeoJSON metadata with spatial scaling information.
    """
    
    # Load the AnnData, full-res image, and GeoJSON (just like we did before)
    adata, geojson = _load_spatial_data(matrix_file, gene_file, barcode_file, coord_file, image_file, geojson_file)
    
    full_res_image = tifffile.imread(image_file)

    # Create high-res and low-res images
    high_res_image, low_res_image = create_rescaled_images(full_res_image)

    return adata, full_res_image, high_res_image, low_res_image, geojson

def rescale_img_and_adata(adata,  microns_per_pixel, img, fluorescence, down_factor=1, high_res_scale=0.25, low_res_scale=0.01):
    if down_factor != 1:
        from skimage.measure import block_reduce
        downscaled_img = block_reduce(img, block_size=(down_factor, down_factor, 1), func=np.mean)
        adata.obs["pxl_row_in_fullres"] = adata.obs["pxl_row_in_fullres"] / down_factor
        adata.obs["pxl_col_in_fullres"] = adata.obs["pxl_col_in_fullres"] / down_factor
        microns_per_pixel_down = microns_per_pixel * down_factor
    else:
        microns_per_pixel_down = microns_per_pixel
        downscaled_img = img
    res = create_rescaled_images(downscaled_img, high_res_scale=high_res_scale, low_res_scale=low_res_scale)
    high_res_image, low_res_image, high_res_scale, low_res_scale = res
    
    high_res_image =fluorescence_to_RGB(high_res_image, colors=list(fluorescence.values()))
    low_res_image = fluorescence_to_RGB(low_res_image, colors=list(fluorescence.values()))
    
    return downscaled_img, high_res_image, low_res_image, microns_per_pixel_down

