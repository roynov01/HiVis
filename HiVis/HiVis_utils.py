# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:55:04 2024

@author: royno
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
from scipy.stats import mannwhitneyu, ttest_ind, spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

import HiVis_plot

MAX_RAM = 50 # maximum GB RAM to use for a variable

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
    if isinstance(pvals, (list, np.ndarray)):
        pvals = pd.Series(pvals)
    elif not isinstance(pvals, pd.Series):
        raise TypeError("Input should be a list, numpy array, or pandas Series.")

    # Identify non-NaN indices and values
    non_nan_mask = pvals.notna()
    pvals_non_nan = pvals[non_nan_mask]

    # Apply BH correction on non-NaN p-values
    _, qvals_corrected, _, _ = multipletests(pvals_non_nan, method=method)
    
    # Create a Series with NaNs in original places and corrected values
    qvals = pd.Series(np.nan, index=pvals.index)
    qvals[non_nan_mask] = qvals_corrected

    # Return qvals in the same format as input
    if isinstance(pvals, pd.Series):
        return qvals
    elif isinstance(pvals, np.ndarray):
        return qvals.values
    else:
        return qvals.tolist()
    

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
             
             
             
# def dge(adata, column, group1, group2=None, umi_thresh=0,layer=None,
#                  method="wilcox",alternative="two-sided",inplace=False):
#     '''
#     Runs differential gene expression analysis between two groups.
#     Values will be saved in self.var: expression_mean, log2fc, pval
#     parameters:
#         * column - which column in obs has the groups classification
#         * group1 - specific value in the "column"
#         * group2 - specific value in the "column". 
#                    if None,will run agains all other values, and will be called "rest"
#         * method - either "wilcox" or "t_test"
#         * alternative - {"two-sided", "less", "greater"}
#         * umi_thresh - use only spots with more UMIs than this number
#         * inplace - modify the adata.var with log2fc, pval and expression columns?
#     '''
#     df = adata.var.copy()
        
#     # Get the expression of the two groups
#     group1_exp = adata[adata.obs[column] == group1].copy()
#     group1_exp = group1_exp[group1_exp.X.sum(axis=1) > umi_thresh].copy()  # delete low quality spots
#     print(f'Normilizing "{group1}" spots')
#     group1_exp.X = matnorm(group1_exp.X,axis="row") 

#     # group1_exp.X = group1_exp.X / group1_exp.X.sum(axis=1).A1[:, None]  # matnorm
#     group1_exp = group1_exp.X.todense()
#     df[group1] = group1_exp.mean(axis=0).A1  # save avarage expression of group1 to vars
        
#     if group2 is None:
#         group2_exp = adata[(adata.obs[column] != group1) & ~adata.obs[column].isna()].copy()
#         group2 = "rest"
#     else:
#         group2_exp = adata[adata.obs[column] == group2].copy()
#     group2_exp = group2_exp[group2_exp.X.sum(axis=1) > umi_thresh].copy()  # delete empty spots

#     print(f'Normilizing "{group2}" spots')
#     group2_exp.X = matnorm(group2_exp.X,axis="row") 
#     # group2_exp.X = group2_exp.X / group2_exp.X.sum(axis=1).A1[:, None]  # matnorm
#     group2_exp = group2_exp.X.todense()
#     df[group2] = group2_exp.mean(axis=0).A1  # save avarage expression of group2 to vars
#     # df[group2 + "_med"] = np.median(group2_exp, axis=0).A1
#     # df[group2+"_med"] = group2_exp.median(axis=0).A1  
#     print(f"Number of spots in group1: {group1_exp.shape}, in group2: {group2_exp.shape}")
#     # Calculate mean expression in each group and log2(group1/group2)
#     df[f"expression_mean_{column}"] = df[[group1,group2]].mean(axis=1)
#     pn = df[f"expression_mean_{column}"][df[f"expression_mean_{column}"]>0].min()
#     df[f"log2fc_{column}"] = (df[group1] + pn) / (df[group2] + pn)
#     df[f"log2fc_{column}"] = np.log2(df[f"log2fc_{column}"])  

#     # Wilcoxon rank-sum test
#     df[f"pval_{column}"] = np.nan
#     for j, gene in enumerate(tqdm(df.index, desc=f"Running wilcoxon on [{column}]")):
#         if (df.loc[gene,group1] == 0) and (df.loc[gene,group2] == 0):
#             p_value = np.nan
#         else:
#             cur_gene_group1 = group1_exp[:,j]
#             cur_gene_group2 = group2_exp[:,j]
#             if method == "wilcox":
                
#                 _, p_value = mannwhitneyu(cur_gene_group1, cur_gene_group2, alternative=alternative)
#             elif method == "t_test":
#                 _, p_value = ttest_ind(cur_gene_group1, cur_gene_group2, alternative=alternative)
#         df.loc[gene,f"pval_{column}"] = p_value
#     if inplace:
#         adata.var = adata.var.join(df, how="left")
#     return df

def dge(adata, column, group1, group2=None, umi_thresh=0,layer=None,
                 method="wilcox",alternative="two-sided",inplace=False):
    '''
    Runs differential gene expression analysis between two groups.
    Values will be saved in self.var: expression_mean, log2fc, pval
    parameters:
        * column - which column in obs has the groups classification
        * group1 - specific value in the "column"
        * group2 - specific value in the "column". 
                   if None,will run agains all other values, and will be called "rest"
        * layer - which layer to get the data from (if None will get from adata.X)
        * method - either "wilcox" or "t_test"
        * alternative - {"two-sided", "less", "greater"}
        * umi_thresh - use only cells with more UMIs than this number
        * inplace - modify the adata.var with log2fc, pval and expression columns?
    '''
    def get_data(ann, lyr):
        return ann.X if lyr is None else ann.layers[lyr]

    df = adata.var.copy()

    group1_adata = adata[adata.obs[column] == group1].copy()
    group1_data = get_data(group1_adata, layer)
    if umi_thresh:
        # Filter out low-UMI rows
        mask1 = group1_data.sum(axis=1) > umi_thresh
        group1_adata = group1_adata[mask1].copy()
        group1_data = get_data(group1_adata, layer)

    print(f'Normalizing "{group1}" spots')
    if layer is None:
        group1_adata.X = matnorm(group1_data, axis="row")
        group1_exp = group1_adata.X.todense()
    else:
        group1_adata.layers[layer] = matnorm(group1_data, axis="row")
        group1_exp = group1_adata.layers[layer].todense()
    df[group1] = group1_exp.mean(axis=0).A1

    if group2 is None:
        group2_adata = adata[(adata.obs[column] != group1) & ~adata.obs[column].isna()].copy()
        group2 = "rest"
    else:
        group2_adata = adata[adata.obs[column] == group2].copy()
    group2_data = get_data(group2_adata, layer)
    if umi_thresh:
        mask2 = group2_data.sum(axis=1) > umi_thresh
        group2_adata = group2_adata[mask2].copy()
        group2_data = get_data(group2_adata, layer)
    print(f'Normalizing "{group2}" spots')
    if layer is None:
        group2_adata.X = matnorm(group2_data, axis="row")
        group2_exp = group2_adata.X.todense()
    else:
        group2_adata.layers[layer] = matnorm(group2_data, axis="row")
        group2_exp = group2_adata.layers[layer].todense()

    df[group2] = group2_exp.mean(axis=0).A1

    print(f"Number of entries in group1: {group1_exp.shape}, in group2: {group2_exp.shape}")

    # Calculate mean expression in each group and log2(group1 / group2)
    df[f"expression_mean_{column}"] = df[[group1, group2]].mean(axis=1)
    # Small pseudonumber (pn) to avoid division by zero
    pn = df[f"expression_mean_{column}"][df[f"expression_mean_{column}"] > 0].min()
    df[f"log2fc_{column}"] = (df[group1] + pn) / (df[group2] + pn)
    df[f"log2fc_{column}"] = np.log2(df[f"log2fc_{column}"])
    df[f"pval_{column}"] = np.nan

    for j, gene in enumerate(tqdm(df.index, desc=f"Running {method} on [{column}]")):
        if (df.loc[gene, group1] == 0) and (df.loc[gene, group2] == 0):
            p_value = np.nan
        else:
            cur_gene_group1 = group1_exp[:, j]
            cur_gene_group2 = group2_exp[:, j]
            if method == "wilcox":
                _, p_value = mannwhitneyu(cur_gene_group1, cur_gene_group2, alternative=alternative)
            elif method == "t_test":
                _, p_value = ttest_ind(cur_gene_group1, cur_gene_group2, alternative=alternative)
            else:
                p_value = np.nan
        df.loc[gene, f"pval_{column}"] = p_value

    if inplace:
        adata.var = adata.var.join(df, how="left")

    return df



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
    

def find_markers(exp_df, celltypes=None, ratio_thresh=2, exp_thresh=0,
                 chosen_fun="max",other_fun="max",ignore=None):
    '''
    Finds markers of celltype/s based on signature matrix:
        * exp_df - dataframe, index are genes, columns are celltypes
        * celltypes (str or list) - column name/names of the chosen celltype/s
        * ratio_thresh - ratio is chosen/other 
        * exp_thresh - process genes that are expressed above X in the chosen celltype/s
        * chosen_fun, other_fun - either "mean" or "max"
        * ignore (list) - list of celltypes to ignore in the "other" group
    '''
    if "gene" in exp_df.columns:
        exp_df.index = exp_df["gene"]
        del exp_df["gene"]
    if not celltypes:
        print(exp_df.columns)
        return
    exp_df = matnorm(exp_df)
    chosen = exp_df[celltypes]
    if isinstance(celltypes, list):
        other_names = exp_df.columns[~exp_df.columns.isin(celltypes)]
        if chosen_fun == "max":
            chosen = chosen.max(axis=1)
        elif chosen_fun == "mean":
            chosen = chosen.mean(axis=1)
    else:
        other_names = exp_df.columns[exp_df.columns != celltypes]
    if ignore:
        other_names = [name for name in other_names if name not in ignore]
    other = exp_df[other_names]
    if other_fun == "max":
        other = other.max(axis=1)
    elif other_fun == "mean":
        other = other.mean(axis=1)
    
    pn = chosen[chosen>0].min()
    markers_df = pd.DataFrame({"chosen_cell":chosen,"other":other},index=exp_df.index.copy())
    markers_df = markers_df.loc[markers_df["chosen_cell"] >= exp_thresh]
    markers_df["ratio"] = (chosen+pn) / (other+pn)
    markers_df["gene"] = markers_df.index
    genes = markers_df.index[markers_df["ratio"] >= ratio_thresh].tolist()
    plot = markers_df.copy()
    plot["chosen_cell"] = np.log10(plot["chosen_cell"])
    plot["other"] = np.log10(plot["other"])

    text = True if len(genes) <= 120 else False
    ax = HiVis_plot.plot_scatter_signif(plot, "chosen_cell", "other",
                                           genes,text=text,color="lightgray",
                                           xlab=f"log10({chosen_fun}({celltypes}))",
                                           ylab=f"log10({other_fun}(other cellstypes))")
    
    return genes, markers_df, ax


def fix_excel_gene_dates(df, handle_duplicates="mean"):
    """
    Fixes gene names in a DataFrame that Excel auto-converted to dates.
        * df - DataFrame containing gene names either in a column named "gene" or in the index.
        * handle_duplicates (str): How to handle duplicates after conversion. Options: "mean" or "first".
    """
    date_to_gene = {
        "1-Mar": "MARCH1", "2-Mar": "MARCH2", "3-Mar": "MARCH3", "4-Mar": "MARCH4", "5-Mar": "MARCH5",
        "6-Mar": "MARCH6", "7-Mar": "MARCH7", "8-Mar": "MARCH8", "9-Mar": "MARCH9",
        "1-Sep": "SEPT1", "2-Sep": "SEPT2", "3-Sep": "SEPT3", "4-Sep": "SEPT4", "5-Sep": "SEPT5",
        "6-Sep": "SEPT6", "7-Sep": "SEPT7", "8-Sep": "SEPT8", "9-Sep": "SEPT9",
        "10-Sep": "SEPT10", "11-Sep": "SEPT11", "12-Sep": "SEPT12", "15-Sep": "SEPT15",
        "10-Mar": "MARCH10", "11-Mar": "MARCH11"
    }  
    
    if 'gene' in df.columns:
        df['gene'] = df['gene'].replace(date_to_gene)  # Replace values in 'gene' column
    else:
        df.index = df.index.to_series().replace(date_to_gene)  # Replace values in the index
    if 'gene' in df.columns:
        df = df.set_index('gene')
    if handle_duplicates == "mean":
        df = df.groupby(df.index).mean()
    elif handle_duplicates == "first":
        df = df[~df.index.duplicated(keep='first')]
    if 'gene' in df.columns or 'gene' in df.index.names:
        df = df.reset_index()
    
    return df


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
                    image_fullres, image_highres, image_lowres, force=False):
    '''Saves cropped images. force - overrite existing files?'''
    def _export_image(img, path):
        nonlocal printed_message
        if not os.path.exists(path) or force:
            if not printed_message:
                print(f"[Saving cropped images] {path_image_fullres}")
                printed_message = True
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            # image = Image.fromarray(img)
            tifffile.imwrite(path, img)
            # image.save(save_path, format='TIFF')
            
    printed_message = False
    
    images = [image_fullres, image_highres, image_lowres]
    paths = [path_image_fullres, path_image_highres, path_image_lowres]
    for img, path in zip(images, paths):
        _export_image(img, path)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        adata.obs["nUMI_log10"] = np.log10(adata.obs["nUMI"])
        adata.var["nUMI_gene_log10"] = np.log10(adata.var["nUMI_gene"])
    mito_genes = adata.var_names[adata.var_names.str.startswith(mito_name_prefix)].values
    adata.obs["mito_sum"] = adata[:,adata.var.index.isin(mito_genes)].X.sum(axis=1).A1
    adata.obs["mito_percent_log10"] = np.log10((adata.obs["mito_sum"] / adata.obs["nUMI"]) * 100)
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
        warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique")
        adata = sc.read_visium(path_input_data, source_image_path=path_image_fullres)
    adata.var_names_make_unique()
    # del adata.uns["spatial"]
    
    # filter spots that are classified to be under tissue
    if on_tissue_only: 
        metadata = metadata.loc[metadata['in_tissue'] == 1,]
        adata = adata[adata.obs['in_tissue'] == 1]
    del metadata["in_tissue"] 
    
    # merge data and metadata
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    metadata.set_index('barcode', inplace=True)
    adata.obs = adata.obs.join(metadata, how='left')
    return adata


def merge_geojsons(geojsons_files, filename_out):
    '''
    Combine geopandas to one file.
    parameters:
        * geojsons_files - list of file paths
        * filename_out - name of the combined file. ends with .shp
    '''
    import geopandas as gpd
    gdfs = [gpd.read_file(file) for file in geojsons_files]
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry')
    if not filename_out.endswith(".shp"):
        filename_out += ".shp"
    combined_gdf.to_file(filename_out, driver="GPKG")


def inspect_df(df, col, n_rows=2):
    '''samples a df and return few rows (n_rows)
    from each unique value of col'''
    subset_df = df.groupby(col).apply(lambda x: x.sample(min(len(x), n_rows))).reset_index(drop=True)
    return subset_df


def pca(df, k_means=None, first_pc=1, title="PCA", number_of_genes=20):
    """
    Performs PCA on a dataframe, optionally applies k-means clustering, and generates plots.

    Parameters:
    - df: DataFrame with genes as rows and samples as columns.
    - k_means: Number of clusters for k-means clustering. If None, clustering is not performed.
    - first_pc: The first principal component to display.
    - title: Title for the PCA plot.
    - number_of_genes: Number of variable genes to plot.

    Returns:
    - A dictionary with PCA plot, elbow plot, PCA DataFrame, variance explained,
      and silhouette scores (if k_means is provided).
    """
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import numpy as np
    
    if first_pc >= df.shape[1]:
        raise ValueError(f"No PC {first_pc +1} possible in a data of {df.shape[1]} samples")

    # Filter out genes with zero expression
    df_filtered = df.loc[df.sum(axis=1) > 0]

    # Perform PCA
    pca_object = PCA()
    pca_result = pca_object.fit_transform(df_filtered.T)

    var_explained = pca_object.explained_variance_ratio_
    elbow_data = pd.DataFrame({
        'PC': np.arange(1, len(var_explained) + 1),
        'Variance': var_explained
    })

    pca_data = pd.DataFrame(pca_result, columns=[f'PC{ i +1}' for i in range(pca_result.shape[1])])
    pca_data['sample'] = df.columns

    x = f'PC{first_pc}'
    y = f'PC{first_pc + 1}'
    xlab = f'PC {first_pc} ({var_explained[first_pc - 1] * 100:.2f}%)'
    ylab = f'PC {first_pc + 1} ({var_explained[first_pc] * 100:.2f}%)'

    # K-means clustering
    if k_means is not None:
        if k_means >= pca_data.shape[0]:
            raise ValueError("k_means needs to be lower than number of samples")
        kmeans = KMeans(n_clusters=k_means, random_state=0).fit(pca_data.iloc[:, first_pc - 1: first_pc + 1])
        pca_data['cluster'] = kmeans.labels_.astype(str)
        silhouette_avg = silhouette_score(pca_data.iloc[:, first_pc - 1:first_pc + 1], kmeans.labels_)
    else:
        pca_data['cluster'] = "1"
        silhouette_avg = None

    # Plot PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_data, x=x, y=y, hue='cluster', palette='Set1', s=100)
    for i in range(pca_data.shape[0]):
        plt.text(pca_data.loc[i, x], pca_data.loc[i, y], pca_data.loc[i, 'sample'])
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if silhouette_avg is not None:
        # plt.suptitle(f'Silhouette mean score: {silhouette_avg:.2f}')
        plt.text(0.5, -0.1, f'Silhouette mean score: {silhouette_avg:.2f}', ha='center', va='center',
                 transform=plt.gca().transAxes)
    pca_plot = plt.gcf()

    # Plot Elbow
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=elbow_data, x='PC', y='Variance', marker='o')
    plt.title('Elbow Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    elbow_plot = plt.gcf()

    # Variable genes plot
    var_genes = pd.DataFrame(pca_object.components_.T, index=df_filtered.index,
                             columns=[f'PC{ i +1}' for i in range(pca_result.shape[1])])

    def plot_variable_genes(pc, num_genes):
        df_pc = var_genes[[pc]].sort_values(by=pc)
        top_bottom_genes = pd.concat([df_pc.head(num_genes), df_pc.tail(num_genes)])
        plt.figure(figsize=(10, 7))
        sns.barplot(x=top_bottom_genes.index, y=top_bottom_genes[pc], color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='black')
        plt.title(f'Top and Bottom {num_genes} Genes for {pc}')
        plt.tight_layout()
        return plt.gcf()

    genes1_plot = plot_variable_genes(x, number_of_genes)
    genes2_plot = plot_variable_genes(y, number_of_genes)

    result = {
        'plot_pca': pca_plot,
        'plot_elbow': elbow_plot,
        'pca_df': pca_data,
        'variance': elbow_data,
        'var_genes': var_genes,
        'plot_genes': [genes1_plot, genes2_plot]
    }

    if silhouette_avg is not None:
        result['silhouette'] = silhouette_avg
    print(f'keys in output: {list(result.keys())}')
    return result


def noise_mean_curve(adata, plot=False, layer=None,signif_thresh=0.95, inplace=False, **kwargs):
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
    X = sm.add_constant(exp_log)
    model = sm.OLS(cv_log, X).fit()
    residuals = cv_log - model.predict(X)
    
    df = pd.DataFrame({
        "gene": np.array(adata.var_names)[valid_genes],
        "mean": mean_expression[valid_genes],
        "mean_log": exp_log,
        "cv": cv[valid_genes],
        "cv_log": cv_log,
        "residual": residuals
    })
    
    if inplace:
        adata.var.loc[df["gene"], "cv"] = df["cv"].values
        adata.var.loc[df["gene"], "expression_mean"] = df["mean"].values
        adata.var.loc[df["gene"], "residual"] = df["residual"].values
    if plot:
        thresh = np.quantile(np.abs(residuals), signif_thresh)
        signif_genes = df.loc[np.abs(df["residual"]) > thresh, "gene"]
        ax = HiVis_plot.plot_scatter_signif(df, "mean_log", "cv_log", color="residual", genes=list(signif_genes), **kwargs)
        return df, ax
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
    if normalize:
        matrix = matnorm(matrix, "row")
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
    qval_df = pd.DataFrame(qval_mat, index=gene_list, columns=gene_list)
    
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
    def _get_dendrogram_order(data, method="average"):
        # When correlation=True, we assume data is already the appropriate
        # distance matrix (1 - correlation), and it is square.
        dist_mat = 1 - data
        dist_condensed = squareform(dist_mat.to_numpy(), checks=False)
        Z = linkage(dist_condensed, method=method)
        dend = dendrogram(Z, no_plot=True)
        return dend["leaves"]
    
    df_out = df.copy()  # so as not to mutate the original

    # For a correlation matrix, the row and column ordering should be the same.
    if correlation:
        if cluster_rows or cluster_cols:
            # Compute ordering once using the original symmetric matrix.
            order = _get_dendrogram_order(df_out, method=method)
            # Apply the same order to both rows and columns.
            df_out = df_out.iloc[order, order]
    else:
        # Normal clustering on data (not a correlation matrix)
        if cluster_rows:
            row_order = _get_dendrogram_order(df_out, method=method) if cluster_rows else None
            df_out = df_out.iloc[row_order, :]
        if cluster_cols:
            col_order = _get_dendrogram_order(df_out.T, method=method) if cluster_cols else None
            df_out = df_out.iloc[:, col_order]

    return df_out


def estimate_dense_memory(matrix):
    '''return size (in GB) of a sparse matrix upon convertion'''
    n_rows, n_cols = matrix.shape
    element_size = matrix.dtype.itemsize  # e.g., 8 for float64
    total_bytes = n_rows * n_cols * element_size
    total_gb = total_bytes / (1024 ** 3)
    return total_gb




