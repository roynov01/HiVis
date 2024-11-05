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
from statsmodels.stats.multitest import multipletests
from PIL import Image
import math
import warnings
import scanpy as sc
import tifffile
import matplotlib.pyplot as plt


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
    

def matnorm(df):
    if isinstance(df, pd.core.series.Series):
        return df.div(df.sum())
    if isinstance(df, (np.ndarray, np.matrix)):
        column_sums = df.sum(axis=0)
        column_sums[column_sums == 0] = 1
        return df / column_sums
    if isinstance(df, pd.core.frame.DataFrame):
        numeric_columns = df.select_dtypes(include='number')
        column_sums = numeric_columns.sum(axis=0)
        column_sums[column_sums == 0] = 1  # Avoid division by zero
        normalized_df = numeric_columns.divide(column_sums, axis=1)
        return normalized_df.astype(np.float32)
    if isinstance(df, list):
        return (pd.Series(df) / sum(df)).tolist()
    else: # pandas
        raise ValueError("df is not a list,numpy or a dataframe")
        

 
    
def validate_exists(file_path):
    if isinstance(file_path, (list, tuple)):
        for path in file_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No such file or directory:\n\t{path}")
    else:
         if not os.path.exists(file_path):
             raise FileNotFoundError(f"No such file or directory:\n\t{file_path}")    
             
             
             
def find_markers(adata, column, group1, group2=None, umi_thresh=0,
                 method="wilcox",alternative="two-sided",inplace=False):
    '''
    Runs differential gene expression analysis between two groups.
    Values will be saved in self.var: expression_mean, log2fc, pval
    parameters:
        * column - which column in obs has the groups classification
        * group1 - specific value in the "column"
        * group2 - specific value in the "column". 
                   if None,will run agains all other values, and will be called "rest"
        * method - either "wilcox" or "t_test"
        * alternative - {"two-sided", "less", "greater"}
        * umi_thresh - use only spots with more UMIs than this number
        * inplace - modify the adata.var with log2fc, pval and expression columns?
    '''
    df = adata.var.copy()
        
    # Get the expression of the two groups
    group1_exp = adata[adata.obs[column] == group1].copy()
    group1_exp = group1_exp[group1_exp.X.sum(axis=1) > umi_thresh]  # delete low quality spots
    print(f'normilizing "{group1}" spots')
    group1_exp.X = group1_exp.X / group1_exp.X.sum(axis=1).A1[:, None]  # matnorm
    group1_exp = group1_exp.X.todense()
    df[group1] = group1_exp.mean(axis=0).A1  # save avarage expression of group1 to vars
    
    # df[group1 + "_med"] = np.median(group1_exp, axis=0).A1
    
    if group2 is None:
        group2_exp = adata[(adata.obs[column] != group1) & ~adata.obs[column].isna()].copy()
        group2 = "rest"
    else:
        group2_exp = adata[adata.obs[column] == group2].copy()
    group2_exp = group2_exp[group2_exp.X.sum(axis=1) > umi_thresh]  # delete empty spots

    print(f'normilizing "{group2}" spots')
    group2_exp.X = group2_exp.X / group2_exp.X.sum(axis=1).A1[:, None]  # matnorm
    group2_exp = group2_exp.X.todense()
    df[group2] = group2_exp.mean(axis=0).A1  # save avarage expression of group2 to vars
    # df[group2 + "_med"] = np.median(group2_exp, axis=0).A1
    # df[group2+"_med"] = group2_exp.median(axis=0).A1  
    print(f"Number of spots in group1: {group1_exp.shape}, in group2: {group2_exp.shape}")
    # Calculate mean expression in each group and log2(group1/group2)
    df[f"expression_mean_{column}"] = df[[group1,group2]].mean(axis=1)
    pn = df[f"expression_mean_{column}"][df[f"expression_mean_{column}"]>0].min()
    df[f"log2fc_{column}"] = (df[group1] + pn) / (df[group2] + pn)
    df[f"log2fc_{column}"] = np.log2(df[f"log2fc_{column}"])  
    
    # df[f"log2fc_med_{column}"] = (df[group1+"_med"] + pn) / (df[group2+"_med"] + pn)
    # df[f"log2fc_med_{column}"] = np.log2(df[f"log2fc_med_{column}"]) 
    
    # Wilcoxon rank-sum test
    df[f"pval_{column}"] = np.nan
    for j, gene in enumerate(tqdm(df.index, desc=f"Running wilcoxon on [{column}]")):
        if (df.loc[gene,group1] == 0) and (df.loc[gene,group2] == 0):
            p_value = np.nan
        else:
            cur_gene_group1 = group1_exp[:,j]
            cur_gene_group2 = group2_exp[:,j]
            if method == "wilcox":
                from scipy.stats import mannwhitneyu
                _, p_value = mannwhitneyu(cur_gene_group1, cur_gene_group2, alternative=alternative)
            elif method == "t_test":
                from scipy.stats import ttest_ind
                _, p_value = ttest_ind(cur_gene_group1, cur_gene_group2, alternative=alternative)
        df.loc[gene,f"pval_{column}"] = p_value
    if inplace:
        adata.var = df.copy()
    return df
    

def _crop_images_permenent(self, adata, image_fullres, image_highres, image_lowres, scalefactor_json):
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
                    image_fullres, image_highres, image_lowres):
    '''Saves cropped images'''
    print("[Saving cropped images]")
    images = [image_fullres, image_highres, image_lowres]
    paths = [path_image_fullres, path_image_highres, path_image_lowres]
    for img, path in zip(images, paths):
        export_image(img, path)
           

def export_image(img, path):
    fileformat = "." + path.split(".")[1]
    save_path = path.replace(fileformat, "_cropped.tif")
    if not os.path.exists(save_path):
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img)
        image.save(save_path, format='TIFF') 
        
    
    
    

def _edit_adata(adata, scalefactor_json, mito_name_prefix):
    adata.obs["pxl_col_in_lowres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
    adata.obs["pxl_row_in_lowres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
    adata.obs["pxl_col_in_highres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
    adata.obs["pxl_row_in_highres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
    adata.obs["um_x"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["microns_per_pixel"]
    adata.obs["um_y"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["microns_per_pixel"]

    # Quality control - number of UMIs and mitochondrial %
    adata.obs["nUMI"] = np.array(adata.X.sum(axis=1).flatten())[0]
    adata.var["nUMI"] = np.array(adata.X.sum(axis=0).flatten())[0]
    mito_genes = adata.var_names[adata.var_names.str.startswith(mito_name_prefix)].values
    adata.obs["mito_sum"] = adata[:,adata.var.index.isin(mito_genes)].X.sum(axis=1).A1
    adata.obs["mito_percent"] = (adata.obs["mito_sum"] / adata.obs["nUMI"]) * 100
    return adata


def _import_data(metadata_path, path_input_data, path_image_fullres, on_tissue_only):
    # load metadata  
    print("[Loading metadata]")        
    metadata = pd.read_parquet(metadata_path)
    if not os.path.isfile(metadata_path.replace(".parquet",".csv")):
        metadata.to_csv(metadata_path.replace(".parquet",".csv"),index=False)

    del metadata["array_row"]
    del metadata["array_col"]
    
    # load data
    print("[Loading data]")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique")
        adata = sc.read_visium(path_input_data, source_image_path=path_image_fullres)
    adata.var_names_make_unique()
    
    if on_tissue_only: # filter spots that are classified to be under tissue
        metadata = metadata.loc[metadata['in_tissue'] == 1,]
        adata = adata[adata.obs['in_tissue'] == 1]
    del metadata["in_tissue"] 
    
    # merge data and metadata
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    metadata.set_index('barcode', inplace=True)
    adata.obs = adata.obs.join(metadata, how='left')
    return adata


def load_images(path_image_fullres, path_image_highres, path_image_lowres):
    print("[Loading images]")
    image_fullres = tifffile.imread(path_image_fullres)
    rgb_dim = image_fullres.shape.index(3)
    if rgb_dim != 2:  # If the RGB dimension is not already last
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
    return image_fullres, image_highres, image_lowres
    
