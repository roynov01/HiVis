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
# from PIL import Image
import math
import warnings
import scanpy as sc
import tifffile
import matplotlib.pyplot as plt
import ViziumHD_plot
from matplotlib.colors import to_rgba


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
             
             
             
def dge(adata, column, group1, group2=None, umi_thresh=0,
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
    print(f'Normilizing "{group1}" spots')
    group1_exp.X = group1_exp.X / group1_exp.X.sum(axis=1).A1[:, None]  # matnorm
    group1_exp = group1_exp.X.todense()
    df[group1] = group1_exp.mean(axis=0).A1  # save avarage expression of group1 to vars
        
    if group2 is None:
        group2_exp = adata[(adata.obs[column] != group1) & ~adata.obs[column].isna()].copy()
        group2 = "rest"
    else:
        group2_exp = adata[adata.obs[column] == group2].copy()
    group2_exp = group2_exp[group2_exp.X.sum(axis=1) > umi_thresh]  # delete empty spots

    print(f'Normilizing "{group2}" spots')
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
    ax = ViziumHD_plot.plot_scatter_signif(plot, "chosen_cell", "other",
                                           genes,text=text,color="lightgray",
                                           xlab=f"log10({chosen_fun}({celltypes}))",
                                           ylab=f"log10({other_fun}(other cellstypes))")
    
    return genes, markers_df, ax


def fix_excel_gene_dates(df, handle_duplicates="mean"):
    """
    Fixes gene names in a DataFrame that Excel auto-converted to dates.
        * df (pd.DataFrame): DataFrame containing gene names either in a column named "gene" or in the index.
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
        if not os.path.exists(path) or force:
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            # image = Image.fromarray(img)
            tifffile.imwrite(path, img)
            # image.save(save_path, format='TIFF')
    print("[Saving cropped images]")
    images = [image_fullres, image_highres, image_lowres]
    paths = [path_image_fullres, path_image_highres, path_image_lowres]
    for img, path in zip(images, paths):
        _export_image(img, path)
    return images
              

def _edit_adata(adata, scalefactor_json, mito_name_prefix):
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
    del adata.uns["spatial"]
    
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

