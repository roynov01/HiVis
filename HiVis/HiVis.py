# -*- coding: utf-8 -*-
"""
HD Integrated Visium Interactive Suite (HiVis)
"""
# General libraries
import os
import dill
import gc
import warnings

from tqdm import tqdm
from copy import deepcopy
# Data libraries
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt, affinity
import anndata as ad
# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image


from . import HiVis_utils
from .Aggregation import Aggregation
from . import HiVis_plot
from . import Aggregation_utils

Image.MAX_IMAGE_PIXELS = 1063425001 # Enable large images loading


def load(filename, directory=''):
    '''
    loads an instance from a pickle format, that have been saved via HiVis.save()
    
    Parameters:
        * filename (str)- full path of pkl file, or just the filename if directory is specified
        
    **Returns** HiVis instance
    '''
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"
    if directory:
        filename = f"{directory}/{filename}"
    HiVis_utils.validate_exists(filename)
    with open(filename, "rb") as f:
        instance = dill.load(f)
    return instance

def new(path_image_fullres:str, path_input_data:str, path_output:str,
             name:str, crop_images=True, properties: dict = None, on_tissue_only=True,min_reads_in_spot=1,
             min_reads_gene=10, fluorescence=False, plot_qc=True):
    '''
    - Loads images, data and metadata.
    - Initializes the connection from the data and metadata to the images coordinates
    - Adds basic QC to the metadata (nUMI, mitochondrial %)
    
    Parameters:
        * path_input_fullres_image (str) - path of the full resolution microscopy image
        * path_input_data (str) - folder with outs of the Visium. Typically square_002um \
                            (with h5 files and with folders filtered_feature_bc_matrix, spatial)
        * path_output (str) - path where to save plots and files
        * name (str) - name of the instance
        * crop_images (bool) - crop the regions outside of the spots cover area
        * properties (dict) - can be any metadata, such as organism, organ, sample_id
        * on_tissue_only (bool) - remove spots that are not classified as "on tissue"
        * min_reads_in_spot (int) - filter out spots with less than X UMIs
        * min_reads_gene (int) - filter out gene that is present in less than X spots
        * fluorescence - either False for H&E, or a dict of channel names and colors. color can be None. Example {"DAPI":"blue"}
        * plot_qc (bool) - plot QC when object is being created
        
    **Returns** HiVis instance
    '''
    # Validate paths of metadata and images
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    path_image_highres = path_input_data + "/spatial/tissue_hires_image.png"
    path_image_lowres = path_input_data + "/spatial/tissue_lowres_image.png"
    json_path = path_input_data + "/spatial/scalefactors_json.json"
    metadata_path = path_input_data + "/spatial/tissue_positions.parquet"
    HiVis_utils.validate_exists([path_image_fullres,path_image_highres,path_image_lowres,json_path,metadata_path])
    
    # Load images
    image_fullres, image_highres, image_lowres = HiVis_utils.load_images(path_image_fullres, path_image_highres, path_image_lowres)
    
    # Load scalefactor_json
    with open(json_path) as file:
        scalefactor_json = json.load(file)
    
   # Load data + metadata
    adata = HiVis_utils._import_data(metadata_path, path_input_data, path_image_fullres, on_tissue_only)
    
    if crop_images:
        # Crop images
        adata, image_fullres, image_highres, image_lowres = HiVis_utils._crop_images_permenent(
            adata, image_fullres, image_highres, image_lowres, scalefactor_json)
        
        # Save cropped images
        path_image_fullres_cropped = path_image_fullres.replace("." + path_image_fullres.split(".")[-1], "_cropped.tif")
        path_image_highres_cropped = path_image_highres.replace("." + path_image_highres.split(".")[-1], "_cropped.tif")
        path_image_lowres_cropped = path_image_lowres.replace("." + path_image_lowres.split(".")[-1], "_cropped.tif")
        HiVis_utils._export_images(path_image_fullres_cropped, path_image_highres_cropped, 
                                      path_image_lowres_cropped,image_fullres,
                                      image_highres, image_lowres)
    
    if fluorescence:
        HiVis_utils._measure_fluorescence(adata, image_fullres, list(fluorescence.keys()), scalefactor_json["spot_diameter_fullres"])

    # Add QC (nUMI, mito %) and unit transformation
    mito_name_prefix = "MT-" if properties.get("organism") == "human" else "mt-"
    HiVis_utils._edit_adata(adata, scalefactor_json, mito_name_prefix)

    # Filter low quality spots and lowly expressed genes
    adata = adata[adata.obs["nUMI"] >= min_reads_in_spot, adata.var["nUMI_gene"] >= min_reads_gene].copy()

    return HiVis(adata, image_fullres, image_highres, image_lowres, scalefactor_json, 
                    name, path_output, properties, agg=None, fluorescence=fluorescence, plot_qc=plot_qc)


class HiVis:
    '''
    Main class. Stores the data and images of the VisiumHD, enables plotting via HiVis.plot, \
    and can store Aggregation instances in HiVis.agg.
    
    To make a new class, call the new() function.
    '''
    def __init__(self, adata, image_fullres, image_highres, image_lowres, scalefactor_json, 
                 name, path_output, properties=None, agg=None, fluorescence=False, plot_qc=True):
        self.agg = agg
        self.name, self.path_output = name, path_output 
        self.properties = properties if properties else {}
        self.organism = self.properties.get("organism")
        if isinstance(image_fullres, str): # paths of images, not the images themselves
            image_fullres, image_highres, image_lowres = HiVis_utils.load_images(image_fullres, image_highres, image_lowres)
        
        self.image_fullres, self.image_highres, self.image_lowres = image_fullres, image_highres, image_lowres
        self.fluorescence = fluorescence
        
        if isinstance(scalefactor_json, str):
            with open(scalefactor_json) as file:
                scalefactor_json = json.load(file)        
        self.json = scalefactor_json
            
        self.adata = adata
        adata.obs["pxl_col_in_lowres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
        adata.obs["pxl_row_in_lowres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
        adata.obs["pxl_col_in_highres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
        adata.obs["pxl_row_in_highres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
        adata.obs["um_x"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["microns_per_pixel"]
        adata.obs["um_y"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["microns_per_pixel"]
        
        self.plot = HiVis_plot.PlotVisium(self)
        if fluorescence:
            self.image_fullres_orig = self.image_fullres.copy()
            self.recolor(fluorescence)
        else:
            self.plot._init_img()
        if plot_qc:
            self.qc(save=True)
            plt.show()
    
    def recolor(self, fluorescence=None, normalization_method="percentile"):
        '''
        Recolors a flurescence image
        
        Parameters:
            * fluorescence is either list of colors or dict {channel: color...}. color can be None.
            * normalization_method - {"percentile", "histogram","clahe","sqrt" or None for minmax}
        '''
        if not self.fluorescence:
            raise ValueError("recolor() works for fluorescence visium only")
        if not fluorescence:
            fluorescence = self.fluorescence
        channels = list(self.fluorescence.keys())    
        if isinstance(fluorescence, list):
            if len(fluorescence) != len(channels):
                raise ValueError(f"Flurescence should include all channels: {channels}")
            self.fluorescence = {channels[i]:fluorescence[i] for i in range(len(channels))}
        elif isinstance(fluorescence, dict):
            if list(fluorescence.keys()) != channels:
                raise ValueError(f"Flurescence should include all channels: {channels}")
            self.fluorescence = fluorescence
        self.image_fullres = HiVis_utils.fluorescence_to_RGB(self.image_fullres_orig, 
                                                                self.fluorescence.values(), 
                                                                normalization_method)
        self.plot._init_img()


    def qc(self, save=False,figsize=(8, 8)):
        '''
        Plots basic QC (spatial, nUMI, mitochondrial %)
        
        Parameters:
            * save (bool) - save the plot in HiVis.path_output
        '''
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2,nrows=2, figsize=figsize)
        ax0 = self.plot.spatial(title=self.name, ax=ax0)
        ax1 = self.plot.hist("mito_percent_log10", title="Mitochondrial content per spot", xlab="log10(Mito %)",ax=ax1)
        ax2 = self.plot.hist("nUMI_log10", title="Number of UMIs per spot", xlab="log10(UMIs)",ax=ax2)
        ax3 = self.plot.hist("nUMI_gene_log10", title="Number of UMIs per gene", xlab="log10(UMIs)",ax=ax3)
        plt.tight_layout()
        if save:
            self.plot.save(figname="QC", fig=fig)
    
    def add_mask(self, mask_path:str, name:str, plot=True, cmap="Paired"):
        '''
        assigns each spot a value based on mask (image).
        
        Parameters:
            * mask_path (str) - path to mask image
            * name (str) - name of the mask (that will be called in the metadata)
            * plot (bool) - plot the mask
            * cmap (str) - colormap for plotting
            
        **Returns** the mask (np.array)
        '''
        HiVis_utils.validate_exists(mask_path)
        
        def _import_mask(mask_path):
            '''imports the mask'''
            print("[Importing mask]")
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            return mask_array
        
        def _plot_mask(mask_array, cmap):
            '''plots the mask'''
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_array, cmap=cmap)
            num_colors = len(np.unique(mask_array[~np.isnan(mask_array)]))
            cmap = plt.cm.get_cmap(cmap, num_colors) 
            legend_elements = [Patch(facecolor=cmap(i), label=f'{i}') for i in range(num_colors)]
            plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.5))
            plt.show()

        def _assign_spots(mask_array, name):
            '''assigns each spot a value from the mask'''
            def _get_mask_value(mask_array, x, y):
                if (0 <= x < mask_array.shape[1]) and (0 <= y < mask_array.shape[0]):
                    return mask_array[y, x]
                else:
                    return None
            def _get_spot_identity(row):
                x = round(row['pxl_col_in_fullres'])
                y = round(row['pxl_row_in_fullres'])
                return _get_mask_value(mask_array, x, y)
            
            tqdm.pandas(desc=f"Assigning spots identity [{name}]")
            
            self.adata.obs[name] = np.nan
            self.adata.obs[name] = self.adata.obs.progress_apply(
                _get_spot_identity, axis=1)

        mask_array = _import_mask(mask_path)
        if plot:
            _plot_mask(mask_array, cmap=cmap)
        _assign_spots(mask_array, name)
        self.plot._init_img()
        print(f"\nTo rename the values in the metadata, call the [update_meta] method with [{name}] and dictionary with current_name:new_name")
        return mask_array
    

    def add_annotations(self, path:str, name:str, measurements=True):
        '''
        Adds annotations made in Qupath (geojson)
        
        Parameters:
            * path (str) - path to geojson file
            * name (str) - name of the annotation (that will be called in the obs)
            * measurements (bool) - include measurements columns 
        '''
        HiVis_utils.validate_exists(path)
        annotations = gpd.read_file(path)
        if "classification" in annotations.columns:
            annotations["classification"] = annotations["classification"].apply(json.loads)
            annotations[name] = [x["name"] for x in annotations["classification"]]
        else:
            annotations[name] = annotations.index
        annotations[f"{name}_id"] = annotations["id"]
        del annotations["id"]
        del annotations["objectType"]
        if "isLocked" in annotations.columns:
            del annotations["isLocked"]
        
        if "measurements" in annotations.columns and measurements:
            measurements_df = pd.json_normalize(annotations["measurements"])
            annotations = gpd.GeoDataFrame(pd.concat([annotations.drop(columns=["measurements"]), measurements_df], axis=1))
            perimeter = annotations.geometry.length
            area = annotations.geometry.area
            annotations["circularity"] = (4 * np.pi * area) / (perimeter ** 2)
            annotations.loc[perimeter == 0, "circularity"] = np.nan
            cols = list(measurements_df.columns) + ["circularity",name,f"{name}_id"]
        else:
            if measurements:
                print("No measurements found")
            cols = [name,f"{name}_id"]
        for col in cols:
            if col in self.adata.obs.columns:
                del self.adata.obs[col]
        obs = gpd.GeoDataFrame(self.adata.obs, 
              geometry=gpd.points_from_xy(self.adata.obs["pxl_col_in_fullres"],
                                          self.adata.obs["pxl_row_in_fullres"]))        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            merged_obs = gpd.sjoin(obs,annotations,how="left",predicate="within")
        merged_obs = merged_obs[~merged_obs.index.duplicated(keep="first")]
        
        self.adata.obs = self.adata.obs.join(pd.DataFrame(merged_obs[cols]),how="left")
        self.plot._init_img()
    
        
    def dge(self, column, group1, group2=None, method="wilcox", two_sided=False,
            umi_thresh=0, inplace=False):
        '''
        Runs differential gene expression analysis between two groups.
        
        Parameters:
            * column (str) - which column in obs has the groups classification
            * group1 - specific value in the "column"
            * group2 - specific value in the "column". \
                       if None, will run against all other values, and will be called "rest"
            * method - either "wilcox" or "t_test"
            * two_sided (bool) - if one sided, will give the pval for each group, \
                          and the minimal of both groups (which will also be FDR adjusted)
            * umi_thresh (int) - use only spots with more UMIs than this number
            * inplace (bool) - modify the adata.var with log2fc, pval and expression columns
            
        **Returns** the DGE results (pd.DataFrame)
        '''
        alternative = "two-sided" if two_sided else "greater"
        df = HiVis_utils.dge(self.adata, column, group1, group2, umi_thresh,
                     method=method, alternative=alternative, inplace=inplace)
        if group2 is None:
            group2 = "rest"
        df = df[[f"pval_{column}",f"log2fc_{column}",group1,group2]]
        df.rename(columns={f"log2fc_{column}":"log2fc"},inplace=True)
        if not two_sided:
            df[f"pval_{group1}"] = 1 - df[f"pval_{column}"]
            df[f"pval_{group2}"] = df[f"pval_{column}"]
            df["pval"] = df[[f"pval_{group1}",f"pval_{group2}"]].min(axis=1)
        else:
            df["pval"] = df[f"pval_{column}"]
        del df[f"pval_{column}"]
        df["qval"] = HiVis_utils.p_adjust(df["pval"])
        df["expression_mean"] = df[[group1, group2]].mean(axis=1)
        df["expression_min"] = df[[group1, group2]].min(axis=1)
        df["expression_max"] = df[[group1, group2]].max(axis=1)
        df["gene"] = df.index
        if inplace:
            var = df.copy()
            var.rename(columns={
                "qval":f"qval_{column}",
                "pval":f"pval_{column}",
                "log2fc":f"log2fc_{column}",
                "expression_mean":f"expression_mean_{column}",
                "expression_min":f"expression_min_{column}",
                "expression_max":f"expression_max_{column}",
                },inplace=True)
            del var["gene"]
            self.adata.var = self.adata.var.join(var, how="left")
        return df
    
    
    def add_agg(self, adata_agg, name):
        '''
        Creates and adds Aggregation to the HiVis instance. Can be accessed by self.agg[name].
        For example single-cells, tissue structures.
        
        Parameters:
            * adata_agg (ad.AnnData) - anndata containing aggregations
            * name (str) - name of the aggregation object
        '''
        if not isinstance(adata_agg, ad.AnnData):
            raise TypeError("adata_agg must be anndata")
        if self.agg:
            if name in self.agg:
                print(f"{name} allready in {self.name}. Renamed previous Agg to 'temp'.")
                self.agg["temp"] = self.agg[name]
                del self.agg[name]
        else:
            self.agg = {}
        agg_name = f"{self.name}_{name}"
        agg = Aggregation(self, adata_agg, name=agg_name)
        self.agg[name] = agg
        
    
    def agg_stardist(self, input_df, name="SC", obs2add=None, obs2agg=None):
        '''
        Adds Aggregation object to self.agg[name], based on CSV output of Stardist pipeline.
        
        Parameters:
            * input_df (pd.DataFrame) - output of Stardist pipeline 
            * name (str) - name to store the Aggregation in. Can be accessed via HiVis.agg[name]
            * obs2agg - what obs to aggregate from the HiVis. \
                        Can be a list of column names. numeric columns will be summed, categorical will be the mode. \
                        Can be a dictionary specifying the aggregation function. \
                        examples: {"value_along_axis":np.median} or {"value_along_axis":[np.median,np.mean]}
            * obs2add (list) - which columns from input_df should be copied to the Aggregation.adata.obs
        '''
        spots_only, cells_only = Aggregation_utils.split_stardist(input_df)
        
        self.adata.obs = self.adata.obs.join(spots_only,how="left")
        
        aggregation_func = Aggregation_utils._aggregate_data_stardist

        adata_agg, _ = Aggregation_utils.new_adata(self.adata, "Cell_ID", aggregation_func,
                                       obs2agg=obs2agg,in_cell_col="in_cell",nuc_col="in_nucleus")
        
        obs2add = [col for col in cells_only.columns if col in obs2add]
        Aggregation_utils.merge_cells(cells_only, adata_agg, obs2add)
        
        self.add_agg(adata_agg, name=name)
    
    
    def add_meta(self, name:str, values, type_="obs"):
        '''
        Adds a vector to metadata (obs or var)
        
        Parameters:
            * name (str) - name of metadata
            * values (array like) - values to add
            * type\_ - either "obs" or "var"
        '''
        if type_ == "obs":
            if name in self.adata.obs.columns:
                raise ValueError(f"[{name}] allready present in adata.obs")
            self.adata.obs[name] = values
        elif type_ == "var":
            if name in self.adata.var.columns:
                raise ValueError(f"[{name}] allready present in adata.var")
            self.adata.var[name] = values
        self.plot._init_img()
    
    def update_meta(self, name:str, values:dict, type_="obs"):
        '''
        Updates values in metadata (obs or var)
        
        Parameters:
            * name (str) - name of metadata
            * values (dict) -{old_value:new_value}
            * type\_ - either "obs" or "var"
        '''
        if type_ == "obs":
            if name not in self.adata.obs.columns:
                raise ValueError(f"No metadata called [{name}] in obs")
            original_dtype = self.adata.obs[name].dtype
            self.adata.obs[name] = self.adata.obs[name].apply(lambda x: values.get(x, x) if pd.notna(x) else x)
            
            # Convert back to original dtype if it was categorical
            if pd.api.types.is_categorical_dtype(original_dtype):
                self.adata.obs[name] = self.adata.obs[name].astype('category') 
        elif type_ == "var":
            if name not in self.adata.var.columns:
                raise ValueError(f"No metadata called [{name}] in var")
            original_dtype = self.adata.var[name].dtype
            self.adata.var[name] = self.adata.var[name].apply(lambda x: values.get(x, x) if pd.notna(x) else x)
            
            # Convert back to original dtype if it was categorical
            if pd.api.types.is_categorical_dtype(original_dtype):
                self.adata.var[name] = self.adata.var[name].astype('category')    
        else:
            raise ValueError("type_ must be either 'obs' or 'var'")
        self.plot._init_img()
        
    
    def pseudobulk(self, by=None, layer=None):
        '''
        Sums the gene expression for each group in a single obs.
        
        Parameters:
            
            * by (str) - return a dataframe, each column is a value in "by" (for example cluster), rows are genes. \
            If None, will return the mean expression of every gene. 
            * layer (str) - which layer in adata to use.
            
        **Returns** the gene expression for each group (pd.DataFrame)
        '''
        if layer is None:
            x = self.adata.X
        else:
            if layer not in self.adata.layers:
                raise KeyError(f"Layer '{layer}' not found in self.adata.layers. Available layers: {list(self.adata.layers.keys())}")
            x = self.adata.layers[layer]
        if by is None:
            pb = x.mean(axis=0).A1
            return pd.Series(pb, index=self.adata.var_names)
        
        unique_groups = self.adata.obs[by].unique()
        unique_groups = unique_groups[pd.notna(unique_groups)]

        n_groups = len(unique_groups)
        n_genes = self.adata.n_vars  
        result = np.zeros((n_groups, n_genes))
        for i, group in enumerate(unique_groups):
            mask = (self.adata.obs[by] == group).values
            if mask.sum() == 0: 
                continue
            group_sum = x[mask].sum(axis=0)  
            group_mean = group_sum / mask.sum() 
            result[i, :] = group_mean.A1     
        return pd.DataFrame(result.T, index=self.adata.var_names, columns=unique_groups)
    
    def noise_mean_curve(self, plot=False, layer=None, signif_thresh=0.95, inplace=False, **kwargs):
        '''
        Generates a noise-mean curve of the data.
        
        Parameters:
            * plot (bool) - plot the curve
            * layer - which layer in the AnnData to use
            * signif_thresh (float) - for plotting, add text for genes in this residual percentile
            * inplace (bool) - add the mean_expression, cv and residuals to VAR
            
        **Returns** dataframe with expression, CV and residuals of each gene (pd.DataFrame). \
            If plot=true, will also return ax.
        '''
        return HiVis_utils.noise_mean_curve(self.adata, plot=plot,layer=layer,
                                               signif_thresh=signif_thresh,inplace=inplace, **kwargs)
    
    def cor(self, what, self_corr_value=None, normilize=True, layer: str = None, inplace=False):
        '''
        Calculates gene(s) correlation.
        
        Parameters:
            * what (str or list) - if str, computes Spearman correlation of a given gene with all genes. \
                                    if list, will compute correlation between all genes in the list
            * self_corr_value - replace the correlation of the gene with itself by this value
            * normalize (bool) - normilize expression before computing correlation
            * layer (str) - which layer in the AnnData to use
            * inplace (bool) - add the correlation to VAR
            
        **Returns** dataframe of spearman correlation between genes (pd.DataFrame)
        '''
        if isinstance(what, str):
            x = self[what]
            return HiVis_utils.cor_gene(self.adata, x, what, self_corr_value, normilize, layer, inplace)
        return HiVis_utils.cor_genes(self.adata, what, self_corr_value, normilize, layer)
        
                
    def export_h5(self, path=None, force=False):
        '''
        Exports the adata as h5ad.
        
        Parameters:
            * path (str) - path to save the h5 file. If None, will save to path_output
            * force (bool) - save file even if it already exists
            
        **Returns** path where the file was saved (str)
        '''
        if path is None:
            path = self.path_output
        path = f"{path}/{self.name}_HiVis.h5ad"
        if not os.path.exists(path) or force:
            print("[Writing h5]")
            self.adata.write(path)
        return path
    
    def export_images(self, path=None, force=False):
        '''
        Exports full, high and low resolution images
        
        Parameters:
            * path (str) - path to save the image files. If None, will save to path_output
            * force (bool)- save files even if they already exists
            
        **Returns** list of [3 images (np.array) and the spatial json]
        '''
        if path is None:
            path = self.path_output
        if not os.path.exists(path):
            os.makedirs(path)
        path_image_fullres = f"{path}/{self.name}_fullres.tif"
        image_fullres = self.image_fullres_orig if self.fluorescence else self.image_fullres
        path_image_highres = f"{path}/{self.name}_highres.tif"
        path_image_lowres = f"{path}/{self.name}_lowres.tif"
        images = HiVis_utils._export_images(path_image_fullres, path_image_highres, 
                                      path_image_lowres, image_fullres, 
                                      self.image_highres, self.image_lowres, force=force)
        
        path_json = f"{path}/{self.name}_scalefactors_json.json"
        with open(path_json, 'w') as file:
            json.dump(self.json, file, indent=4)
        
        cols = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres','pxl_col_in_fullres']
        path_obs = f"{path}/{self.name}_tissue_positions.csv"
        self.adata.obs[cols].to_csv(path_obs, index=True, index_label="barcode")
        
        images.append(self.json)
        
        return images

    
    def get(self, what, cropped=False):
        '''
        Get a vector from data (a gene) or metadata (from obs or var). or subset the object.
        
        Parameters:
            * what - if string, will get data or metadata. else, will return a new HiVis object that is spliced. \
                     the splicing is passed to the self.adata
            * cropped - get the data from the adata_cropped after plotting spatial
            
        **Returns**: either np.array of data or, if subsetting, a new HiVis instance
        '''
        adata = self.adata_cropped if cropped else self.adata
        if isinstance(what, str):  # Easy access to data or metadata arrays
            if what in adata.obs.columns:  # Metadata
                column_data = adata.obs[what]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype(str).values
                return column_data.values
            if what in adata.var.index:  # A gene
                return np.array(adata[:, what].X.todense().ravel()).flatten()
            if what in adata.var.columns:  # Gene metadata
                column_data = adata.var[what]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype(str).values
                return column_data.values
            obs_cols_lower = adata.obs.columns.str.lower()
            if what.lower() in obs_cols_lower:
                col_name = adata.obs.columns[obs_cols_lower.get_loc(what.lower())]
                column_data = adata.obs[col_name]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype(str).values
                return column_data.values
            if self.organism == "mouse" and (what.lower().capitalize() in adata.var.index):
                return np.array(adata[:, what.lower().capitalize()].X.todense()).flatten()
            if self.organism == "human" and (what.upper() in adata.var.index):
                return np.array(adata[:, what.upper()].X.todense()).flatten()
            var_cols_lower = adata.var.columns.str.lower()
            if what.lower() in var_cols_lower:
                col_name = adata.var.columns[var_cols_lower.get_loc(what.lower())]
                column_data = adata.var[col_name]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype(str).values
                return column_data.values
        else:
            # Create a new HiVis object based on adata subsetting
            return self.subset(what, remove_empty_pixels=False)
            
    def subset(self, what=(slice(None), slice(None)), remove_empty_pixels=False, crop_agg=True):
        '''
        Create a new HiVis objects based on adata subsetting.
        
        Parameters:
            - what (tuple) - tuple of two elements. slicing instruction for adata. examples:
                - (slice(None), slice(None)): Select all spots and all genes.
                - ([0, 1, 2], slice(None)): Select the first three spots and all genes.
                - (slice(None), ['GeneA', 'GeneB']): Select all spots and specific genes.
                - (adata.obs['obs1'] == 'value', slice(None)): Select spots where 
                  the 'obs1' column in adata.obs is 'value', and all genes.
            - remove_empty_pixels (bool) - if True, the images will only contain pixels under bins
            - crop_agg (bool) - crop Agg objects. If False, plotting of aggregations might break.
            
        **Returns** new HiVis instance
        '''
        adata = self.adata[what].copy()
        image_fullres_crop, image_highres_crop, image_lowres_crop, xlim_pixels_fullres, ylim_pixels_fullres = self.__crop_images(adata, remove_empty_pixels)
        name = self.name + "_subset" if not self.name.endswith("_subset") else ""
        adata_shifted = self.__shift_adata(adata, xlim_pixels_fullres, ylim_pixels_fullres)
        new_obj = HiVis(adata_shifted, image_fullres_crop, image_highres_crop, 
                           image_lowres_crop, self.json, name, self.path_output,agg=None,plot_qc=False,
                           properties=self.properties.copy(),fluorescence=self.fluorescence.copy() if self.fluorescence else None)    
        # update the link in all aggregations to the new HiVis instance
        if self.agg: 
            for agg in self.agg:
                if crop_agg:
                    adata_agg = self.agg[agg].adata.copy()
                    idx_col = adata_agg.obs.index.name
                    adata_agg_shifted = adata_agg[adata_agg.obs.index.isin(adata_shifted.obs[idx_col]),adata_shifted.var_names]
                    adata_agg_shifted.var = adata_agg_shifted.var.loc[:,~adata_agg_shifted.var.columns.str.startswith(("cor_","exp_"))]
                    adata_agg_shifted = self.__shift_adata(adata_agg_shifted, xlim_pixels_fullres, ylim_pixels_fullres)
                else:
                    adata_agg_shifted = self.agg[agg].adata
                new_obj.add_agg(adata_agg_shifted.copy(),agg)
        return new_obj
   

    def __crop_images(self, adata, remove_empty_pixels=False):
        '''
        Helper function for get().
        Crops the images based on the spatial coordinates in a subsetted `adata` 
        and adjusts the adata accordingly (shifts x, y)
        remove_empty_pixels - whether to remove pixels that dont have spots on them.
        '''
        # Crop images
        def _crop_img(adata, img, col, row):
            '''crops one image by the x,y values in adata.obs, as specified by col, row'''
            pxl_col = adata.obs[col].values
            pxl_row = adata.obs[row].values
            xlim_pixels = [int(np.floor(pxl_col.min())), int(np.ceil(pxl_col.max()))]
            ylim_pixels = [int(np.floor(pxl_row.min())), int(np.ceil(pxl_row.max()))]
            # Ensure the limits are within the image boundaries
            xlim_pixels = [max(0, xlim_pixels[0]), min(img.shape[1], xlim_pixels[1])]
            ylim_pixels = [max(0, ylim_pixels[0]), min(img.shape[0], ylim_pixels[1])]
            if xlim_pixels[1] <= xlim_pixels[0] or ylim_pixels[1] <= ylim_pixels[0]:
                raise ValueError("Invalid crop dimensions.")
            img_crop = img[ylim_pixels[0]:ylim_pixels[1],xlim_pixels[0]:xlim_pixels[1],:].copy()
            
            if remove_empty_pixels:
                # remove pixels in images that don't have spots
                pxl_cols_shifted = pxl_col - xlim_pixels[0]
                pxl_rows_shifted = pxl_row - ylim_pixels[0]
                mask = np.zeros((img_crop.shape[0], img_crop.shape[1]), dtype=bool)
                for cx, cy in zip(pxl_cols_shifted, pxl_rows_shifted):
                    # Ensure we only mark valid coordinates within the cropped image
                    cx = int(cx)  
                    cy = int(cy)
                    if 0 <= cx < img_crop.shape[1] and 0 <= cy < img_crop.shape[0]:
                        mask[cy, cx] = True
                # Set non-adata remove pixels that are not covered by spots
                background_value = 0 if self.fluorescence else 255 # black for fluorescence, white for RGB
                img_crop[~mask] = background_value
            return img_crop, xlim_pixels, ylim_pixels
        
        image_fullres = self.image_fullres_orig if self.fluorescence else self.image_fullres
        image_fullres_crop, xlim_pixels_fullres, ylim_pixels_fullres = _crop_img(adata, image_fullres, "pxl_col_in_fullres", "pxl_row_in_fullres")
        image_highres_crop , _ , _ = _crop_img(adata, self.image_highres, "pxl_col_in_highres", "pxl_row_in_highres")
        image_lowres_crop , _ , _ = _crop_img(adata, self.image_lowres, "pxl_col_in_lowres", "pxl_row_in_lowres")

        return image_fullres_crop, image_highres_crop, image_lowres_crop,xlim_pixels_fullres, ylim_pixels_fullres
    
    
    def __shift_adata(self, adata, xlim_pixels_fullres, ylim_pixels_fullres):
        """
        Shifts the coordinates in an adata, based on xlim, ylim (in pixel space). \
        Also shifts the geometry WKT in micron space.
        """
        adata_shifted = adata.copy()
        drop_columns = ["pxl_col_in_lowres","pxl_row_in_lowres",
                        "pxl_col_in_highres","pxl_row_in_highres",
                        "um_x","um_y"]
        adata_shifted.obs.drop(columns=drop_columns, inplace=True, errors="ignore")
    
        # Shift the coordinates
        adata_shifted.obs["pxl_col_in_fullres"] -= xlim_pixels_fullres[0]
        adata_shifted.obs["pxl_row_in_fullres"] -= ylim_pixels_fullres[0]
    
        # Shift the geometry in micron space
        if "geometry" in adata_shifted.obs.columns:
            x_offset_microns = xlim_pixels_fullres[0] * self.json["microns_per_pixel"]
            y_offset_microns = ylim_pixels_fullres[0] * self.json["microns_per_pixel"]
    
            def _shift_wkt_geometry(geom_wkt):
                if isinstance(geom_wkt, str) and geom_wkt.strip():
                    geom = wkt.loads(geom_wkt)
                    geom = affinity.translate(geom, xoff=-x_offset_microns, yoff=-y_offset_microns)
                    return geom.wkt  # Store back as WKT
                return np.nan
    
            adata_shifted.obs["geometry"] = (
                adata_shifted.obs["geometry"]
                .fillna("")
                .apply(_shift_wkt_geometry)
            )
    
        return adata_shifted
        
    def __getitem__(self, what):
        '''Get a vector from data (a gene) or metadata (from obs or var). or subset the object.'''
        item = self.get(what, cropped=False)
        if item is None:
            raise KeyError(f"[{what}] isn't in data or metadatas")
        return item
    
    def remove_pixels(self, column: str, values: list, marging=1):
        '''
        Removes pixels in images, based on adata.obs[column].isin(values).
        
        Parameters:
            * marging (int) - how many pixels to extend the removed pixels.
            
        **Returns** new HiVis instance.
        '''
        
        # Identify which pixels to remove based on the given condition
        obs_values = self.adata.obs[column]
        remove_mask = obs_values.isin([v for v in values if not pd.isna(v)])
        if any(pd.isna(v) for v in values): # Handle NaNs
            remove_mask |= obs_values.isna()
    
        # Determine the background color: Black (0) if fluorescence images, white (255) otherwise
        if self.fluorescence:
            img_fullres_new = self.image_fullres_orig.copy() if self.image_fullres_orig is not None else None
            background_value = 0
        else:
            img_fullres_new = self.image_fullres.copy() if self.image_fullres is not None else None
            background_value = 255
        img_highres_new = self.image_highres.copy() if self.image_highres is not None else None
        img_lowres_new = self.image_lowres.copy() if self.image_lowres is not None else None
    
        # Extract spot diameter and compute corresponding sizes for each resolution
        spot_diameter_fullres = self.json['spot_diameter_fullres']
        # For indexing, we need integer sizes
        
        from math import ceil
        spot_size_fullres = int(ceil(spot_diameter_fullres))
        spot_size_hires = int(ceil(spot_diameter_fullres * self.json['tissue_hires_scalef']))
        spot_size_lowres = int(ceil(spot_diameter_fullres * self.json['tissue_lowres_scalef']))
        # Ensure sizes are at least 1
        spot_size_fullres = max(spot_size_fullres, 1)
        spot_size_hires = max(spot_size_hires, 1)
        spot_size_lowres = max(spot_size_lowres, 1)
        
    
        # The image info tuples as before
        img_info = [
            (img_fullres_new, "pxl_col_in_fullres", "pxl_row_in_fullres", spot_size_fullres),
            (img_highres_new, "pxl_col_in_highres", "pxl_row_in_highres", spot_size_hires),
            (img_lowres_new, "pxl_col_in_lowres", "pxl_row_in_lowres", spot_size_lowres)
        ]
    
        images = []
        for i, (img_new, col_name, row_name, spot_size) in enumerate(img_info):
            if img_new is not None:
                pxl_cols = self.adata.obs[col_name].values.astype(int)
                pxl_rows = self.adata.obs[row_name].values.astype(int)
                half_spot = spot_size // 2 + marging # +1 is marging
    
                # Instead of removing one pixel, remove a square region
                for idx, to_remove in enumerate(remove_mask):
                    if to_remove:
                        r = pxl_rows[idx]
                        c = pxl_cols[idx]
    
                        # Compute the boundaries of the square, clamped within image bounds
                        top = max(r - half_spot, 0)
                        bottom = min(r + half_spot + 1, img_new.shape[0])  # +1 to include the boundary
                        left = max(c - half_spot, 0)
                        right = min(c + half_spot + 1, img_new.shape[1])
    
                        # Set the entire block to background_value
                        img_new[top:bottom, left:right, :] = background_value
            images.append(img_new)
        
        # Create a new object with the modified images
        name = self.name + "_edited" if not self.name.endswith("_edited") else self.name
        new_obj = HiVis(self.adata.copy(),images[0],images[1],
                           images[2],self.json,name,self.path_output,
                           properties=self.properties.copy(),
                           fluorescence=self.fluorescence.copy() if self.fluorescence else None)        
        return new_obj

        
    def __str__(self):
        s = f"# {self.name} #\n"
        if hasattr(self, "organism"): s += f"\tOrganism: {self.organism}\n"
        if hasattr(self, "organ"): s += f"\tOrgan: {self.organ}\n"
        if hasattr(self, "sample_id"): s += f"\tID: {self.sample_id}\n"
        s += f"\tSize: {self.adata.shape[0]} x {self.adata.shape[1]}\n"
        s += '\nobs: '
        s += ', '.join(list(self.adata.obs.columns))
        s += '\n\nvar: '
        s += ', '.join(list(self.adata.var.columns))
        if self.agg:
            s += '\n\nAggregations:\n'
            for agg in self.agg:
                s += f"[{agg}]\tshape: {self.agg[agg].adata.shape[0]} x {self.agg[agg].adata.shape[1]}\n"
        return s
    
    def __repr__(self):
        # s = f"HiVis[{self.name}]"
        s = self.__str__()
        return s
    
    def __delitem__(self, key):
        '''deletes metadata'''
        if isinstance(key, str):
            if key in self.adata.obs:
                del self.adata.obs[key]
            elif key in self.adata.var.columns:
                del self.adata.var[key]
            else:
                raise KeyError(f"'{key}' not found in adata.obs")
            self.plot._init_img()
        else:
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
    
    def head(self, n=5):
        '''**Returns** HiVis.adata.obs.head(n), where n is number of rows'''
        return self.adata.obs.head(n)
    
    @property
    def shape(self):
        '''**Returns** HiVis.adata.shape'''
        return self.adata.shape
    
    @property
    def columns(self):
        '''**Returns** HiVis.adata.obs.columns'''
        return self.adata.obs.columns.copy()
    
    def rename(self, new_name: str, new_out_path=False, full=False):
        '''
        Renames the object and changes the path_output.
        If full is False, the name will be added to the current (previous) name
        '''
        if full:
            self.name = new_name
        else:
            self.name = self.name.replace("_subset","")
            self.name = f"{self.name}_{new_name}"
        if new_out_path:
            self.path_output = self.path_output + f"/{new_name}"
        
    
    def update(self, agg=False):
        '''Updates the methods in the instance. Should be used after modifying the source code in the class'''
        HiVis_utils.update_instance_methods(self)
        HiVis_utils.update_instance_methods(self.plot)
        self.plot._init_img()
        if agg and self.agg:
            for agg in self.agg:
                self.agg[agg].update()
        else:
            _ = gc.collect()

    def copy(self, new_name=None, new_out_path=False, full=False):
        '''
        Creates a deep copy of the instance.
        if new_name is specified, renames the object and changes the path_output.
        If full is False, the name will be added to the current (previous) name.
        
        **Returns** new HiVis instance
        '''
        new = deepcopy(self)
        if new_name:
            new.rename(new_name, new_out_path=new_out_path, full=full)
        return new
    
    def save(self, path=None):
        '''
        Saves the instance in pickle format.
        If no path specified, will save in the path_output as the name of the instance.
        
        **Returns** the path of the file (str)
        '''
        print(f"SAVING [{self.name}]")
        if not path:
            path = f"{self.path_output}/{self.name}.pkl"
        else:
            if not path.endswith(".pkl"):
                path += ".pkl"
        with open(path, "wb") as f:
            dill.dump(self, f)            
        return path

