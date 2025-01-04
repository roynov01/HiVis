# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:28:30 2024

@author: royno
"""
# General libraries
# import warnings
import os
import dill
from tqdm import tqdm
from copy import deepcopy
# Data libraries
import json
import numpy as np
import pandas as pd
import geopandas as gpd
# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from PIL import Image

import ViziumHD_utils
import ViziumHD_sc_class
import ViziumHD_plot

Image.MAX_IMAGE_PIXELS = 1063425001 # Enable large images loading
FULLRES_THRESH = 1000 # in microns, below which, a full-res image will be plotted
HIGHRES_THRESH = 3000 # in microns, below which, a high-res image will be plotted
DEFAULT_COLORS = ["white","purple","blue","yellow","red"]

# TODO add
# CELLPOSE


def load(filename, directory=''):
    '''
    loads an instance from a pickle format
    parameters:
        * filename - full path of pkl file, or just the filename if directory is specified
    '''
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"
    if directory:
        filename = f"{directory}/{filename}"
    ViziumHD_utils.validate_exists(filename)
    with open(filename, "rb") as f:
        instance = dill.load(f)
    return instance

def new(path_image_fullres:str, path_input_data:str, path_output:str,
             name:str, properties: dict = None, on_tissue_only=True,min_reads_in_spot=1,
             min_reads_gene=10, fluorescence=False):
    '''
    - Loads images (fullres, highres, lowres)
    - Loads data and metadata
    - croppes the images based on the data
    - initializes the connection from the data and metadata to the images coordinates
    - adds basic QC to the metadata (nUMI, mitochondrial %)
    parameters:
        * path_input_fullres_image - path for the fullres image
        * path_input_data - folder with outs of the Visium. typically square_002um
                            (with h5 files and with folders filtered_feature_bc_matrix, spatial)
        * path_output - path where to save plots and files
        * name - name of the instance
        * properties - dict of properties, such as organism, organ, sample_id
        * on_tissue_only - remove spots that are not classified as "on tissue"?
        * min_reads_in_spot - filter out spots with less than X UMIs
        * min_reads_gene - filter out gene that is present in less than X spots
        * fluorescence - either False for H&E, or a dict of channel names and colors
    '''
    # Validate paths of metadata and images
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    path_image_highres = path_input_data + "/spatial/tissue_hires_image.png"
    path_image_lowres = path_input_data + "/spatial/tissue_lowres_image.png"
    json_path = path_input_data + "/spatial/scalefactors_json.json"
    metadata_path = path_input_data + "/spatial/tissue_positions.parquet"
    ViziumHD_utils.validate_exists([path_image_fullres,path_image_highres,path_image_lowres,json_path,metadata_path])
    
    # Load images
    image_fullres, image_highres, image_lowres = ViziumHD_utils.load_images(path_image_fullres, path_image_highres, path_image_lowres)
    
    # Load scalefactor_json
    with open(json_path) as file:
        scalefactor_json = json.load(file)
    
   # Load data + metadata
    adata = ViziumHD_utils._import_data(metadata_path, path_input_data, path_image_fullres, on_tissue_only)
    
    # Crop images and initiates micron to pixel conversions for plotting
    adata, image_fullres, image_highres, image_lowres = ViziumHD_utils._crop_images_permenent(
        adata, image_fullres, image_highres, image_lowres, scalefactor_json)
    
    # Save cropped images
    path_image_fullres_cropped = path_image_fullres.replace("." + path_image_fullres.split(".")[-1], "_cropped.tif")
    path_image_highres_cropped = path_image_highres.replace("." + path_image_highres.split(".")[-1], "_cropped.tif")
    path_image_lowres_cropped = path_image_lowres.replace("." + path_image_lowres.split(".")[-1], "_cropped.tif")
    ViziumHD_utils._export_images(path_image_fullres_cropped, path_image_highres_cropped, 
                                  path_image_lowres_cropped,image_fullres,
                                  image_highres, image_lowres)
    
    if fluorescence:
        ViziumHD_utils._measure_fluorescence(adata, image_fullres, list(fluorescence.keys()), scalefactor_json["spot_diameter_fullres"])

    # Add QC (nUMI, mito %) and unit transformation
    mito_name_prefix = "MT-" if properties.get("organism") == "human" else "mt-"
    ViziumHD_utils._edit_adata(adata, scalefactor_json, mito_name_prefix)

    # Filter low quality spots and lowly expressed genes
    adata = adata[adata.obs["nUMI"] >= min_reads_in_spot, adata.var["nUMI_gene"] >= min_reads_gene].copy()

    return ViziumHD(adata, image_fullres, image_highres, image_lowres, scalefactor_json, 
                    name, path_output, properties, SC=None, fluorescence=fluorescence)


class ViziumHD:
    def __init__(self, adata, image_fullres, image_highres, image_lowres, scalefactor_json, 
                 name, path_output, properties=None, SC=None, fluorescence=False):
        self.SC = SC
        self.name, self.path_output = name, path_output 
        self.properties = properties if properties else {}
        self.organism = self.properties.get("organism")
        if isinstance(image_fullres, str): # paths of images, not the images themselves
            image_fullres, image_highres, image_lowres = ViziumHD_utils.load_images(image_fullres, image_highres, image_lowres)
        
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
        self.xlim_max = (adata.obs['um_x'].min(), adata.obs['um_x'].max())
        self.ylim_max = (adata.obs['um_y'].min(), adata.obs['um_y'].max())
        
        self.plot = ViziumHD_plot.PlotVizium(self)
        if fluorescence:
            self.image_fullres_orig = self.image_fullres.copy()
            self.recolor(fluorescence)
        else:
            self.__init_img()

        self.qc(save=True)
    
    def recolor(self, fluorescence=None, normalization_method="percentile"):
        '''
        Recolors a flurescence image
        parameters:
            * fluorescence is either list of colors or dict {channel: color...}
            * normalization_method - {"percentile", "histogram","clahe","sqrt" or None for minmax}
        '''
        if not self.fluorescence:
            raise ValueError("recolor() works for fluorescence visium only")
        if not fluorescence:
            fluorescence = self.fluorescence
            if not normalization_method:
                print(f'Choose colors for flurescence: {self.fluorescence}\nNormalization methods:\n"percentile", "histogram","clahe","sqrt" or None for minmax')
                return
        channels = list(self.fluorescence.keys())    
        if isinstance(fluorescence, list):
            if len(fluorescence) != len(channels):
                raise ValueError(f"Flurescence should include all channels: {channels}")
            self.fluorescence = {channels[i]:fluorescence[i] for i in range(len(channels))}
        elif isinstance(fluorescence, dict):
            if list(fluorescence.keys()) != channels:
                raise ValueError(f"Flurescence should include all channels: {channels}")
            self.fluorescence = fluorescence
        self.image_fullres = ViziumHD_utils.fluorescence_to_RGB(self.image_fullres_orig, 
                                                                self.fluorescence.values(), 
                                                                normalization_method)
        self.__init_img()


    def qc(self, save=False,figsize=(10, 10)):
        '''plots basic QC (nUMI, mitochondrial %)'''
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2,nrows=2, figsize=figsize)
        ax0 = self.plot.spatial(title=self.name, ax=ax0)
        ax1 = self.plot.hist("mito_percent_log10", title="Mitochondrial content per spot", xlab="log10(Mito %)",ax=ax1)
        ax2 = self.plot.hist("nUMI_log10", title="Number of UMIs per spot", xlab="log10(UMIs)",ax=ax2)
        ax3 = self.plot.hist("nUMI_gene_log10", title="Number of UMIs per gene", xlab="log10(UMIs)",ax=ax3)
        plt.tight_layout()
        if save:
            self.plot.save(figname="QC", fig=fig)
    
    def __init_img(self):
        '''resets the cropped image and updates the cropped adata'''
        self.image_cropped = None
        self.plot.ax_current = None # stores the last plot that was made
        # self.xlim_cur, self.ylim_cur = None, None
        self.pixel_x, self.pixel_y = None, None 
        self.adata_cropped = self.adata
        self.crop() # creates self.adata_cropped & self.image_cropped
        
    
    def add_mask(self, mask_path:str, name:str, plot=True, cmap="Paired"):
        '''
        assigns each spot a value based on mask (image).
        parameters:
            * mask_path - path to mask image
            * name - name of the mask (that will be called in the metadata)
            * plot - plot the mask?
            * cmap - colormap for plotting
        '''
        ViziumHD_utils.validate_exists(mask_path)
        
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
        self.__init_img()
        print(f"\nTo rename the values in the metadata, call the [update_meta] method with [{name}] and dictionary with current_name:new_name")
        return mask_array
    

    def add_annotations(self, path:str, name:str, measurements=True):
        '''
        Adds annotations made in Qupath (geojson)
        parameters:
            * path - path to geojson file
            * name -  name of the annotation (that will be called in the metadata)
            * measurements - include measurements columns? 
        '''
        ViziumHD_utils.validate_exists(path)
        annotations = gpd.read_file(path)
        if "classification" in annotations.columns:
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
            cols = [name,f"{name}_id"]
        for col in cols:
            if col in self.adata.obs.columns:
                del self.adata.obs[col]
        obs = gpd.GeoDataFrame(self.adata.obs, 
              geometry=gpd.points_from_xy(self.adata.obs["pxl_col_in_fullres"],
                                          self.adata.obs["pxl_row_in_fullres"]))        
        
        merged_obs = gpd.sjoin(obs,annotations,how="left",predicate="within")
        merged_obs = merged_obs[~merged_obs.index.duplicated(keep="first")]
        
        self.adata.obs = self.adata.obs.join(pd.DataFrame(merged_obs[cols]),how="left")
        self.__init_img()
    
        
    def dge(self, column, group1, group2=None, method="wilcox", two_sided=False,
            umi_thresh=0, inplace=False):
        '''
        Runs differential gene expression analysis between two groups.
        Values will be saved in self.var: expression_mean, log2fc, pval
        parameters:
            * column - which column in obs has the groups classification
            * group1 - specific value in the "column"
            * group2 - specific value in the "column". 
                       if None,will run agains all other values, and will be called "rest"
            * method - either "wilcox" or "t_test"
            * two_sided - if one sided, will give the pval for each group, 
                          and the minimal of both groups (which will also be FDR adjusted)
            * umi_thresh - use only spots with more UMIs than this number
            * expression - function F {mean, mean, max} F(mean(group1),mean(group2))
            * inplace - modify the adata.var with log2fc, pval and expression columns?
        '''
        alternative = "two-sided" if two_sided else "greater"
        df = ViziumHD_utils.dge(self.adata, column, group1, group2, umi_thresh,
                     method=method, alternative=alternative, inplace=False)
        df = df[[f"pval_{column}",f"log2fc_{column}",group1,group2]]
        df.rename(columns={f"log2fc_{column}":"log2fc"},inplace=True)
        if not two_sided:
            df[f"pval_{group1}"] = 1 - df[f"pval_{column}"]
            df[f"pval_{group2}"] = df[f"pval_{column}"]
            df["pval"] = df[[f"pval_{group1}",f"pval_{group2}"]].min(axis=1)
        else:
            df["pval"] = df[f"pval_{column}"]
        del df[f"pval_{column}"]
        df["qval"] = ViziumHD_utils.p_adjust(df["pval"])
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
    
    def add_meta(self, name:str, values, type_="obs"):
        '''
        adds a vector to metadata (obs or var)
        parameters:
            * name - name of metadata
            * values (array like) - values to add
            * type_ - either "obs" or "var"
        '''
        if type_ == "obs":
            if name in self.adata.obs.columns:
                raise ValueError(f"[{name}] allready present in adata.obs")
            self.adata.obs[name] = values
        elif type_ == "var":
            if name in self.adata.var.columns:
                raise ValueError(f"[{name}] allready present in adata.var")
            self.adata.var[name] = values
        self.__init_img()
    
    def update_meta(self, name:str, values:dict, type_="obs"):
        '''
        updates values in metadata (obs or var)
        parameters:
            * name - name of metadata
            * values - dict, {old_value:new_value}
            * type_ - either "obs" or "var"
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
        
        # Call to initialize image if needed
        self.__init_img()
        
            
    def export_h5(self, path=None, force=False):
        '''exports the adata. can also save the obs as parquet'''
        if path is None:
            path = self.path_output
        path = f"{path}/{self.name}_viziumHD.h5ad"
        if not os.path.exists(path) or force:
            print("[Writing h5]")
            self.adata.write(path)
        return self.adata
    
    def export_images(self, path=None, force=False):
        '''
        exports full,high and low resolution images
        '''
        if path is None:
            path = self.path_output
        if not os.path.exists(path):
            os.makedirs(path)
        path_image_fullres = f"{path}/{self.name}_fullres.tif"
        image_fullres = self.image_fullres_orig if self.fluorescence else self.image_fullres
        path_image_highres = f"{path}/{self.name}_highres.tif"
        path_image_lowres = f"{path}/{self.name}_lowres.tif"
        images = ViziumHD_utils._export_images(path_image_fullres, path_image_highres, 
                                      path_image_lowres, image_fullres, 
                                      self.image_highres, self.image_lowres, force=force)
        
        path_json = f"{path}/{self.name}_scalefactors_json.json"
        with open(path_json, 'w') as file:
            json.dump(self.json, file, indent=4)
        
        images.append(self.json)
        
        return images

    def aggregate_cells(self, input_df, columns=None, custom_agg=None, sep="\t"):
        if self.SC:
            if input('Single cell allready exists, if you want to aggregate again pres "y"') not in ("y","Y"):
                return
        self.SC = ViziumHD_sc_class.new_from_segmentation(self, input_df,columns,custom_agg,sep)

    def aggregate_annotations(self,group_col,columns,custom_agg):
        if self.SC:
            if input('Single cell allready exists, if you want to aggregate again pres "y"') not in ("y","Y"):
                return
        self.SC = ViziumHD_sc_class.new_from_annotations(self, group_col,columns,custom_agg)
    
    def crop(self, xlim=None, ylim=None, resolution=None):
        '''
        Crops the images and adata based on xlim and ylim in microns. 
        saves it in self.adata_cropped and self.image_cropped
        xlim, ylim: tuple of two values, in microns
        resolution - if None, will determine automatically, other wise, "full","high" or "low"
        '''
        microns_per_pixel = self.json['microns_per_pixel'] 
    
        # If xlim or ylim is None, set to the full range of the data
        if xlim is None:
            xlim = self.xlim_max
        if ylim is None:
            ylim = self.ylim_max
    
        # Decide which image to use based on lim_size:
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        lim_size = max(x_range, y_range)
        
        if resolution == "full":
            image = self.image_fullres
            scalef = 1  # No scaling needed for full-resolution
            pxl_col, pxl_row = 'pxl_col_in_fullres', 'pxl_row_in_fullres'
        elif resolution == "high":
            image = self.image_highres
            scalef = self.json['tissue_hires_scalef']
            pxl_col, pxl_row = 'pxl_col_in_highres', 'pxl_row_in_highres'
        elif resolution == "low":
            image = self.image_lowres
            scalef = self.json['tissue_lowres_scalef']
            pxl_col, pxl_row = 'pxl_col_in_lowres', 'pxl_row_in_lowres'       
        elif lim_size <= FULLRES_THRESH: # Use full-resolution image
            image = self.image_fullres
            scalef = 1  # No scaling needed for full-resolution
            pxl_col, pxl_row = 'pxl_col_in_fullres', 'pxl_row_in_fullres'
            print("Full-res image selected")
        elif lim_size <= HIGHRES_THRESH: # Use high-resolution image
            image = self.image_highres
            scalef = self.json['tissue_hires_scalef']  
            pxl_col, pxl_row = 'pxl_col_in_highres', 'pxl_row_in_highres'
            print("High-res image selected")
        else: # Use low-resolution image
            image = self.image_lowres
            scalef = self.json['tissue_lowres_scalef']  
            pxl_col, pxl_row = 'pxl_col_in_lowres', 'pxl_row_in_lowres'
            print("Low-res image selected")
    
        adjusted_microns_per_pixel = microns_per_pixel / scalef
        # refresh the adata_cropped
        
        xlim_pxl = [int(lim/ adjusted_microns_per_pixel) for lim in xlim]
        ylim_pxl = [int(lim/ adjusted_microns_per_pixel) for lim in ylim]
                
        # Ensure indices are within the bounds of the image dimensions
        xlim_pxl[0], ylim_pxl[0] = max(xlim_pxl[0], 0), max(ylim_pxl[0], 0)
        xlim_pxl[1], ylim_pxl[1] = min(xlim_pxl[1], image.shape[1]), min(ylim_pxl[1], image.shape[0])
        if xlim_pxl[0] >= xlim_pxl[1] or ylim_pxl[0] >= ylim_pxl[1]:
            raise ValueError("Calculated pixel indices are invalid.")
        
        # Crop the selected image
        self.image_cropped = image[ylim_pxl[0]:ylim_pxl[1], xlim_pxl[0]:xlim_pxl[1]]
    
        # Create a mask to filter the adata based on xlim and ylim
        x_mask = (self.adata.obs['um_x'] >= xlim[0]) & (self.adata.obs['um_x'] <= xlim[1])
        y_mask = (self.adata.obs['um_y'] >= ylim[0]) & (self.adata.obs['um_y'] <= ylim[1])
        mask = x_mask & y_mask
    
        # Crop the adata
        self.adata_cropped = self.adata[mask]
    
        # Adjust adata coordinates relative to the cropped image
        self.pixel_x = self.adata_cropped.obs[pxl_col] - xlim_pxl[0]
        self.pixel_y = self.adata_cropped.obs[pxl_row] - ylim_pxl[0]
        # self.xlim_cur, self.ylim_cur = xlim, ylim
    
        return xlim, ylim, adjusted_microns_per_pixel 
    
    def get(self, what, cropped=False):
        '''
        get a vector from data (a gene) or metadata (from obs or var). or subset the object.
        parameters:
            * what - if string, will get data or metadata. 
                     else, will return a new ViziumHD object that is spliced.
                     the splicing is passed to the self.adata
            * cropped - get the data from the adata_cropped after crop() or plotting methods?
        '''
        adata = self.adata_cropped if cropped else self.adata
        if isinstance(what, str): # easy acess to data or metadata arrays
            if what in adata.obs.columns: # Metadata
                return adata.obs[what].values
            if what in adata.var.index: # A gene
                return np.array(adata[:, what].X.todense().ravel()).flatten()
            if what in adata.var.columns: # Gene metadata
                return adata.var[what].values
            obs_cols_lower = adata.obs.columns.str.lower()
            if what.lower() in obs_cols_lower:
                col_name = adata.obs.columns[obs_cols_lower.get_loc(what.lower())]
                return adata.obs[col_name].values
            if self.organism == "mouse" and (what.lower().capitalize() in adata.var.index):
                return np.array(adata[:, what.lower().capitalize()].X.todense()).flatten()
            if self.organism == "human" and (what.upper() in adata.var.index):
                return np.array(adata[:, what.upper()].X.todense()).flatten()
            var_cols_lower = adata.var.columns.str.lower()
            if what.lower() in var_cols_lower:
                col_name = adata.var.columns[var_cols_lower.get_loc(what.lower())]
                return adata.var[col_name].values
        else:
            # Create a new ViziumHD objects based on adata subsetting
            return self.subset(what, remove_empty_pixels=False)
            
    def subset(self, what=(slice(None), slice(None)), remove_empty_pixels=False):
        '''
        Create a new ViziumHD objects based on adata subsetting.
        parameters:
            * remove_empty_pixels - if True, the images will only contain pixels under visium spots
            * what - tuple of two elements. slicing instruction for adata. examples:
                - (slice(None), slice(None)): Select all spots and all genes.
                - ([0, 1, 2], slice(None)): Select the first three spots and all genes.
                - (slice(None), ['GeneA', 'GeneB']): Select all spots and specific genes.
                - (adata.obs['obs1'] == 'value', slice(None)): Select spots where 
                  the 'obs1' column in adata.obs is 'value', and all genes.
        '''
        adata = self.adata[what].copy()
        adata_shifted, image_fullres_crop, image_highres_crop, image_lowres_crop = self.__crop_images(adata, remove_empty_pixels)
        name = self.name + "_subset" if not self.name.endswith("_subset") else ""
        single_cell = None
        if self.SC is not None: 
            single_cell = self.SC.subset(what)    
        new_obj = ViziumHD(adata_shifted, image_fullres_crop, image_highres_crop, 
                           image_lowres_crop, self.json, name, self.path_output,SC=single_cell,
                           properties=self.properties.copy(),fluorescence=self.fluorescence.copy() if self.fluorescence else None)
        if single_cell: # update the link in SC to the new ViziumHD instance
            new_obj.SC = ViziumHD_sc_class.SingleCell(new_obj, new_obj.SC.adata)
        return new_obj
   
    def __crop_images(self, adata, remove_empty_pixels):
        '''
        Helper function for get().
        Crops the images based on the spatial coordinates in a subsetted `adata` 
        and adjusts the adata accordingly (shifts x, y)
        remove_empty_pixels
        '''
        # Crop images
        def _crop_img(adata, img, col, row):
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

        # Shift adata
        adata_shifted = adata.copy()
        drop_columns = ["pxl_col_in_lowres","pxl_row_in_lowres","pxl_col_in_highres",
                        "pxl_row_in_highres","um_x","um_y"]
        adata_shifted.obs.drop(columns=drop_columns, inplace=True)
        adata_shifted.obs["pxl_col_in_fullres"] -= xlim_pixels_fullres[0]
        adata_shifted.obs["pxl_row_in_fullres"] -= ylim_pixels_fullres[0]
        return adata_shifted, image_fullres_crop, image_highres_crop, image_lowres_crop
        
    def __getitem__(self, what):
        '''get a vector from data (a gene) or metadata (from obs or var). or subset the object.'''
        item = self.get(what, cropped=False)
        if item is None:
            raise KeyError(f"[{what}] isn't in data or metadatas")
        return item
    
    def remove_pixels(self, column: str, values: list, marging=1):
        '''
        removes pixels in images, based on adata.obs[column].isin(values).
        returns new ViziumHD object.
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
        print(f"{spot_diameter_fullres=},{self.json['tissue_hires_scalef']=}")
        print(f"{spot_size_fullres=},{spot_size_hires=},{spot_size_lowres=}")
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
        new_obj = ViziumHD(self.adata.copy(),images[0],images[1],
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
        if self.SC is not None:
            s += f"\tSingle cells shape: {self.SC.adata.shape[0]} x {self.SC.adata.shape[1]}"
        return s
    
    def __repr__(self):
        s = f"ViziumHD[{self.name}]"
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
            self.__init_img()
        else:
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
    
    def head(self, n=5):
        return self.adata.obs.head(n)
    
    @property
    def shape(self):
        return self.adata.shape
    
    @property
    def columns(self):
        return self.adata.obs.columns.copy()
    
    def update(self):
        '''updates the methods in the instance'''
        ViziumHD_utils.update_instance_methods(self)
        ViziumHD_utils.update_instance_methods(self.plot)
        self.__init_img()
        # update also the SC
        # if self.SC is not None:
            # ViziumHD_utils.update_instance_methods(self.SC)
            # ViziumHD_utils.update_instance_methods(self.SC.plot)
    
    def copy(self):
        return deepcopy(self)
    
    def save(self, path=None):
        '''saves the instance in pickle format'''
        print(f"SAVING [{self.name}]")
        if not path:
            path = f"{self.path_output}/{self.name}.pkl"
        else:
            if not path.endswith(".pkl"):
                path += ".pkl"
        with open(path, "wb") as f:
            dill.dump(self, f)
        return path

