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

# TODO add
# CELLPOSE
# Add general function plot_MA(df, cond1, cond2, pval_col="pval", exp_thresh=0, qval_thresh=0.05).
#   if not pval, then just plot all genes

def load(filename, directory=''):
    '''loads an instance from a pickle format'''
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
             min_reads_gene=10):
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
    '''
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    path_image_highres = path_input_data + "/spatial/tissue_hires_image.png"
    path_image_lowres = path_input_data + "/spatial/tissue_lowres_image.png"
    json_path = path_input_data + "/spatial/scalefactors_json.json"
    metadata_path = path_input_data + "/spatial/tissue_positions.parquet"
    
    # Validate paths of metadata and images
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
    ViziumHD_utils._export_images(path_image_fullres, path_image_highres, path_image_lowres,
                        image_fullres, image_highres, image_lowres)
    
    # Add QC (nUMI, mito %) and unit transformation
    mito_name_prefix = "MT-" if properties.get("organism") == "human" else "mt-"
    ViziumHD_utils._edit_adata(adata, scalefactor_json, mito_name_prefix)

    # Filter low quality spots and lowly expressed genes
    adata = adata[adata.obs["nUMI"] >= min_reads_in_spot, adata.var["nUMI_gene"] >= min_reads_gene].copy()

    return ViziumHD(adata, image_fullres, image_highres, image_lowres, scalefactor_json, name, path_output, properties, SC=None)


class ViziumHD:
    def __init__(self, adata, image_fullres, image_highres, image_lowres, scalefactor_json, name, path_output, properties=None, SC=None):
        self.SC = SC
        self.name, self.path_output = name, path_output 
        self.properties = properties if properties else {}
        self.organism = self.properties.get("organism")
        self.image_fullres, self.image_highres, self.image_lowres = image_fullres, image_highres, image_lowres
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
        self.__init_img()
        self.qc(save=True)

        
    def qc(self, save=False,figsize=(10, 10)):
        '''plots basic QC (nUMI, mitochondrial %)'''
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2,nrows=2, figsize=figsize)
        ax0 = self.plot.spatial(title=self.name, ax=ax0)
        ax1 = self.plot.hist("mito_percent_log10", title="Mitochondrial content per spot", xlab="log10(Mito %)",ax=ax1)
        ax2 = self.plot.hist("nUMI_log10", title="Number of UMIs per spot", xlab="log10(UMIs)",ax=ax2)
        ax3 = self.plot.hist("nUMI_gene_log10", title="Number of UMIs per gene", xlab="log10(UMIs)",ax=ax3)
        plt.tight_layout()
        if save:
            self.plot.save(filename="QC", fig=fig)
    
    def __init_img(self):
        '''resets the cropped image and updates the cropped adata'''
        self.image_cropped = None
        self.plot.ax_current = None # stores the last plot that was made
        self.xlim_cur, self.pixel_x, self.ylim_cur, self.pixel_y = None, None, None, None
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
    

    def add_annotations(self, path:str, name:str):
        '''
        Adds annotations made in Qupath (geojson)
        parameters:
            * path - path to geojson file
            * name -  name of the annotation (that will be called in the metadata)
        '''
        ViziumHD_utils.validate_exists(path)
        annotations = gpd.read_file(path)
        if "classification" in annotations.columns:
            annotations[name] = [x["name"] for x in annotations["classification"]]
        else:
            annotations[name] = annotations.index
        del annotations["id"]
        del annotations["objectType"]
        if name in self.adata.obs.columns:
            del self.adata.obs[name]
        obs = gpd.GeoDataFrame(self.adata.obs, 
              geometry=gpd.points_from_xy(self.adata.obs["pxl_col_in_fullres"],
                                          self.adata.obs["pxl_row_in_fullres"]))        
        
        merged_obs = gpd.sjoin(obs,annotations,how="left",predicate="within")
        merged_obs = merged_obs[~merged_obs.index.duplicated(keep="first")]
        
        self.adata.obs = self.adata.obs.join(pd.DataFrame(merged_obs[[name]]),how="left")
        self.__init_img()
    
        
    def dge(self, column, group1, group2=None, umi_thresh=1,
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
        df = ViziumHD_utils.dge(self.adata,column, group1, group2,umi_thresh,
                     method=method,alternative=alternative,inplace=inplace)
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
                raise ValueError(f"No metadata called [{name}]")
            if pd.api.types.is_categorical_dtype(self.adata.obs[name]):
                self.adata.obs[name] = self.adata.obs[name].cat.rename_categories(values)
            else:
                self.adata.obs[name] = self.adata.obs[name].replace(values)
        elif type_ == "var":
            if name not in self.adata.var.columns:
                raise ValueError(f"No metadata called [{name}]")   
            if pd.api.types.is_categorical_dtype(self.adata.var[name]):
                self.adata.var[name] = self.adata.var[name].cat.rename_categories(values)
            else:
                self.adata.var[name] = self.adata.var[name].replace(values)
        self.__init_img()
        

    
    

        
            
    def export_h5(self, path=None):
        '''exports the adata. can also save the obs as parquet'''
        if not path:
            path = f"{self.path_output}/{self.name}_viziumHD.h5ad"
        self.adata.write(path)
        return path
    
    def export_images(self, fullres=True, highres=True, lowres=True):
        if fullres:
            ViziumHD_utils.export_image(self.image_fullres, self.path_output+"/fullres_image.tif")
        if highres:
            ViziumHD_utils.export_image(self.image_highres, self.path_output+"/highres_image.tif")
        if lowres:
            ViziumHD_utils.export_image(self.image_lowres, self.path_output+"/lowres_image.tif")

    def sc_create(self, category):
        if self.SC:
            if input('Single cell allready exists, if you want to aggregate again pres "y"') not in ("y","Y"):
                return
        params = []
        self.SC = ViziumHD_sc_class.SingleCell(self, params)

    def sc_transfer_meta(self, what:str):
        '''transfers metadata assignment from the single-cell to the spots'''
        pass
    
    
    def crop(self, xlim=None, ylim=None):
        '''
        Crops the images and adata based on xlim and ylim in microns. 
        saves it in self.adata_cropped and self.image_cropped
        xlim, ylim: tuple of two values, in microns
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
        if lim_size >= HIGHRES_THRESH: # Use low-resolution image
            image = self.image_lowres
            scalef = self.json['tissue_lowres_scalef']  
            pxl_col, pxl_row = 'pxl_col_in_lowres', 'pxl_row_in_lowres'
            print("Low-res image selected")
        elif lim_size <= FULLRES_THRESH: # Use full-resolution image
            image = self.image_fullres
            scalef = 1  # No scaling needed for full-resolution
            pxl_col, pxl_row = 'pxl_col_in_fullres', 'pxl_row_in_fullres'
            print("Full-res image selected")
        else: # Use high-resolution image
            image = self.image_highres
            scalef = self.json['tissue_hires_scalef']  
            pxl_col, pxl_row = 'pxl_col_in_highres', 'pxl_row_in_highres'
            print("High-res image selected")
    
        adjusted_microns_per_pixel = microns_per_pixel / scalef
        # refresh the adata_cropped
        
        if len(self.adata.obs.columns) == len(self.adata_cropped.obs.columns):
            if xlim == self.xlim_cur and ylim == self.ylim_cur: # Same values as last crop() call
                return xlim, ylim, adjusted_microns_per_pixel 
        
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
        self.xlim_cur, self.ylim_cur = xlim, ylim
    
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
            if what.lower() in adata.obs.columns.str.lower(): 
                return adata.obs[what.lower()].values
            if self.organism == "mouse" and (what.lower().capitalize() in adata.var.index):
                return np.array(adata[:, what.lower().capitalize()].X.todense().ravel()).flatten() 
            if self.organism == "human" and (what.upper() in adata.var.index):
                return np.array(adata[:, what.upper()].X.todense().ravel()).flatten() 
            if what.lower() in adata.var.columns.str.lower(): 
                return adata.var[what.lower()].values
        else:
            # Create a new ViziumHD objects based on adata subsetting
            adata = self.adata[what].copy()
            adata_shifted, image_fullres_crop, image_highres_crop, image_lowres_crop = self.__crop_images(adata)
            name = self.name + "_subset" if not self.name.endswith("_subset") else ""
            new_obj = ViziumHD(adata_shifted, image_fullres_crop, image_highres_crop, 
                               image_lowres_crop, self.json, name, self.path_output)
            return new_obj
   
    def __crop_images(self, adata):
        '''
        Helper function for get().
        Crops the images based on the spatial coordinates in a subsetted `adata` 
        and adjusts the adata accordingly (shifts x, y)'''
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
            return img_crop, xlim_pixels, ylim_pixels
        
        image_fullres_crop, xlim_pixels_fullres, ylim_pixels_fullres = _crop_img(adata, self.image_fullres, "pxl_col_in_fullres", "pxl_row_in_fullres")
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
    
    def update(self):
        '''updates the methods in the instance'''
        ViziumHD_utils.update_instance_methods(self)
        ViziumHD_utils.update_instance_methods(self.plot)
        self.__init_img()
        # update also the SC
        if self.SC is not None:
            ViziumHD_utils.update_instance_methods(self.SC)
            ViziumHD_utils.update_instance_methods(self.SC.plot)
    
    def copy(self):
        return deepcopy(self)
    
    def save(self, path=None):
        '''saves the instance in pickle format'''
        if not path:
            path = f"{self.path_output}/{self.name}.pkl"
        else:
            if not path.endswith(".pkl"):
                path += ".pkl"
        with open(path, "wb") as f:
            dill.dump(self, f)
        return path

