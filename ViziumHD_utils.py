# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:28:30 2024

@author: royno
"""
# general libraries
import importlib
# import types
import warnings
import sys
import os
import dill
from tqdm import tqdm
from copy import deepcopy
# data libraries

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import scanpy as sc
import math
# Image libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps
from PIL import Image
import tifffile

Image.MAX_IMAGE_PIXELS = 1063425001 # Enable large images
FULLRES_MAX_PLOT_SIZE = 2500 # in fullres pixels
POINTS_PER_INCH = 72
MAX_BARS = 30 # in barplot
FULLRES_THRESH = 800
HIGHRES_THRESH = 3000
PAD_CONSTANT = 0.3
# DEFAULT_COLOR = "lightgray"
DEFAULT_COLOR ='None'

class ViziumHD:
    def __init__(self, path_input_fullres_image, path_input_data, path_output, name, properties: dict = None, on_tissue_only=True):
        self.sc = None
        self.name, self.path_output, self.properties = name, path_output, properties if properties else {}
        properties = {} if properties is None else properties
        self.organism = properties.get("organism")
        self.organ = properties.get("organ")
        self.sample_id = properties.get("sample_id")
        
        if not os.path.exists(path_output):
            os.makedirs(path_output)
            
        # load images
        print("[Loading images]")
        image_fullres = tifffile.imread(path_input_fullres_image)
        rgb_dim = image_fullres.shape.index(3)
        if rgb_dim != 2:  # If the RGB dimension is not already last
            axes_order = list(range(image_fullres.ndim))  # Default axes order
            axes_order.append(axes_order.pop(rgb_dim))  # Move the RGB dim to the last position
            image_fullres = image_fullres.transpose(axes_order)
        self.image_fullres = image_fullres
        
        hires_image_path = path_input_data + "/spatial/tissue_hires_image.png"
        self.image_highres = plt.imread(hires_image_path)
        lowres_image_path = path_input_data + "/spatial/tissue_lowres_image.png"
        self.image_lowres = plt.imread(lowres_image_path)
        
        
        # load json
        json_path = path_input_data + "/spatial/scalefactors_json.json"
        with open(json_path) as file:
            self.json = json.load(file)
        
        # load metadata      
        print("[Loading metadata]")
        metadata_path = path_input_data + "/spatial/tissue_positions.csv"
        if not os.path.isfile(metadata_path):
            metadata = pd.read_parquet(metadata_path.replace(".csv",".parquet"))
            metadata.to_csv(metadata_path, index=False)
        else:
            metadata = pd.read_csv(metadata_path)
        del metadata["array_row"]
        del metadata["array_col"]
        

                
        # load data
        print("[Loading data]")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique")
            self.adata = sc.read_visium(path_input_data, source_image_path=path_input_fullres_image)
        self.adata.var_names_make_unique()
        
        if on_tissue_only: # filter spots that are classified to be under tissue
            metadata = metadata.loc[metadata['in_tissue'] == 1,]
            self.adata = self.adata[self.adata.obs['in_tissue'] == 1]
        del metadata["in_tissue"]    
        # merge data and metadata
        metadata = metadata[~metadata.index.duplicated(keep='first')]
        metadata.set_index('barcode', inplace=True)
        self.adata.obs = self.adata.obs.join(metadata, how='left')
        self.adata.obs["nUMI"] = np.array(self.adata.X.sum(axis=1).flatten())[0]
                
        self.__crop_img_permenent()
        self.__export_images(path_input_fullres_image, hires_image_path, lowres_image_path)
        
        self.adata.obs["pxl_col_in_lowres"] = self.adata.obs["pxl_col_in_fullres"] * self.json["tissue_lowres_scalef"]
        self.adata.obs["pxl_row_in_lowres"] = self.adata.obs["pxl_row_in_fullres"] * self.json["tissue_lowres_scalef"]

        self.adata.obs["pxl_col_in_highres"] = self.adata.obs["pxl_col_in_fullres"] * self.json["tissue_hires_scalef"]
        self.adata.obs["pxl_row_in_highres"] = self.adata.obs["pxl_row_in_fullres"] * self.json["tissue_hires_scalef"]

        self.adata.obs["um_x"] = self.adata.obs["pxl_col_in_fullres"] * self.json["microns_per_pixel"]
        self.adata.obs["um_y"] = self.adata.obs["pxl_row_in_fullres"] * self.json["microns_per_pixel"]
        
        self.xlim_max = (self.adata.obs['um_x'].min(), self.adata.obs['um_x'].max())
        self.ylim_max = (self.adata.obs['um_y'].min(), self.adata.obs['um_y'].max())
        self.__init_img()
        
        # Plot quality control - number of UMIs:
        self.hist("nUMI", title="Number of UMIs", save=True, xlab="Number of unique reads")
    
    def __init_img(self):
        self.image_cropped = None
        self.ax_current = None # stores the last plot that was made
        self.xlim_cur, self.pixel_x, self.ylim_cur, self.pixel_y = None, None, None, None
        self.adata_cropped = self.adata
        self.crop() # creates self.adata_cropped & self.image_cropped
        
    def __crop_img_permenent(self):
        pxl_col_in_fullres = self.adata.obs["pxl_col_in_fullres"].values
        pxl_row_in_fullres = self.adata.obs["pxl_row_in_fullres"].values
        
        xlim_pixels_fullres = [math.floor(pxl_col_in_fullres.min()), math.ceil(pxl_col_in_fullres.max())]
        ylim_pixels_fullres = [math.floor(pxl_row_in_fullres.min()), math.ceil(pxl_row_in_fullres.max())]
        # Ensure the limits are within the image boundaries
        xlim_pixels_fullres = [max(0, xlim_pixels_fullres[0]), min(self.image_fullres.shape[1], xlim_pixels_fullres[1])]
        ylim_pixels_fullres = [max(0, ylim_pixels_fullres[0]), min(self.image_fullres.shape[0], ylim_pixels_fullres[1])]
        # print(f"BEFORE:\n\t{self.image_fullres.shape}\n\t{self.image_highres.shape}\n\t{self.image_lowres.shape}\n" \
        #       f"\t{self.adata.obs['pxl_col_in_fullres'].min()},{self.adata.obs['pxl_col_in_fullres'].max()}\n\t{self.adata.obs['pxl_row_in_fullres'].min()},{self.adata.obs['pxl_row_in_fullres'].max()}")
        
        # Crop the full-resolution image
        self.image_fullres = self.image_fullres[ylim_pixels_fullres[0]:ylim_pixels_fullres[1],
                                               xlim_pixels_fullres[0]:xlim_pixels_fullres[1],:]
        
        # Adjust limits for high-resolution image and crop
        scaling_factor_hires = self.json["tissue_hires_scalef"]
        xlim_pixels_highres = [x*scaling_factor_hires for x in xlim_pixels_fullres]
        ylim_pixels_highres = [y*scaling_factor_hires for y in ylim_pixels_fullres]
        xlim_pixels_highres[0], xlim_pixels_highres[1] = math.floor(xlim_pixels_highres[0]), math.ceil(xlim_pixels_highres[1])
        ylim_pixels_highres[0], ylim_pixels_highres[1] = math.floor(ylim_pixels_highres[0]), math.ceil(ylim_pixels_highres[1])
        self.image_highres = self.image_highres[ylim_pixels_highres[0]:ylim_pixels_highres[1],
                                               xlim_pixels_highres[0]:xlim_pixels_highres[1],:]
    
        # Adjust limits for low-resolution image and crop
        scaling_factor_lowres = self.json["tissue_lowres_scalef"]
        xlim_pixels_lowres = [x*scaling_factor_lowres for x in xlim_pixels_fullres]
        ylim_pixels_lowres = [y*scaling_factor_lowres for y in ylim_pixels_fullres]
        xlim_pixels_lowres[0], xlim_pixels_lowres[1] = math.floor(xlim_pixels_lowres[0]), math.ceil(xlim_pixels_lowres[1])
        ylim_pixels_lowres[0], ylim_pixels_lowres[1] = math.floor(ylim_pixels_lowres[0]), math.ceil(ylim_pixels_lowres[1])
        self.image_lowres = self.image_lowres[ylim_pixels_lowres[0]:ylim_pixels_lowres[1],
                                             xlim_pixels_lowres[0]:xlim_pixels_lowres[1],:]
        
        # Shift the metadata to the new poisition
        self.adata.obs["pxl_col_in_fullres"] = self.adata.obs["pxl_col_in_fullres"] - xlim_pixels_fullres[0]
        self.adata.obs["pxl_row_in_fullres"] = self.adata.obs["pxl_row_in_fullres"] - ylim_pixels_fullres[0]
        # print(f"BEFORE:\n\t{self.image_fullres.shape}\n\t{self.image_highres.shape}\n\t{self.image_lowres.shape}\n" \
        #       f"\t{self.adata.obs['pxl_col_in_fullres'].min()},{self.adata.obs['pxl_col_in_fullres'].max()}\n\t{self.adata.obs['pxl_row_in_fullres'].min()},{self.adata.obs['pxl_row_in_fullres'].max()}")
    
    def __export_images(self, path_input_fullres_image, hires_image_path, lowres_image_path):
        print("[Saving cropped images]")
        images = [self.image_fullres, self.image_highres, self.image_lowres]
        paths = [path_input_fullres_image, hires_image_path, lowres_image_path]
        for im, path in zip(images, paths):
            fileformat = "." + path.split(".")[1]
            save_path = path.replace(fileformat, "_cropped.tif")
            if not os.path.exists(save_path):
                if im.max() <= 1:
                    im = (im * 255).astype(np.uint8)
                image = Image.fromarray(im)
                image.save(save_path, format='TIFF')
    
    def add_mask(self, mask_path, name, plot=True, cmap="Paired"):
        mask_array = self.__import_mask(mask_path)
        if plot:
            self.__plot_mask(mask_array, cmap=cmap)
        self.__assign_spots(mask_array, name)
        self.__init_img()
        print(f"\nTo rename the values in the metadata, call the [update_meta] method with [{name}] and dictionary with current_name:new_name")
        return mask_array
    
    def __plot_mask(self, mask_array, cmap):
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_array, cmap=cmap)
        num_colors = len(np.unique(mask_array[~np.isnan(mask_array)]))
        cmap = plt.cm.get_cmap(cmap, num_colors) 
        legend_elements = [Patch(facecolor=cmap(i), label=f'{i}') for i in range(num_colors)]
        plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.5))
        plt.show()
        
    def __import_mask(self, mask_path):
        print("[Importing mask]")
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        return mask_array
    
    def __assign_spots(self, mask_array, name):
        def get_mask_value(mask_array, x, y):
            if (0 <= x < mask_array.shape[1]) and (0 <= y < mask_array.shape[0]):
                return mask_array[y, x]
            else:
                return None
        def get_spot_identity(row):
            x = round(row['pxl_col_in_fullres'])
            y = round(row['pxl_row_in_fullres'])
            return get_mask_value(mask_array, x, y)
        
        tqdm.pandas(desc=f"Assigning spots identity [{name}]")
        
        self.adata.obs[name] = np.nan
        self.adata.obs[name] = self.adata.obs.progress_apply(
            get_spot_identity, axis=1)
        
    def add_annotations(self, path, name):
        '''Adds annotations made in Qupath (json), returns colormap that can be used as plot() cmap parameter'''
        annotations = gpd.read_file(path)
        if "classification" in annotations.columns:
            annotations[name] = [x["name"] for x in annotations["classification"]]
        else:
            annotations[name] = annotations.index
        del annotations["id"]
        del annotations["objectType"]

        obs = gpd.GeoDataFrame(self.adata.obs, 
              geometry=gpd.points_from_xy(self.adata.obs["pxl_col_in_fullres"],
                                          self.adata.obs["pxl_row_in_fullres"]))        
        
        merged_obs = gpd.sjoin(obs,annotations,how="left",predicate="within")
        merged_obs = merged_obs[~merged_obs.index.duplicated(keep="first")]
        self.adata.obs = self.adata.obs.join(pd.DataFrame(merged_obs[[name]]),how="left")
        self.__init_img()
        
    def add_col(self, name, values):
        if name in self.adata.obs.columns:
            print(f"[{name}] allready present in adata.obs")
            return
        self.adata.obs[name] = values
        self.__init_img()
    
    def update_meta(self, name, values):
        if name not in self.adata.obs.columns:
            raise ValueError(f"No metadata called [{name}]")
        self.adata.obs[name] = self.adata.obs[name].replace(values)
        self.__init_img()
     
    def save_fig(self, filename, ax=None, open_file=False, format='png', dpi=300):
        if ax is None:
            if self.current_ax is None:
                print(f"No ax present in {self.name}")
                return
            ax = self.current_ax
        fig = ax.get_figure()
        path = f"{self.path_output}/{filename}.png"
        fig.savefig(path, format='png', dpi=300, bbox_inches='tight')
        if open_file:
            os.startfile(path)

    def crop(self, xlim=None, ylim=None):
        '''Crops the image and adata based on xlim and ylim in microns.
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
    
    def __get_dot_size(self, adjusted_microns_per_pixel):
        bin_size_pixels = self.json['bin_size_um'] / adjusted_microns_per_pixel 
        dpi = plt.gcf().get_dpi()
        # dpi = mpl.rcParams['figure.dpi']
        points_per_pixels = POINTS_PER_INCH / dpi
        dot_size = bin_size_pixels * points_per_pixels 
        return dot_size
        
    
    def plot(self, what=None, image=True, title=None, cmap="viridis",
                  legend=True, alpha=1, figsize=(8, 8), save=False,
                  xlim=None, ylim=None, legend_title=None, axis_labels=True, pad=False):
        '''
        limits are in microns
        '''
        title = what if title is None else title
        if legend_title is None:
            legend_title = what.capitalize() if what and what==what.lower else None
            
        xlim, ylim, adjusted_microns_per_pixel = self.crop(xlim, ylim)
        size = self.__get_dot_size(adjusted_microns_per_pixel)
        if pad:
            size *= PAD_CONSTANT
        fig, ax = plt.subplots(figsize=figsize)

        if image: # Plot image
            # ax.imshow(self.image_cropped, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])
            ax.imshow(self.image_cropped)

        if what: 
            values = self[what]
            if np.issubdtype(values.dtype, np.number):  # Filter values that are 0
                mask = values > 0
            else:
                mask = [True for _ in values]   # No need for filtering
            values = values[mask]
            x = self.pixel_x[mask]
            y = self.pixel_y[mask]
            
            if np.issubdtype(values.dtype, np.number): 
                argsort_values = np.argsort(values)
                x, y, values = x.iloc[argsort_values], y.iloc[argsort_values], values[argsort_values]
            
            # Plot scatter:
            ax = plot_scatter(x, y, values, size=size,title=title,
                          figsize=figsize,alpha=alpha,cmap=cmap,ax=ax,
                          legend=legend,xlab=None,ylab=None, 
                          legend_title=legend_title)
            
        if axis_labels:
            ax.set_xlabel("Spatial 1 (µm)")
            ax.set_ylabel("Spatial 2 (µm)")
            
        height, width = self.image_cropped.shape[:2]  
        set_axis_ticks(ax, width, adjusted_microns_per_pixel, axis='x')
        set_axis_ticks(ax, height, adjusted_microns_per_pixel, axis='y')    
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)     
        
        # Save figure:
        self.current_ax = ax
        if save:
            self.save_fig(f"{what}_SPATIAL")
        return ax
    
    def hist(self, what, bins=20, xlim=None, title=None, ylab=None,xlab=None,
             save=False, figsize=(8,8), cmap=None, color="blue"):
        
        title = what if title is None else title
        self.crop() # resets adata_cropped to full image
        to_plot = pd.Series(self[what])
        ax = plot_hist(to_plot,bins=bins,xlim=xlim,title=title,figsize=figsize,
                       cmap=cmap,color=color,ylab=ylab,xlab=xlab)            
        self.current_ax = ax
        if save:
            self.save_fig(f"{what}_HIST")
        return ax
            
    def export_h5(self, save_parquet=True):
        path = f"{self.path_output}/{self.name}_viziumHD.h5ad"
        self.adata.write(path)
        if save_parquet:
            self.adata.obs.to_parquet(path.replace("_viziumHD.h5ad","_spots_metadata.parquet"),index=False)    
        return path

    def aggregate(self, category):
        if self.sc:
            if input('Single cell allready exists, if you want to aggregate again pres "y"') not in ("y","Y"):
                return

    def assign_clusters_from_sc(self):
        pass
    

    
    def __getitem__(self, what):
        '''
        Retrieves the values of the adata_cropped. to get full values, run self.crop() prior.
        '''
        if isinstance(what, str): # easy acess to data or metadata arrays
            if what in self.adata_cropped.obs.columns: # Metadata
                return self.adata_cropped.obs[what].values
            if what in self.adata_cropped.var.index: # A gene
                return np.array(self.adata_cropped[:, what].X.todense().ravel()).flatten() 
            if what.lower() in self.adata_cropped.obs.columns.str.lower(): 
                return self.adata_cropped.obs[what.lower()].values
            if self.organism == "mouse" and (what.lower().capitalize() in self.adata_cropped.var.index):
                return np.array(self.adata_cropped[:, what.lower().capitalize()].X.todense().ravel()).flatten() 
            if self.organism == "human" and (what.upper() in self.adata_cropped.var.index):
                return np.array(self.adata_cropped[:, what.upper()].X.todense().ravel()).flatten() 
            raise KeyError(f"[{what}] isn't in data or metadata")
        else:
            copy = self.copy()
            copy.adata = copy.adata[what]
            copy.var, copy.obs = copy.adata.var, copy.adata.obs 
            copy.__init_img()
            return copy
            

    def __str__(self):
        s = f"# {self.name} #\n\n"
        if hasattr(self, "organism"): s += f"\tOrganism: {self.organism}\n"
        if hasattr(self, "organ"): s += f"\tOrgan: {self.organ}\n"
        if hasattr(self, "sample_id"): s += f"\tID: {self.sample_id}\n"
        s += f"\tSize: {self.adata.shape[0]} x {self.adata.shape[1]}\n"
        if self.sc is not None:
            s += f"\tSingle cells shape: {self.sc.adata.shape[0]} x {self.sc.adata.shape[1]}"
        return s
    
    def __repr__(self):
        s = self.__str__()
        s += '\nobs: '
        s += ', '.join(list(self.adata.obs.columns))
        return s
    
    def __delitem__(self, key):
        if isinstance(key, str):
            if key in self.adata.obs:
                del self.adata.obs[key]
                self.__init_img()
            else:
                raise KeyError(f"'{key}' not found in adata.obs")
        else:
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
    
    def head(self, n=5):
        return self.adata.obs.head(n)
    
    @property
    def shape(self):
        return self.adata.shape
    
    def update(self):
        update_instance_methods(self)
        self.__init_img()
        # update also the sc
        if self.sc is not None:
            update_instance_methods(self.sc)
    
    def copy(self):
        return deepcopy(self)
    
    def save(self, path=None):
        if not path:
            path = f"{self.path_output}/{self.name}.pkl"
        else:
            if not path.endswith(".pkl"):
                path += ".pkl"
        with open(path, "wb") as f:
            dill.dump(self, f)
        return path

    @classmethod
    def load_ViziumHD(cls, filename, directory=''):
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        if directory:
            filename = f"{directory}/{filename}"
        with open(filename, "rb") as f:
            instance = dill.load(f)
        return instance


    
class SingleCell:
     def __init__(self, ):
         self.adata = None

     
     def plot_spatial(self, column, cells=False, celltypes=None, image=None, title=None, cmap=None, 
                   legend=False, alpha=True, figsize=(8, 8), 
                   xlim=None, ylim=None):
         pass
     
     def plot_cells(self, column, celltypes=None, image=None, title=None, cmap=None, 
                   legend=False, alpha=True, figsize=(8, 8), 
                   xlim=None, ylim=None):
         # requires cells annotations geopandas
         pass
     
     def plot_umap(self, features, out_path=None,title=None,size=None,
              figsize=(8,8),file_type='png',legend_loc='right margin'):
         pass
     
     
     def export_h5(self, path=None):
         pass
    
     @property
     def shape(self):
         return self.adata.shape
     
     def copy(self):
         return deepcopy(self)
        
        
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
   

def plot_hist(values, bins=10, show_zeroes=False, xlim=None, title=None, figsize=(8,8), 
              cmap=None, color="blue", ylab="Count",xlab=None):
    '''values: pd.Series'''
    fig, ax = plt.subplots(figsize=figsize)

    if np.issubdtype(values.dtype, np.number):
        if not show_zeroes:
            values = values[values > 0]
        counts, edges, patches = ax.hist(values,bins=bins,color=color)
        if xlim:
            ax.set_xlim(xlim)
        lower, upper = ax.get_xlim()
        relevant_counts = counts[(edges[:-1] >= lower) & (edges[:-1] <= upper)]
        max_count = relevant_counts.max() if len(relevant_counts) > 0 else counts.max()

    # Set ylim a little above the maximum count
        ax.set_ylim([0, max_count * 1.1])
    else: # Categorical case
        value_counts = values.value_counts()
        if isinstance(cmap, str):
            colors = get_colors(value_counts.index, cmap) if cmap else color
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


def plot_scatter(x, y, values, title=None, size=1, legend=True, xlab=None, ylab=None, 
                   cmap='viridis', figsize=(8, 8), alpha=1, legend_title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    if legend_title is None:
        legend_title = title
    if np.issubdtype(values.dtype, np.number): # Numeric case: Use colorbar
        scatter = plt.scatter(x, y, c=values, cmap=cmap, marker='s',
                              alpha=alpha, s=size,edgecolor='none')
        if legend:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label(legend_title)
    else: # Categorical case: Use legend 
        unique_values = np.unique(values.astype(str))
        unique_values = unique_values[unique_values != 'nan']
        if isinstance(cmap, str):
            colors = get_colors(unique_values, cmap)
            color_map = {val: colors[i] for i, val in enumerate(unique_values)}  
        elif isinstance(cmap, dict):
            color_map = {val: cmap.get(val,DEFAULT_COLOR) for val in unique_values}
        else:
            raise ValueError("cmap must be a string (colormap name) or a dictionary")
        print(f"{cmap=},{unique_values=},{color_map=}")
        for val in unique_values: # Plot each category with its color
            if values.dtype == bool:
                values = values.astype(str)
            mask = values == val
            ax.scatter(x[mask], y[mask], color=color_map[val], edgecolor='none',
                        label=str(val), marker='s', alpha=alpha, s=size)
        if legend:
            legend_elements = [Patch(facecolor=color_map[val], label=str(val)) for val in unique_values]
            ax.legend(handles=legend_elements, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    return ax

def get_colors(values, cmap):
    '''return a list of colors, in the length of the unique values, based on cmap'''
    if isinstance(values, pd.core.series.Series):
        unique_values = values.unique()
    else:
        unique_values = np.unique(values.astype(str))
    cmap_obj = colormaps.get_cmap(cmap)
    cmap_len = cmap_obj.N
    num_unique = len(unique_values)
    if num_unique <= cmap_len:
        colors = [cmap_obj(i / (num_unique - 1)) for i in range(num_unique)]
    else:
        # If there are more unique values than colors in the colormap, cycle through the colormap
        colors = [cmap_obj(i % cmap_len / (cmap_len - 1)) for i in range(num_unique)]
    return colors

def set_axis_ticks(ax, length_in_pixels, adjusted_microns_per_pixel, axis='x', num_ticks_desired=6):
    # Calculate the total length in microns
    total_microns = length_in_pixels * adjusted_microns_per_pixel

    # Define candidate step sizes in microns
    candidate_steps = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 1500, 2000]

    # Choose a step size that results in 5-7 ticks with round numbers
    for step in candidate_steps:
        num_ticks = total_microns / step
        if (num_ticks_desired-1) <= num_ticks <= (num_ticks_desired+1):
            break
    else:
        # If none of the candidate steps fit, calculate an approximate step size
        step = total_microns / num_ticks_desired
        step = round(step / 10) * 10  # Round to the nearest multiple of 10

    # Generate tick positions and labels
    tick_labels_microns = np.arange(0, total_microns + step, step)
    tick_positions_pixels = tick_labels_microns / adjusted_microns_per_pixel

    # Set ticks and labels on the specified axis
    if axis == "x":
        ax.set_xticks(tick_positions_pixels)
        ax.set_xticklabels([f"{int(tick)}" for tick in tick_labels_microns])
    elif axis == "y":
        ax.set_yticks(tick_positions_pixels)
        ax.set_yticklabels([f"{int(tick)}" for tick in tick_labels_microns])
    else:
        raise ValueError("Axis must be 'x' or 'y'")


        
    
    
    
    