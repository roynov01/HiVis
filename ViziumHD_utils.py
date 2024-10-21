# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:28:30 2024

@author: royno
"""
# General libraries
import importlib
import warnings
import sys
import os
import dill
from tqdm import tqdm
from copy import deepcopy
from subprocess import Popen, PIPE
# Data libraries
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import scanpy as sc
import math
# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps
import plotly.express as px
import seaborn as sns
from adjustText import adjust_text
# Image processing libraries
from PIL import Image
import tifffile

Image.MAX_IMAGE_PIXELS = 1063425001 # Enable large images loading
POINTS_PER_INCH = 72
MAX_BARS = 30 # in barplot
FULLRES_THRESH = 1000 # in microns, below which, a full-res image will be plotted
HIGHRES_THRESH = 3000 # in microns, below which, a high-res image will be plotted
PAD_CONSTANT = 0.3 # padding of squares in scatterplot
DEFAULT_COLOR ='None' # for plotting categorical
chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe' # to open HTML plots

class ViziumHD:
    '''
    A class for storage of VisiumHD data.
    Data methods:
        * add_mask() - assigns each spot a value based on mask (image)
        * add_annotations() - assigns each spot a value based on annotations (geogson)
        * find_markers() - runs differential gene expression analysis between two groups
        * add_meta() - adds a vector to metadata (obs or var)
        * update_meta() - renames metadata (obs or var)
        * export_h5() - exports data and metadata s h5 file
        * get() - get a vector of data or metadata (gene, obs or var)
    plotting methods:
        * plot_qc () - plots nUMI and mitochondrial %
        * plot_spatial() - plots image and/or spatial maps of data or metadata 
        * plot_hist() - histogram for numerical data or metadata, barplot for categorical
        * save_fig() - saves a figure
    general methods:
        * save() - exports the instance as pickle
        * loadViziumHD() - classmethod. loads instance from pickle file
        * update() - updates the instance methods
        * copy() - creates a copy of the instance
        * head() - returns the head of the obs
    dunder methods:
        * del - deletes metadata
        * getitem - get a vector of data or metadata (gene, obs or var)
    '''
    def __init__(self, path_input_fullres_image:str, path_input_data:str, path_output:str,
                 name:str, properties: dict = None, on_tissue_only=True,min_reads_in_spot=1,
                 min_reads_gene=10):
        '''
        - Loads images (fullres, highres, lowres)
        - Loads data and metadata
        - croppes the images based on the data
        - initializes the connection from the data and metadata to the images coordinates
        - performs basic QC (nUMI, mitochondrial %)
        parameters:
            * path_input_fullres_image - path for the fullres image
            * path_input_data - folder with outs of the Visium. typically square_002um
                                (with h5 files and with folders filtered_feature_bc_matrix, spatial)
            * path_output - path where to save plots and files
            * name - name of the instance
            * properties - dict of properties, such as organism, organ, sample_id
            * on_tissue_only - remove spots that are not classified as "on tissue"?
        '''
        self.sc = None
        self.name, self.path_output, self.properties = name, path_output, properties if properties else {}
        properties = {} if properties is None else properties
        self.organism = properties.get("organism")
        self.organ = properties.get("organ")
        self.sample_id = properties.get("sample_id")
        
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        
        hires_image_path = path_input_data + "/spatial/tissue_hires_image.png"
        lowres_image_path = path_input_data + "/spatial/tissue_lowres_image.png"
        json_path = path_input_data + "/spatial/scalefactors_json.json"
        metadata_path = path_input_data + "/spatial/tissue_positions.parquet"
        
        # validate paths of metadata and images
        validate_exists([path_input_fullres_image,hires_image_path,lowres_image_path,json_path,metadata_path])
        
        # load images
        print("[Loading images]")
        image_fullres = tifffile.imread(path_input_fullres_image)
        rgb_dim = image_fullres.shape.index(3)
        if rgb_dim != 2:  # If the RGB dimension is not already last
            axes_order = list(range(image_fullres.ndim))  # Default axes order
            axes_order.append(axes_order.pop(rgb_dim))  # Move the RGB dim to the last position
            image_fullres = image_fullres.transpose(axes_order)
        self.image_fullres = image_fullres
        self.image_highres = plt.imread(hires_image_path)
        self.image_lowres = plt.imread(lowres_image_path)
        
        # load json
        
        with open(json_path) as file:
            self.json = json.load(file)
        
        # load metadata      
        print("[Loading metadata]")        
        metadata = pd.read_parquet(metadata_path)
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
        
        # crop images and initiates micron to pixel conversions for plotting
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
        
        # Quality control - number of UMIs and mitochondrial %
        self.adata.obs["nUMI"] = np.array(self.adata.X.sum(axis=1).flatten())[0]
        self.adata.var["nUMI"] = np.array(self.adata.X.sum(axis=0).flatten())[0]
        mito_name_prefix = "MT-" if self.properties["organism"] == "human" else "mt-"
        mito_genes = self.adata.var_names[self.adata.var_names.str.startswith(mito_name_prefix)].values
        mito_sum = self.adata[:,self.adata.var.index.isin(mito_genes)].X.sum(axis=1).A1
        mito_percentage = (mito_sum / self.adata.obs["nUMI"]) * 100
        self.add_meta("mito_sum", mito_sum)
        self.add_meta("mito_percent", mito_percentage)
        
        # plot QC
        self.plot_qc(save=True)
        
        # filter low quality spots and lowly expressed genes
        self.adata = self.adata[self.adata.obs["nUMI"] >= min_reads_in_spot, :]
        self.adata = self.adata[:, self.adata.var["nUMI"] >= min_reads_gene]
        self.__init_img()
        
    def plot_qc(self, save=False,figsize=(10, 5)):
        '''plots basic QC (nUMI, mitochondrial %)'''
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)
        ax1 = self.plot_hist("mito_percent", title="Mito %", save=save, xlab="Mito %",ax=ax1)
        ax2 = self.plot_hist("nUMI", title="Number of UMIs", save=save, xlab="Number of unique reads",ax=ax2)
        plt.tight_layout()
        plt.title("QC - before filtration")
        if save:
            self.save_fig(filename="QC", fig=fig)
    
    def __init_img(self):
        '''resets the cropped image and updates the cropped adata'''
        self.image_cropped = None
        self.ax_current = None # stores the last plot that was made
        self.xlim_cur, self.pixel_x, self.ylim_cur, self.pixel_y = None, None, None, None
        self.adata_cropped = self.adata
        self.crop() # creates self.adata_cropped & self.image_cropped
        
    def __crop_img_permenent(self):
        '''
        crops the images, based on the coordinates from the metadata. 
        shifts the metadata to start at x=0, y=0.
        at first run, will save the cropped images.
        '''
        pxl_col_in_fullres = self.adata.obs["pxl_col_in_fullres"].values
        pxl_row_in_fullres = self.adata.obs["pxl_row_in_fullres"].values
        
        xlim_pixels_fullres = [math.floor(pxl_col_in_fullres.min()), math.ceil(pxl_col_in_fullres.max())]
        ylim_pixels_fullres = [math.floor(pxl_row_in_fullres.min()), math.ceil(pxl_row_in_fullres.max())]
        # Ensure the limits are within the image boundaries
        xlim_pixels_fullres = [max(0, xlim_pixels_fullres[0]), min(self.image_fullres.shape[1], xlim_pixels_fullres[1])]
        ylim_pixels_fullres = [max(0, ylim_pixels_fullres[0]), min(self.image_fullres.shape[0], ylim_pixels_fullres[1])]

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

    def __export_images(self, path_input_fullres_image, hires_image_path, lowres_image_path):
        '''helper method for __crop_img_permenent(). saves cropped images'''
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
    
    def add_mask(self, mask_path:str, name:str, plot=True, cmap="Paired"):
        '''
        assigns each spot a value based on mask (image).
        parameters:
            * mask_path - path to mask image
            * name - name of the mask (that will be called in the metadata)
            * plot - plot the mask?
            * cmap - colormap for plotting
        '''
        validate_exists(mask_path)
        mask_array = self.__import_mask(mask_path)
        if plot:
            self.__plot_mask(mask_array, cmap=cmap)
        self.__assign_spots(mask_array, name)
        self.__init_img()
        print(f"\nTo rename the values in the metadata, call the [update_meta] method with [{name}] and dictionary with current_name:new_name")
        return mask_array
    
    def __plot_mask(self, mask_array, cmap):
        '''helper method for add_mask(). plots the mask'''
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_array, cmap=cmap)
        num_colors = len(np.unique(mask_array[~np.isnan(mask_array)]))
        cmap = plt.cm.get_cmap(cmap, num_colors) 
        legend_elements = [Patch(facecolor=cmap(i), label=f'{i}') for i in range(num_colors)]
        plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.5))
        plt.show()
        
    def __import_mask(self, mask_path):
        '''helper method for add_mask(). imports the mask'''
        print("[Importing mask]")
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        return mask_array
    
    def __assign_spots(self, mask_array, name):
        '''helper method for add_mask(). assigns each spot a value from the mask'''
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
        
    def add_annotations(self, path:str, name:str):
        '''
        Adds annotations made in Qupath (geojson)
        parameters:
            * path - path to geojson file
            * name -  name of the annotation (that will be called in the metadata)
        '''
        validate_exists(path)
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
    
        
    def find_markers(self, column, group1, group2=None, method="wilcox"):
        '''
        Runs differential gene expression analysis between two groups.
        Values will be saved in self.var: expression_mean, log2fc, pval
        parameters:
            * column - which column in obs has the groups classification
            * group1 - specific value in the "column"
            * group2 - specific value in the "column". 
                       if None,will run agains all other values, and will be called "rest"
            * method - either "wilcox" or "t_test"
        '''
        # Get the expression of the two groups
        group1_exp = self.adata[self.adata.obs[column] == group1].copy()
        group1_exp = group1_exp[group1_exp.X.sum(axis=1) > 1]  # delete empty spots
        group1_exp.X = group1_exp.X / group1_exp.X.sum(axis=1).A1[:, None]  # matnorm
        group1_exp = group1_exp.X.todense()
        self.adata.var[group1] = group1_exp.mean(axis=0).A1  # save avarage expression of group1 to vars

        if group2 is None:
            group2_exp = self.adata[(self.adata.obs[column] != group1) & ~self.adata.obs[column].isna()].copy()
            group2 = "rest"
        else:
            group2_exp = self.adata[self.adata.obs[column] == group2].copy()
        group2_exp = group2_exp[group2_exp.X.sum(axis=1) > 1]  # delete empty spots
        group2_exp.X = group2_exp.X / group2_exp.X.sum(axis=1).A1[:, None]  # matnorm
        group2_exp = group2_exp.X.todense()
        self.adata.var[group2] = group2_exp.mean(axis=0).A1  # save avarage expression of group2 to vars
        
        # Calculate mean expression in each group and log2(group1/group2)
        self.adata.var[f"expression_mean_{column}"] = self.adata.var[[group1,group2]].mean(axis=1)
        pn = self.adata.var[f"expression_mean_{column}"][self.adata.var[f"expression_mean_{column}"]>0].min()
        self.adata.var[f"log2fc_{column}"] = (self.adata.var[group1] + pn) / (self.adata.var[group2] + pn)
        self.adata.var[f"log2fc_{column}"] = np.log2(self.adata.var[f"log2fc_{column}"])  
        
        # Wilcoxon rank-sum test
        self.adata.var[f"pval_{column}"] = np.nan
        for j, gene in enumerate(tqdm(self.adata.var.index, desc=f"Running wilcoxon on [{self.name}][{column}]")):
            if (self.adata.var.loc[gene,group1] == 0) and (self.adata.var.loc[gene,group2] == 0):
                p_value = np.nan
            else:
                cur_gene_group1 = group1_exp[:,j]
                cur_gene_group2 = group2_exp[:,j]
                if method == "wilcox":
                    from scipy.stats import mannwhitneyu
                    _, p_value = mannwhitneyu(cur_gene_group1, cur_gene_group2, alternative="two-sided")
                elif method == "t_test":
                    from scipy.stats import ttest_ind
                    _, p_value = ttest_ind(cur_gene_group1, cur_gene_group2, alternative="two-sided")
            self.adata.var.loc[gene,f"pval_{column}"] = p_value
    
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
        
     
    def save_fig(self, filename:str, fig=None, ax=None, open_file=False, format='png', dpi=300):
        '''
        saves a figure or ax. 
        parameters:
            * filename - name of plot
            * fig (optional) - plt.Figure object to save.
            * ax - ax to save. if not passed, will use self.current_ax
            * open_file - open the file?
            * format - format of file
        '''
        if fig is None:
            if ax is None:
                if self.current_ax is None:
                    print(f"No ax present in {self.name}")
                    return
                ax = self.current_ax
            fig = ax.get_figure()
        path = f"{self.path_output}/{self.name}_{filename}.{format}"
        fig.savefig(path, format=format, dpi=dpi, bbox_inches='tight')
        if open_file:
            os.startfile(path)

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
    
    def __get_dot_size(self, adjusted_microns_per_pixel:float):
        '''gets the size of spots, depending on adjusted_microns_per_pixel'''
        bin_size_pixels = self.json['bin_size_um'] / adjusted_microns_per_pixel 
        dpi = plt.gcf().get_dpi()
        # dpi = mpl.rcParams['figure.dpi']
        points_per_pixels = POINTS_PER_INCH / dpi
        dot_size = bin_size_pixels * points_per_pixels 
        return dot_size
        
    def plot_spatial(self, what=None, image=True, ax=None, title=None, cmap="viridis", 
                  legend=True, alpha=1, figsize=(8, 8), save=False,
                  xlim=None, ylim=None, legend_title=None, axis_labels=True, pad=False):
        '''
        plots the image, and/or data/metadata (spatial plot)
        parameters:
            * what - what to plot. can be metadata (obs/var colnames or a gene)
            * image - plot image?
            * ax (optional) - matplotlib ax, if not passed, new figure will be created with size=figsize
            * cmap - colorbar to use
            * title, legend_title, axis_labels - strings
            * legend - show legend?
            * xlim - two values, in microns
            * ylim - two values, in microns
            * pad - scale the size of dots to be smaller
            * alpha - transparency of scatterplot. value between 0 and 1
            * save - save the image?
        '''
        title = what if title is None else title
        if legend_title is None:
            legend_title = what.capitalize() if what and what==what.lower else None
            
        xlim, ylim, adjusted_microns_per_pixel = self.crop(xlim, ylim)
        size = self.__get_dot_size(adjusted_microns_per_pixel)
        if pad:
            size *= PAD_CONSTANT
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)


        if image: # Plot image
            # ax.imshow(self.image_cropped, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])
            ax.imshow(self.image_cropped)

        if what: 
            values = self.get(what, cropped=True)
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
        if title:
            ax.set_title(title)    
            
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
    
    def plot_hist(self, what, bins=20, xlim=None, title=None, ylab=None,xlab=None,ax=None,
             save=False, figsize=(8,8), cmap=None, color="blue"):
        '''
        plots histogram of data or metadata. if categorical, will plot barplot
        parameters:
            * what - what to plot. can be metadata (obs/var colnames or a gene)
            * bins - number of bins of the histogram
            * ax (optional) - matplotlib ax, if not passed, new figure will be created with size=figsize
            * cmap - colorbar to use. overrides the color argument for barplot
            * color - color of the histogram
            * title, xlab, ylab - strings
            * xlim - two values, where to crop the x axis
            * save - save the image?
        '''
        title = what if title is None else title
        self.crop() # resets adata_cropped to full image
        to_plot = pd.Series(self.get(what, cropped=True))
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = plot_histogram(to_plot,bins=bins,xlim=xlim,title=title,figsize=figsize,
                       cmap=cmap,color=color,ylab=ylab,xlab=xlab,ax=ax)            
        self.current_ax = ax
        if save:
            self.save_fig(f"{what}_HIST")
        return ax
            
    def export_h5(self, path=None):
        '''exports the adata. can also save the obs as parquet'''
        if not path:
            path = f"{self.path_output}/{self.name}_viziumHD.h5ad"
        self.adata.write(path)
        return path

    def sc_create(self, category):
        if self.sc:
            if input('Single cell allready exists, if you want to aggregate again pres "y"') not in ("y","Y"):
                return
        params = []
        self.sc = SingleCell(self, params)

    def sc_transfer_meta(self, what:str):
        '''transfers metadata assignment from the single-cell to the spots'''
        pass
    
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
            copy = self.copy()
            copy.adata = copy.adata[what]
            copy.__init_img()
            return copy
        
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
        if self.sc is not None:
            s += f"\tSingle cells shape: {self.sc.adata.shape[0]} x {self.sc.adata.shape[1]}"
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
        update_instance_methods(self)
        self.__init_img()
        # update also the sc
        if self.sc is not None:
            update_instance_methods(self.sc)
    
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

    @classmethod
    def load_ViziumHD(cls, filename, directory=''):
        '''loads an instance from a pickle format'''
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        if directory:
            filename = f"{directory}/{filename}"
        validate_exists(filename)
        with open(filename, "rb") as f:
            instance = dill.load(f)
        return instance


    
class SingleCell:
     def __init__(self, vizium_instance):
         self.viz = vizium_instance
         self.adata = None
         self.path_output = self.viz.path_output + "/single_cell"

     def __init_img(self):
         pass
     
     def crop(self):
         pass
     
     def plot_spatial(self, ):
         pass
     
     def plot_cells(self, column, celltypes=None, image=None, title=None, cmap=None, 
                   legend=False, alpha=True, figsize=(8, 8), 
                   xlim=None, ylim=None):
         # requires cells annotations geopandas
         pass
     
     def plot_umap(self, features, out_path=None,title=None,size=None,
              figsize=(8,8),file_type='png',legend_loc='right margin'):
         pass
     
     def plot_hist(self):
         pass
     
     
     def export_h5(self, path=None):
         if not path:
             path = f"{self.path_output}/{self.name}_viziumHD.h5ad"
         self.adata.write(path)
         return path
    
     @property
     def shape(self):
         return self.adata.shape
     
     def __str__(self):
         s = f"# single-cells # {self.viz.name} #\n"
         s += f"\tSize: {self.adata.shape[0]} x {self.adata.shape[1]}\n"
         s += '\nobs: '
         s += ', '.join(list(self.adata.obs.columns))
         s += '\n\nvar: '
         s += ', '.join(list(self.adata.var.columns))
         return s
     
     def __repr__(self):
         s = f"SingleCell[{self.viz.name}]"
         return s
     
     def __delitem__(self, key):
         '''deletes metadata'''
         if isinstance(key, str):
             if key in self.adata.obs:
                 del self.adata.obs[key]
             elif key in self.adata.var:
                 del self.adata.var[key]
             else:
                 raise KeyError(f"'{key}' not found in adata.obs")
             self.__init_img()
         else:
             raise TypeError(f"Key must be a string, not {type(key).__name__}")
     
     def head(self, n=5):
         return self.adata.obs.head(n)   
        
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
   

def plot_histogram(values, bins=10, show_zeroes=False, xlim=None, title=None, figsize=(8,8), 
              cmap=None, color="blue", ylab="Count",xlab=None,ax=None):
    '''values: pd.Series'''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')

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
        
def open_html(html_file,chrome_path=chrome_path):
    process = Popen(['cmd.exe', '/c', chrome_path, html_file], stdout=PIPE, stderr=PIPE)

def scatter(df,x,y,save_path,text="gene",color=None,size=None,xlab=None,ylab=None,title=None,open_fig=True,legend_title=None):
    if color:
        if size:
            legend_title = [color, size] if not legend_title else legend_title
            fig = px.scatter(df, x=x, y=y,hover_data=[text],color=color,size=size, labels={color: legend_title[0],size: legend_title[1]})
        else:
            legend_title = color if not legend_title else legend_title
            fig = px.scatter(df, x=x, y=y,hover_data=[text],color=color, labels={color: legend_title})
    else:
        fig = px.scatter(df, x=x, y=y,hover_data=[text],color=color,size=size)
    fig.update_traces(marker_size=10, 
        hoverinfo='text+x+y',
        # text=df[text], 
        mode='markers+text')
    if legend_title is None:
        legend_title = color
    fig.update_layout(template="simple_white",
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        title_font=dict(size=30, family="Arial", color="Black"),
        xaxis_title_font=dict(size=24, family="Arial", color="Black"),
        yaxis_title_font=dict(size=24, family="Arial", color="Black"))
    fig.write_html(save_path) 
    if open_fig:
        open_html(save_path)    
    
def scatter_seaborn(df,x_col,y_col,genes=None,figsize=(8,8),size=10,legend=False,
                    ax=None,xlab=None,ylab=None,out_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    df['type'] = ""
    
    sns.scatterplot(data=df[df['type'] == ''], x=x_col, y=y_col,s=size,legend=legend,
                    ax=ax,color="blue",edgecolor=None)
    ax.axhline(y=0,color="k",linestyle="--")
    if genes:
        df.loc[df['gene'].isin(genes),"type"] = "selected"
        subplot = df[df['type'] != '']
        if not subplot.empty:
            sns.scatterplot(data=subplot, x=x_col, y=y_col,color="red",
                            s=size,legend=False,ax=ax,edgecolor="k")
        texts = [ax.text(
            subplot[x_col].iloc[i], 
            subplot[y_col].iloc[i], 
            subplot['gene'].iloc[i],
            color="red",
            fontsize=14,
            ) for i in range(len(subplot))]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                    force_text=(0.6, 0.6))
    
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    if out_path:
        if not out_path.endswith(".png"):
            out_path += ".png"
        plt.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    
    return ax
    
def validate_exists(file_path):
    if isinstance(file_path, (list, tuple)):
        for path in file_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No such file or directory:\n\t{path}")
    else:
         if not os.path.exists(file_path):
             raise FileNotFoundError(f"No such file or directory:\n\t{file_path}")    
             
def fisher_method(pvalues):
    from scipy.stats import chi2
    pvalues = pvalues.dropna()
    k = len(pvalues)
    if k == 0:
        return np.nan
    chi_stat = -2 * np.sum(np.log(pvalues))
    p_combined = 1 - chi2.cdf(chi_stat, 2 * k)
    return p_combined