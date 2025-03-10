# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:29:05 2024

@author: royno
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from adjustText import adjust_text
import plotly.express as px
from subprocess import Popen, PIPE
import shapely.wkt
import shapely.affinity
import geopandas as gpd
import tempfile
import time



POINTS_PER_INCH = 72
MAX_SQUARES_TO_DRAW_EXACT = 500 # how many squares to draw in perfect positions in spatial plot
MAX_BARS = 30 # in barplot
PAD_CONSTANT = 0.3 # padding of squares in scatterplot
DEFAULT_COLOR ='None' # for plotting categorical
chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
FULLRES_THRESH = 1000 # in microns, below which, a full-res image will be plotted
HIGHRES_THRESH = 3000 # in microns, below which, a high-res image will be plotted

class PlotVizium:
    '''Handles all plotting for ViziumHD object'''
    def __init__(self, vizium_instance):
        self.main = vizium_instance
        self.current_ax = None
        self.xlim_max = (vizium_instance.adata.obs['um_x'].min(), vizium_instance.adata.obs['um_x'].max())
        self.ylim_max = (vizium_instance.adata.obs['um_y'].min(), vizium_instance.adata.obs['um_y'].max())
        
    def _crop(self, xlim=None, ylim=None, resolution=None):
        '''
        Crops the images and adata based on xlim and ylim in microns. 
        saves it in self.adata_cropped and self.image_cropped
        xlim, ylim: tuple of two values, in microns
        resolution - if None, will determine automatically, other wise, "full","high" or "low"
        '''
        microns_per_pixel = self.main.json['microns_per_pixel'] 
    
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
            image = self.main.image_fullres
            scalef = 1  # No scaling needed for full-resolution
            pxl_col, pxl_row = 'pxl_col_in_fullres', 'pxl_row_in_fullres'
        elif resolution == "high":
            image = self.main.image_highres
            scalef = self.main.json['tissue_hires_scalef']
            pxl_col, pxl_row = 'pxl_col_in_highres', 'pxl_row_in_highres'
        elif resolution == "low":
            image = self.main.image_lowres
            scalef = self.main.json['tissue_lowres_scalef']
            pxl_col, pxl_row = 'pxl_col_in_lowres', 'pxl_row_in_lowres'       
        elif lim_size <= FULLRES_THRESH: # Use full-resolution image
            image = self.main.image_fullres
            scalef = 1  # No scaling needed for full-resolution
            pxl_col, pxl_row = 'pxl_col_in_fullres', 'pxl_row_in_fullres'
            print("Full-res image selected")
        elif lim_size <= HIGHRES_THRESH: # Use high-resolution image
            image = self.main.image_highres
            scalef = self.main.json['tissue_hires_scalef']  
            pxl_col, pxl_row = 'pxl_col_in_highres', 'pxl_row_in_highres'
            print("High-res image selected")
        else: # Use low-resolution image
            image = self.main.image_lowres
            scalef = self.main.json['tissue_lowres_scalef']  
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
        x_mask = (self.main.adata.obs['um_x'] >= xlim[0]) & (self.main.adata.obs['um_x'] <= xlim[1])
        y_mask = (self.main.adata.obs['um_y'] >= ylim[0]) & (self.main.adata.obs['um_y'] <= ylim[1])
        mask = x_mask & y_mask
    
        # Crop the adata
        self.main.adata_cropped = self.main.adata[mask]
    
        # Adjust adata coordinates relative to the cropped image
        self.pixel_x = self.main.adata_cropped.obs[pxl_col] - xlim_pxl[0]
        self.pixel_y = self.main.adata_cropped.obs[pxl_row] - ylim_pxl[0]
    
        return xlim, ylim, adjusted_microns_per_pixel 
    
    def _init_img(self):
        '''resets the cropped image and updates the cropped adata'''
        self.image_cropped = None
        self.ax_current = None # stores the last plot that was made
        self.pixel_x, self.pixel_y = None, None 
        self.main.adata_cropped = self.main.adata
        self._crop() # creates self.main.adata_cropped & self.image_cropped
    
    def save(self, figname:str, fig=None, ax=None, open_file=False, format_='png', dpi=300):
        '''
        saves a figure or ax. 
        parameters:
            * figname - name of plot
            * fig (optional) - plt.Figure object to save, can be a dataframe for writing csv.
            * ax - ax to save. if not passed, will use self.current_ax
            * open_file - open the file?
            * format - format of file
        '''
        path = f"{self.main.path_output}/{self.main.name}_{figname}.{format_}"
        if fig is None:
            if ax is None:
                if self.current_ax is None:
                    raise ValueError(f"No ax present in {self.main.name}")
                ax = self.current_ax
            fig = ax.get_figure()
        return save_fig(path, fig, open_file, format_, dpi)
    
    def __get_dot_size(self, adjusted_microns_per_pixel:float):
        '''gets the size of spots, depending on adjusted_microns_per_pixel'''
        bin_size_pixels = self.main.json['bin_size_um'] / adjusted_microns_per_pixel 
        dpi = plt.gcf().get_dpi()
        # dpi = mpl.rcParams['figure.dpi']
        points_per_pixels = POINTS_PER_INCH / dpi
        dot_size = bin_size_pixels * points_per_pixels 
        return dot_size
    
    
    def spatial(self, what=None, image=True, img_resolution=None, ax=None, title=None, cmap="winter", 
                  legend=True, alpha=1, figsize=(8, 8), save=False, exact=None,brightness=0,contrast=1,
                  xlim=None, ylim=None, legend_title=None, axis_labels=True, pad=False):
        '''
        plots the image, and/or data/metadata (spatial plot)
        parameters:
            * what - what to plot. can be metadata (obs/var colnames or a gene)
            * image - plot image?
            * img_resolution - "low","high","full". If None, will determine automatically
            * ax - matplotlib ax, if not passed, new figure will be created with size=figsize
            * cmap - colorbar to use
            * title, legend_title, axis_labels - strings
            * legend - show legend?
            * xlim, ylim - two values each, in microns
            * pad - scale the size of dots to be smaller
            * alpha - transparency of scatterplot. value between 0 and 1
            * save - save the image?
            * exact - plot the squares at the exact size? more time costly
            * brightness - increases brigtness, for example 0.2. 
            * contrast - > 1 increases contrast, < 1 decreases.
        '''
        title = what if title is None else title
        if legend_title is None:
            legend_title = what.capitalize() if what and what==what.lower else what
        xlim, ylim, adjusted_microns_per_pixel = self._crop(xlim, ylim, resolution=img_resolution)
        if exact is None:
            if (xlim[1] - xlim[0] + ylim[1] - ylim[0]) <= MAX_SQUARES_TO_DRAW_EXACT:
                exact = True
            else:
                exact = False
        if exact:
            size = self.main.json['bin_size_um']/adjusted_microns_per_pixel
        else: 
            size = self.__get_dot_size(adjusted_microns_per_pixel)
            if pad:
                size *= PAD_CONSTANT
            
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        height, width = self.image_cropped.shape[:2]  
        if image: # Plot image
            extent = None
            # ax.imshow(self.image_cropped, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])
            if exact:
                extent = [0, width, height, 0]
            img = self.image_cropped.copy()
            if contrast != 1:
                # Change contrast
                mean_value = np.mean(img)  
                img_contrast = mean_value + contrast * (img - mean_value)
                img = np.clip(img_contrast, 0, 1)
                # Change brigtness
            if brightness:
                img = np.clip(img + brightness, 0, 1)
                
            ax.imshow(img, extent=extent)

        if what: 
            values = self.main.get(what, cropped=True)
            if values is None:
                raise ValueError(f"{what} not found in adata")
            if np.issubdtype(values.dtype, np.number):  # Filter values that are 0
                if np.nansum(values) == 0:
                    raise ValueError(f"{what} is equal to zero in the specified xlim,ylim")
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
            if exact:
                ax = _plot_squares_exact(x, y, values, size=size,title=title,
                              alpha=alpha,cmap=cmap,ax=ax,
                              legend=legend,xlab=None,ylab=None, 
                              legend_title=legend_title)
                ax.set_aspect('equal')
            else:
                ax = plot_scatter(x, y, values, size=size,title=title,
                              alpha=alpha,cmap=cmap,ax=ax,
                              legend=legend,xlab=None,ylab=None, 
                              legend_title=legend_title)

        if axis_labels:
            ax.set_xlabel("Spatial 1 (µm)")
            ax.set_ylabel("Spatial 2 (µm)")
            set_axis_ticks(ax, width, adjusted_microns_per_pixel, axis='x')
            set_axis_ticks(ax, height, adjusted_microns_per_pixel, axis='y')
        else:
            ax.set_xticks([])  
            ax.set_xticklabels([]) 
            ax.set_yticks([])  
            ax.set_yticklabels([]) 
        if title:
            ax.set_title(title)    
            
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)     
        
        # Save figure:
        self.current_ax = ax
        if save:
            self.save(f"{(what + '_') if what else ''}SPATIAL")
        return ax
    
    def hist(self, what, bins=20, xlim=None, title=None, ylab=None,xlab=None,ax=None,
             save=False, figsize=(8,8), cmap=None, color="blue",cropped=True):
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
            * cropped - if False and plot.spatial was run with xlim, ylim hist will be on cropped area
        '''
        title = what if title is None else title
        if cropped:
            self._crop() # resets adata_cropped to full image
        to_plot = pd.Series(self.main.get(what, cropped=True))
        if to_plot is None:
            raise ValueError(f"'{what}' not in adata")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = plot_histogram(to_plot,bins=bins,xlim=xlim,title=title,figsize=figsize,
                       cmap=cmap,color=color,ylab=ylab,xlab=xlab,ax=ax)            
        self.current_ax = ax
        if save:
            self.save(f"{what}_HIST")
        return ax
    
    
    def __repr__(self):
        s = f"Plots available for [{self.main.name}]:\n\tsave(), spatial(), hist()"
        if self.main.SC:
            s += "\n\nand for sc:\n\tsave(), spatial(), hist(), cells(), umap()"
        return s
    
    
class PlotSC:
    '''Handles all plotting for SingleCell object'''
    def __init__(self, sc_instance):
        self.main = sc_instance
        self.current_ax = None
        self.xlim_max = (self.main.viz.adata.obs['um_x'].min(), self.main.viz.adata.obs['um_x'].max())
        self.ylim_max = (self.main.viz.adata.obs['um_y'].min(), self.main.viz.adata.obs['um_y'].max())
        self.geometry = None
        self._crop()
        
    def _crop(self, xlim=None, ylim=None, resolution=None, geometry=False):
        '''
        Creates self.main.adata_cropped, based on xlim, ylim, in um units.
        parameters:
            * xlim,ylim both have two elements. if None, will take the maximal limits.
            * resolution - can be "full","high","low" or None
            * geometry - initialize / update self.geometry?
        '''
        # If xlim or ylim is None, set to the full range of the data
        if xlim is None:
            xlim = self.xlim_max
        if ylim is None:
            ylim = self.ylim_max
    
        x_mask = (self.main.adata.obs['um_x'] >= xlim[0]) & (self.main.adata.obs['um_x'] <= xlim[1])
        y_mask = (self.main.adata.obs['um_y'] >= ylim[0]) & (self.main.adata.obs['um_y'] <= ylim[1])
        mask = x_mask & y_mask
    
        # Crop the adata
        self.main.adata_cropped = self.main.adata[mask].copy()
    
        # Adjust adata coordinates relative to the cropped image
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        lim_size = max(x_range, y_range)
        
        if resolution == "full":
            pxl_col, pxl_row, scalef = 'pxl_col_in_fullres', 'pxl_row_in_fullres', 1
        elif resolution == "high":
            pxl_col, pxl_row, scalef = 'pxl_col_in_highres', 'pxl_row_in_highres', self.main.viz.json['tissue_hires_scalef']
        elif resolution == "low":
            pxl_col, pxl_row, scalef = 'pxl_col_in_lowres', 'pxl_row_in_lowres', self.main.viz.json['tissue_lowres_scalef']      
        elif lim_size <= FULLRES_THRESH: 
            pxl_col, pxl_row, scalef = 'pxl_col_in_fullres', 'pxl_row_in_fullres', 1
        elif lim_size <= HIGHRES_THRESH: 
            pxl_col, pxl_row, scalef = 'pxl_col_in_highres', 'pxl_row_in_highres', self.main.viz.json['tissue_hires_scalef'] 
        else: 
            pxl_col, pxl_row, scalef = 'pxl_col_in_lowres', 'pxl_row_in_lowres', self.main.viz.json['tissue_lowres_scalef']  
        microns_per_pixel = self.main.viz.json['microns_per_pixel'] 
        adjusted_microns_per_pixel = microns_per_pixel / scalef        
        xlim_pxl = [int(lim/ adjusted_microns_per_pixel) for lim in xlim]
        ylim_pxl = [int(lim/ adjusted_microns_per_pixel) for lim in ylim]
 
        self.pixel_x = self.main.adata_cropped.obs[pxl_col] - xlim_pxl[0]
        self.pixel_y = self.main.adata_cropped.obs[pxl_row] - ylim_pxl[0]
        
        if geometry:
            self._init_geometry(adjusted_microns_per_pixel, xlim_pxl, ylim_pxl)
            
            
    def _init_geometry(self, adjusted_microns_per_pixel, xlim_pxl, ylim_pxl):
        """
        Initialize or refresh self._geometry from self.main.adata.obs["geometry"].
        If self._geometry is already defined, you could skip re-initializing
        unless you've changed the data externally.
        """
        obs = self.main.adata_cropped.obs
        if "geometry" not in obs.columns:
            print("'geometry' column isn't in OBS")
            self.geometry = None
            return
        
        # Convert WKT → Shapely geometry
        geometry = (obs["geometry"].dropna().apply(shapely.wkt.loads))
        
        # Build a GeoDataFrame in micron space (before scaling)
        gdf = gpd.GeoDataFrame(obs.drop(columns="geometry"),geometry=geometry,crs=None) # EPSG:4326?
        
        # Scale from microns to pixels 
        scale_factor = 1.0 / adjusted_microns_per_pixel
        gdf = gdf.dropna(subset=["geometry"]).copy()

        
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: shapely.affinity.scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        )
        
        # Shift so that (xlim[0], ylim[0]) to (0,0)
        x_shift = -xlim_pxl[0]
        y_shift = -ylim_pxl[0]
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: shapely.affinity.translate(geom, xoff=x_shift, yoff=y_shift))
        
        self.geometry = gdf
    
    def save(self, figname:str, fig=None, ax=None, open_file=False, format_='png', dpi=300):
        '''
        saves a figure or ax. 
        parameters:
            * figname - name of plot
            * fig (optional) - plt.Figure object to save, can be a dataframe for writing csv.
            * ax - ax to save. if not passed, will use self.current_ax
            * open_file - open the file?
            * format_ - format of file
        '''
        path = f"{self.main.path_output}/{self.main.viz.name}_{figname}.{format_}"
        if fig is None:
            if ax is None:
                if self.current_ax is None:
                    raise ValueError(f"No ax present in {self.main.viz.name}")
                ax = self.current_ax
            fig = ax.get_figure()
        return save_fig(path, fig, open_file, format_, dpi)
    
    
    def spatial(self, what=None, image=True, img_resolution=None, ax=None, title=None, cmap="winter", 
                  legend=True, alpha=1, figsize=(8, 8), save=False, size=1,brightness=0,contrast=1,
                  xlim=None, ylim=None, legend_title=None, axis_labels=True):
        '''
        Plot a spatial representation of self.adata.
        parameters:
            * what - what to color the cells with (fill) - can be column from obs or a gene
            * image - plot the image underneath the cells?
            * img_resolution - which resolution to use for the image - can be "full","high","low"
            * brightness, contrast - for image modification
            * cmap - can be string (name of pellate), list of colors, 
                     or in categorical values case, a dict {"value":"color"}
            * xlim, ylim - two values each, in microns
            * save - save the plot?
            * figsize, legend, alpha, title, legend_title, axis_labels - cosmetic parameters  
        '''
        title = what if title is None else title
        if legend_title is None:
            legend_title = what.capitalize() if what and what==what.lower else what

        self._crop(xlim, ylim, resolution=img_resolution)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = self.main.viz.plot.spatial(image=image, ax=ax,brightness=brightness,title=title,
                            contrast=contrast,xlim=xlim,ylim=ylim,img_resolution=img_resolution,
                            axis_labels=axis_labels)
        
        if what: 
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            
            values = self.main.get(what, cropped=True)
            if np.issubdtype(values.dtype, np.number):  # Filter values that are 0
                mask = values > 0
            else:
                mask = [True for _ in values]   # No need for filtering
            values = values[mask]
            x = self.pixel_x[mask]
            y = self.pixel_y[mask]
            # height = self.main.viz.plot.image_cropped.shape[0]
            # self.pixel_y = height - self.pixel_y # Flip Y axis
            
            if np.issubdtype(values.dtype, np.number): 
                argsort_values = np.argsort(values)
                x, y, values = x.iloc[argsort_values], y.iloc[argsort_values], values[argsort_values]
            
            # Plot scatter:
            
            ax = plot_scatter(x, y, values, size=size,title=title,
                          alpha=alpha,cmap=cmap,ax=ax,
                          legend=legend,xlab=None,ylab=None, 
                          legend_title=legend_title,marker="o")



        # Save figure:
        self.current_ax = ax
        if save:
            self.save(f"{(what + '_') if what else ''}SPATIAL")
        return ax

    def hist(self, what, bins=20, xlim=None, title=None, ylab=None,xlab=None,ax=None,
             save=False, figsize=(8,8), cmap=None, color="blue",cropped=True):
        '''
        plots histogram of data or metadata. if categorical, will plot barplot.
        parameters:
            * what - what to plot. can be metadata (obs/var colnames or a gene)
            * bins - number of bins of the histogram
            * ax (optional) - matplotlib ax, if not passed, new figure will be created with size=figsize
            * cmap - colorbar to use. overrides the color argument for barplot
            * color - color of the histogram
            * title, xlab, ylab - strings
            * save - save the plot?
            * xlim - two values, where to crop the x axis
            * save - save the image?
        '''
        title = what if title is None else title
        if cropped:
            self._crop() # resets adata_cropped to full image
        to_plot = pd.Series(self.main.get(what, cropped=True))
        if to_plot is None:
            raise ValueError(f"'{what}' not in adata")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = plot_histogram(to_plot,bins=bins,xlim=xlim,title=title,figsize=figsize,
                       cmap=cmap,color=color,ylab=ylab,xlab=xlab,ax=ax)            
        self.current_ax = ax
        if save:
            self.save(f"{what}_HIST")
        return ax
    
    def cells(self, what=None, image=True, img_resolution=None, xlim=None, ylim=None, 
              figsize=(8, 8), line_color="black",cmap="viridis", alpha=0.7, linewidth=1,save=False,
              legend=True, ax=None, title=None, legend_title=None, brightness=0,contrast=1,axis_labels=True):
        '''
        Plot a UMAP of self.adata.
        parameters:
            * what - what to color the cells with (fill) - can be column from obs or a gene
            * image - plot the image underneath the cells?
            * img_resolution - which resolution to use for the image - can be "full","high","low"
            * brightness, contrast - for image modification
            * cmap - can be string (name of pellate), list of colors, 
                     or in categorical values case, a dict {"value":"color"}
            * xlim, ylim - two values each, in microns
            * save - svae the plot?
            * figsize, line_color, legend, linewidth, title, legend_title, axis_labels - cosmetic parameters            
        '''
        if "geometry" not in self.main.adata.obs.columns:
            raise ValueError("No 'geometry' column found in adata.obs.")
            
        self._crop(xlim=xlim, ylim=ylim, resolution=img_resolution, geometry=True)
        
        if self.geometry.empty:
            raise ValueError(f"No cells found in limits x={xlim}, y={ylim}")
        
        title = what if title is None else title
        if legend_title is None:
            legend_title = what.capitalize() if what and what==what.lower else what
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = self.main.viz.plot.spatial(image=image, ax=ax,brightness=brightness,title=title,axis_labels=axis_labels,
                            contrast=contrast,xlim=xlim,ylim=ylim,img_resolution=img_resolution)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if line_color is not None:
            self.geometry.boundary.plot(ax=ax, color=line_color, linewidth=linewidth)
        
        if what: 
            values = self.main.get(what, cropped=True, geometry=True) 
            if values is None:
                raise KeyError(f"No values in [{what}]")
            # if len(values) != len(self.main.adata_cropped):
            #     raise ValueError("Can only plot OBS or gene expression")
            self.geometry["temp"] = values
            
            if np.issubdtype(values.dtype, np.number):
                if isinstance(cmap, str):
                    cmap_obj = colormaps.get_cmap(cmap)
                elif isinstance(cmap, list):
                    cmap_obj = LinearSegmentedColormap.from_list("custom_cmap", cmap)
                self.geometry.plot(column="temp", ax=ax, cmap=cmap_obj, legend=False, alpha=alpha)
                if legend:
                    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
                    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                    sm.set_array([])  
                    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
                    cbar.set_label(legend_title)
            else: # Categorical case
                unique_values = np.unique(values.astype(str))
                unique_values = unique_values[unique_values != 'nan']
                if isinstance(cmap, (str,list)):
                    colors = get_colors(unique_values, cmap)
                    color_map = {val: colors[i] for i, val in enumerate(unique_values)}  
                elif isinstance(cmap, dict):
                    color_map = {val: cmap.get(val,DEFAULT_COLOR) for val in unique_values}
                else:
                    raise ValueError("cmap must be a string (colormap name) or a dictionary")
                for val in unique_values: # Plot each category with its color
                    values = values.astype(str)
                    mask = (self.geometry["temp"].astype(str) == val)
                    sub_gdf = self.geometry[mask]
                    if sub_gdf.empty:
                        continue
                    sub_gdf.plot(ax=ax,facecolor=color_map[val],edgecolor="none",alpha=alpha,label=str(val))
                if legend:
                    legend_elements = [Patch(facecolor=color_map[val], label=str(val)) for val in unique_values]
                    ax.legend(handles=legend_elements, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))
      
            # self.geometry.plot(column="temp",ax=ax,cmap=cmap,legend=legend,alpha=alpha)
            self.geometry.drop(columns="temp", inplace=True)
        self.current_ax = ax
        if save:
            self.save(f"{what}_CELLS")
        return ax

        
    def umap(self, features=None, title=None, size=None,layer=None,legend=True,texts=False,
              legend_loc='right margin', save=False, ax=None, figsize=(8,8),cmap="viridis"):
        '''
        Plot a UMAP of self.adata.
        parameters:
            * features - if None, won't color. can be a string or list of strings,
                        passed to scanpy.pl.umap
            * layer - which layer to use from the self.adata. If None, will use X
            * texts - add text in the center of mass of categorical case
            * cmap - can be string (name of pellate), list of colors, 
                     or in categorical values case, a dict {"value":"color"}
            * figsize, size, legend, legend_loc, title, legend_title - cosmetic parameters            
        '''
        if 'X_umap' not in self.main.adata.obsm:
            raise ValueError("UMAP embedding is missing. Run `sc.tl.umap()` after PCA.")
        if ax:
            if not isinstance(features, str):
                raise ValueError("ax can be passed for a single feature only")
        else:
            if isinstance(features, str):
                features = [features]
                fig, ax = plt.subplots(figsize=figsize)
        if not legend:
            legend_loc="none"
        
        color_values = self.main[features[0] if isinstance(features, list) else features] 
        if isinstance(cmap, (str, list, dict)):
            colors = get_colors(color_values, cmap)
        unique_values = np.unique(color_values.astype(str))
        if len(unique_values) == len(colors):
            self.main.adata.uns[f'{features[0]}_colors'] = colors  # Set colors for the feature categories
        else:
            raise ValueError("Mismatch between number of unique values and generated colors.")    
        ax = sc.pl.umap(self.main.adata, color=features,use_raw=False,size=size,ax=ax,
                        title=title,show=False,legend_loc=legend_loc,layer=layer)
        del self.main.adata.uns[f'{features[0]}_colors']

        if texts and isinstance(features, str):
            values = self.main.adata.obs[features]
        
            if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype.name == 'category':
                cluster_coords = self.main.adata.obsm['X_umap']
                values = self.main.adata.obs[features]
                unique_clusters = values.unique()
                for cluster in unique_clusters:
                    mask = values == cluster
                    centroid_x = cluster_coords[mask, 0].mean()
                    centroid_y = cluster_coords[mask, 1].mean()
            
                    plt.text(centroid_x, centroid_y, str(cluster), color='black',
                             fontsize=10, ha='center', va='center', weight='bold')
                
        self.current_ax = ax
        if save:
            self.save(f"{features}_UMAP")
        return ax
    
    
    def cor(self, gene, number_of_genes=10,ax=None,figsize=(8,8),
            save=False,color="black", color_genes="red",size=15,text=True,cmap="copper"):
        
        if f"cor_{gene}" in self.main.adata.var:
            df = pd.DataFrame({"r":self.main.adata.var[f"cor_{gene}"],
                               "expression_mean":self.main.adata.var[f"exp_{gene}"],
                               "qval":self.main.adata.var[f"cor_qval_{gene}"],
                               "gene":self.main.adata.var_names})
        else:
            df = self.main.cor(gene,inplace=True)
            df.rename(columns={f"exp_{gene}":"expression_mean"},inplace=True)
            df.rename(columns={f"cor_qval_{gene}":"qval"},inplace=True)
            
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)    
        
        df["expression_mean_log10"] = np.log10(df["expression_mean"])
        df["qval_log10"] = -np.log(df["qval"] + df["qval"][df["qval"]>0].min())
        df.index = df["gene"].values
        df = df.dropna(subset=["expression_mean_log10", "r", "qval_log10"])
        cor_series_clean = df["r"]
        top_abs_indices = cor_series_clean.abs().nlargest(number_of_genes).index

        # Retrieve the original correlations (with their sign) in the order of their absolute value
        top_cor = cor_series_clean.loc[top_abs_indices]
        # top_cor = cor_series_clean.nlargest(number_of_genes)
        # if top_cor.iloc[0] > 0.9:
        #     df.loc[top_cor.index[0],"r"] = np.nan
        
        top_genes = list(top_cor.index)

        ax = plot_scatter_signif(df, "expression_mean_log10", "r",genes=top_genes,
                            title=gene,text=text,color="qval_log10",ax=ax,
                            xlab="log10(mean expression)",size=size,cmap=cmap,
                            ylab="Spearman correlation",legend=True,color_genes="black")
        print(df.loc[df["gene"].isin(top_genes),["r","expression_mean","qval"]].sort_values(by="r", ascending=False))
        self.current_ax = ax
        if save:
            self.save(f"{gene}_COR")
        return ax 
        
    
    def __repr__(self):
        s = f"Plots available for [{self.main.name}].sc:\n\tsave(), spatial(), hist(), cells(), umap(), cor()"
        return s


def save_fig(path, fig, open_file=False, format_='png', dpi=300): 
    '''Save a fig object'''
    if isinstance(fig, pd.DataFrame):
        path = path.replace(f".{format_}",".csv")
        fig.to_csv(path)
        return path
    fig.savefig(path, format=format_, dpi=dpi, bbox_inches='tight')
    if open_file:
        os.startfile(path)
    return path

def plot_scatter(x, y, values, title=None, size=1, legend=True, xlab=None, ylab=None, 
                   cmap='winter', figsize=(8, 8), alpha=1, legend_title=None, ax=None,marker='s'):
    '''
    Plots a scatterplot based on coordinates and values.
    parameters:
        * x, y, values - coordinates and values to plot. lists, Series, or arrays
        * cmap - can be string (name of pellate), list of colors, 
                 or in categorical values case, a dict {"value":"color"}
        * marker - shape of dots. "s" for square, "o" or "." for circle
        * figsize, size, legend, xlab, ylab, title, legend_title - cosmetic parameters
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set_aspect('equal')
    if legend_title is None:
        legend_title = title

    # Ensure x, y, and values are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)
    
    if np.issubdtype(values.dtype, np.number): # Numeric case: Use colorbar
        if isinstance(cmap, str):
            cmap_obj = colormaps.get_cmap(cmap)
        elif isinstance(cmap, list):
            cmap_obj = LinearSegmentedColormap.from_list("custom_cmap", cmap)
        scatter = ax.scatter(x, y, c=values, cmap=cmap_obj, marker=marker,
                              alpha=alpha, s=size,edgecolor='none')
        if legend:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label(legend_title)
    else: # Categorical case: Use legend 
        unique_values = np.unique(values.astype(str))
        unique_values = unique_values[unique_values != 'nan']
        if isinstance(cmap, (str,list)):
            colors = get_colors(unique_values, cmap)
            color_map = {val: colors[i] for i, val in enumerate(unique_values)}  
        elif isinstance(cmap, dict):
            color_map = {val: cmap.get(val,DEFAULT_COLOR) for val in unique_values}
        else:
            raise ValueError("cmap must be a string (colormap name) or a dictionary")
        for val in unique_values: # Plot each category with its color
            if values.dtype == bool:
                values = values.astype(str)
            mask = values == val
            ax.scatter(x[mask], y[mask], color=color_map[val], edgecolor='none',
                        label=str(val), marker=marker, alpha=alpha, s=size)
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

    
def plot_scatter_signif(df, x_col, y_col,
                        genes=None, genes2=None,  # genes for group1 and group2
                        text=True, figsize=(8,8), size=10, legend=False, title=None,
                        ax=None, xlab=None, ylab=None, out_path=None,
                        color="black", color_genes="red", color_genes2="blue",
                        x_line=None, y_line=None,cmap="viridis",repel=False):
    '''
    Plots a scatterplot based on a dataframe.
    
    Parameters:
        df: pd.DataFrame
        x_col, y_col: names of the columns in df to plot
        genes: list of gene names to highlight as group 1
        genes2: list of gene names to highlight as group 2 (optional)
        text: bool, whether to annotate gene names on the plot
        figsize: tuple, figure size
        size: marker size
        legend: bool, whether to include legend
        title: str, plot title
        ax: matplotlib Axes, if provided
        xlab, ylab: labels for the x and y axes
        out_path: str, path to save the figure (if None, the plot is not saved)
        color: color for background points
        color_genes: color for genes in group 1
        color_genes2: color for genes in group 2
        x_line, y_line: numbers to add vertical and horizontal reference lines
    '''
    
    # Create an axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        
    # Create a column to mark gene groups. (Empty string means "not selected".)
    df["group"] = ""
    
    # Ensure there is a column named "gene" to work with.
    if "gene" not in df.columns:
        df["gene"] = df.index.values
    
    # Mark genes for group 1 if provided.
    if genes is True:
        df["distance"] = np.sqrt(df[x_col]**2 + df[y_col]**2)
        top_genes = df.nlargest(100, "distance")["gene"].tolist()
        df.loc[df["gene"].isin(top_genes), "group"] = "group1"
    elif genes:
        df.loc[df["gene"].isin(genes), "group"] = "group1"
    
    # Mark genes for group 2 if provided.
    if genes2:
        df.loc[df["gene"].isin(genes2), "group"] = "group2"
    
    # Plot background points (those not in any group)
    if color in df.columns:
        ax = sns.scatterplot(data=df, x=x_col, y=y_col,palette=cmap,
                        s=size, legend=legend,ax=ax, hue=color, edgecolor=None)
    else:
        ax = sns.scatterplot(data=df[df["group"] == ""], x=x_col, y=y_col,
                        s=size, legend=legend,ax=ax, color=color, edgecolor=None)
    
    # Add reference lines if specified.
    if y_line is not None:
        ax.axhline(y=y_line if y_line is not True else 0,
                   color="k", linestyle="--")
    if x_line is not None:
        ax.axvline(x=x_line if x_line is not True else 0,
                   color="k", linestyle="--")
    
    # Prepare a list to collect text objects so that they can be adjusted together.
    texts = []
    # Plot group 1 points and (optionally) add text labels.
    group1_df = df[df["group"] == "group1"]
    if not group1_df.empty:
        if not color in df.columns:
            sns.scatterplot(data=group1_df,
                            x=x_col, y=y_col,
                            color=color_genes,
                            s=size, legend=False, ax=ax, edgecolor="k")
        if text:
            for _, row in group1_df.iterrows():
                texts.append(ax.text(row[x_col], row[y_col], row["gene"],
                                     color=color_genes, fontsize=14))
    
    # Plot group 2 points and (optionally) add text labels.
    group2_df = df[df["group"] == "group2"]
    if not group2_df.empty:
        sns.scatterplot(data=group2_df,
                        x=x_col, y=y_col,
                        color=color_genes2,
                        s=size, legend=False, ax=ax, edgecolor="k")
        if text:
            for _, row in group2_df.iterrows():
                texts.append(ax.text(row[x_col], row[y_col], row["gene"],
                                     color=color_genes2, fontsize=14))
    if repel:
    # Adjust text to reduce overlap if any text labels were added.
        if text and texts:
            adjust_text(texts,
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                        force_text=(0.1, 0.1),   # Instead of (3, 3)
                        ax=ax)
    
    # Set labels and title
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_title(title)
    
    # Save the figure if an output path is provided.
    if out_path:
        if not out_path.endswith(".png"):
            out_path += ".png"
        plt.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    del df["group"]
    
    return ax


def plot_MA(df, qval_thresh=0.25, exp_thresh=0, fc_thresh=0 ,figsize=(8,8), ax=None, title=None,
            size=10, colname_exp="expression_mean",colname_qval="qval", 
            colname_fc="log2fc", n_texts=130, ylab="log2(ratio)",repel=False):
    '''
    Plots a MA plot of the output of ViziumHD.dge().
    parameters:
        * exp_thresh - show only genes with expression higher than this value
        * qval_thresh, fc_thresh - values above/below which consider a pojnt as significant
        * size, ylab, title, figsize - cosmetic parameters
        * colname_exp - can be "expression_mean","expression_min","expression_max"
        * n_texts - maximal number of texts to display. above, will only color the dots
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    plot = df.loc[df[colname_exp] >= exp_thresh].copy()
    plot["exp"] = np.log10(plot[colname_exp])
    plot["signif"] = (plot[colname_qval] <= qval_thresh) & (abs(plot[colname_fc]) >= fc_thresh)
    if not "gene" in plot.columns:
        plot["gene"] = plot.index
    signif_genes = plot.loc[plot["signif"]==True,"gene"].tolist()
    text = True if len(signif_genes) < n_texts else False
    ax = plot_scatter_signif(plot, "exp", colname_fc, genes=signif_genes,
                             text=text, title=title,ax=ax,size=size,repel=repel,
                             xlab=f"log10({colname_exp.replace('_',' ')})",
                             ylab=ylab,y_line=0,color_genes="red", color="gray")
    return ax


def plot_scatter_html(df,x,y,save_path=None,text="gene",color="black",size=1,
                      xlab=None,ylab=None,title=None,legend_title=None):
    '''
    Creates plotly express interactive scatterplot.
    parameters:
        * df, x, y - data, x axis column name, y axis column name
        * color - color spots by a column in the df
        * size - change size of dots by a column in the df
        * open_fig - open the file with default machine software?
        * save_path - path of html file, where to save the plot.
        * xlab, ylab, title, legend_title - cosmetic parameters
    '''
    def open_html(html_file,chrome_path=chrome_path):
        process = Popen(['cmd.exe', '/c', chrome_path, html_file], stdout=PIPE, stderr=PIPE)
    
    plot_kwargs = {"x": x,"y": y,"hover_data": [text],"labels": {}}

    # Handle color (categorical vs fixed color)
    if color in df.columns:
        plot_kwargs["color"] = color
        plot_kwargs["labels"][color] = legend_title if legend_title else color
        plot_kwargs["hover_data"].append(color)
    elif isinstance(color, str):  # If it's a fixed color
        plot_kwargs["color_discrete_sequence"] = [color]
    else:
        plot_kwargs["color_discrete_sequence"] = ["black"] 

    if size in df.columns:
        plot_kwargs["size"] = size
        plot_kwargs["labels"][size] = legend_title if legend_title else size
        plot_kwargs["hover_data"].append(size)
    else:
        plot_kwargs["size_max"] = 10  # Default size for points
        
    fig = px.scatter(df, **plot_kwargs)    
    fig.update_traces(marker_size=10,hoverinfo='text+x+y',mode='markers+text')

    fig.update_layout(template="simple_white",
        title=title,xaxis_title=xlab,yaxis_title=ylab,
        title_font=dict(size=30, family="Arial", color="Black"),
        xaxis_title_font=dict(size=24, family="Arial", color="Black"),
        yaxis_title_font=dict(size=24, family="Arial", color="Black"))
    

    if save_path is None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        fig.write_html(tmp_path)
        open_html(tmp_path)
        time.sleep(2)  # Allow some time for the browser to load the file
        os.remove(tmp_path)  # Delete the temporary file
    else:
        fig.write_html(save_path) 
 

def plot_histogram(values, bins=10, show_zeroes=False, xlim=None, title=None, figsize=(8,8), 
              cmap=None, color="blue", ylab="Count",xlab=None,ax=None):
    '''
    Plots histogram from numeric values or barplot for categorical values.
    Parameters:
        * values: pd.Series
        * show_zeroes - include count of zeroes in numerical case?
        * cmap - used for categorical case. has higher priority than color.
        * xlim, figsize, ylab, xlab - cosmetic parameters
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    if np.issubdtype(values.dtype, np.number):
        if not show_zeroes:
            values = values[values > 0]
        counts, edges, patches = ax.hist(values,bins=bins,color=None if cmap else color)
        
        if cmap is not None:
            colormap = plt.cm.get_cmap(cmap, len(patches))  # Generate enough colors for all bins
            for i, patch in enumerate(patches):
                patch.set_facecolor(colormap(i))
                
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


def plot_pie(series, figsize=(4,4),title=None,ax=None,cmap="Set1",capitalize=True):
    '''plots a piechart of pd.Series'''
    from matplotlib.patches import Circle

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        
    counts = series.value_counts()
    categories = counts.index
    if capitalize:
        categories = categories.str.capitalize()
    values = counts.values

    if isinstance(cmap, dict):
        if capitalize:
            cmap = {k.capitalize(): v for k, v in cmap.items()}
        colors = [cmap[category] for category in categories]
    else:
        colors = get_colors(values,cmap=cmap)

    wedges, texts = ax.pie(values,
        labels=values,          # Display counts as labels
        labeldistance=1.05,      # Position labels outside the pie
        startangle=90,          # Rotate pie chart for better orientation
        colors=colors)
    circle = Circle((0, 0), 0.5, color='white', zorder=2) 
    ax.add_artist(circle)
    handles = []
    for w, category in zip(wedges, categories):
        facecolor = w.get_facecolor()
        handles.append(plt.Line2D([0], [0],marker='o',color=facecolor,
                label=category, markersize=15,linestyle='None'))

    legend = ax.legend(handles=handles,title=title,loc='center',
        bbox_to_anchor=(0.5, 0.5),bbox_transform=ax.transAxes,frameon=False)
    legend.set_zorder(3)

    ax.axis('equal')  # Ensure pie chart is a circle
    if title is not None:
        ax.set_title(title)
    return ax


def get_colors(values, cmap):
    '''return a list of colors, in the length of the unique values, based on cmap'''
    if isinstance(values, pd.core.series.Series):
        unique_values = values.unique()
    else:
        unique_values = np.unique(values.astype(str))
    if isinstance(cmap, str):
        cmap_obj = colormaps.get_cmap(cmap)
    elif isinstance(cmap, list):
        cmap_obj = LinearSegmentedColormap.from_list("custom_cmap", cmap)
    cmap_len = cmap_obj.N
    num_unique = len(unique_values)
    if num_unique == 1:
        # Assign a single color (e.g., middle of the colormap)
        colors = [cmap_obj(0.5)]
    elif num_unique <= cmap_len:
        # Map each unique value to a unique color in the colormap
        colors = [cmap_obj(i / (num_unique - 1)) for i in range(num_unique)]
    else:
        # If there are more unique values than colors in the colormap, cycle through the colormap
        colors = [cmap_obj(i % cmap_len / (cmap_len - 1)) for i in range(num_unique)]
    return colors


def set_axis_ticks(ax, length_in_pixels, adjusted_microns_per_pixel, axis='x', num_ticks_desired=6):
    '''sets ticks and ticklabels at round numbers'''
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


def _plot_squares_exact(x, y, values, title=None, size=1, legend=True, xlab=None, ylab=None, 
                 cmap='winter', figsize=(8, 8), alpha=1, legend_title=None, ax=None):
    '''
    Plots sqares in the exact size
    parameters:
        * cmap - str, name of colormap, or list of colors. if categorical, can also be a dict {"val":"color"}
        * title, legend, ylab, xlab, figsize, alpha, legend_title - cosmetic parameters
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    if legend_title is None:
        legend_title = title

    # Ensure x, y, and values are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)

    # Set the aspect ratio to 'equal' to ensure squares remain squares
    ax.set_aspect('equal')

    if np.issubdtype(values.dtype, np.number):  # Numeric case: Use colorbar
        # Normalize the values for the colormap
        if isinstance(cmap, str):
            cmap_obj = colormaps.get_cmap(cmap)
        elif isinstance(cmap, list):
            cmap_obj = LinearSegmentedColormap.from_list("custom_cmap", cmap)
        norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))

        # Add rectangles for each data point
        for xi, yi, vi in zip(x, y, values):
            # Calculate the lower-left corner position to center the square at (xi, yi)
            ll_corner_x = xi - size / 2
            ll_corner_y = yi - size / 2

            # Create a rectangle (square) centered at (xi, yi)
            square = patches.Rectangle(
                (ll_corner_x, ll_corner_y),   # (x, y) of lower-left corner
                size,size,    # Width, Height in data units
                facecolor=cmap_obj(norm(vi)), edgecolor='none',alpha=alpha)
            ax.add_patch(square)

        # Create a ScalarMappable for the colorbar
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        if legend:
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label(legend_title)

    else:  # Categorical case: Use legend
        unique_values = np.unique(values.astype(str))
        unique_values = unique_values[unique_values != 'nan']
        if isinstance(cmap, (str,list)):
            colors = get_colors(unique_values, cmap)
            color_map = {val: colors[i] for i, val in enumerate(unique_values)}  
        elif isinstance(cmap, dict):
            color_map = {val: cmap.get(val, 'gray') for val in unique_values}
        else:
            raise ValueError("cmap must be a string (colormap name) or a dictionary")

        # Add rectangles for each category
        for val in unique_values:
            mask = values == val
            xi = x[mask]
            yi = y[mask]
            color = color_map[val]
            for xj, yj in zip(xi, yi):
                ll_corner_x = xj - size / 2
                ll_corner_y = yj - size / 2
                square = patches.Rectangle(
                    (ll_corner_x, ll_corner_y),size,size,facecolor=color,
                    edgecolor='none',alpha=alpha)
                ax.add_patch(square)

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


def plot_heatmap(heatmap_data, x_y_val=None, normilize=False, sort=True, 
                 sort_method="sum",ax=None, xlab=None, ylab=None, title=None,
                 cmap="coolwarm",figsize=(8,16),legend=True, legend_title=None):
    '''
    Plots a heatmap.
    parameters:
        * heatmap_data - either a "heatmap ready" df, where genes are index,
                                or three columns, of category(x), gene (y), value
        * x_y_val (list) - if the heatmap_data is three columns, specify. 
                                category(x), gene (y), value
        * normilize - whether to normilize each row to the maximal value of the row
        * sort - sort the rows?
        * sort_method - if sort is True, how to sort? possible values are "sum","std","mean"
        * ax - matplotlib Axes, if provided
        * figsize, cmap, legend, xlab, ylab, title, legend_title - cosmetic parameters
    '''
    if x_y_val:
        heatmap_data = heatmap_data.pivot(index=x_y_val[1], columns=x_y_val[0], values=x_y_val[2])
    if normilize:
        heatmap_data = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    if sort:
        if sort_method == "sum":
            heatmap_data["delta"] = heatmap_data.sum(axis=1, skipna=True)
        elif sort_method == "mean":
            heatmap_data["delta"] = heatmap_data.mean(axis=1, skipna=True)
        elif sort_method == "std":
            heatmap_data["delta"] = heatmap_data.std(axis=1, skipna=True)
        else:
            raise ValueError(f"Invalid sort_method: {sort_method}. "
                             "Choose from ['sum','mean','std']")
        heatmap_data = heatmap_data.sort_values(by="delta", ascending=False)
        del heatmap_data["delta"]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(heatmap_data, aspect='auto',cmap=cmap)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)
    
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)
    if legend:
        cbar = ax.figure.colorbar(img, ax=ax)    
        if legend_title:
            cbar.set_label(legend_title)
    
    return ax


def plot_dotplot(df, x, y, size_col, val_col,
                 normalize_size=False, normalize_col=False, sort=True, sort_method="sum",
                 ax=None, xlab=None, ylab=None, title=None,max_dot_size=100, 
                 cmap="coolwarm", figsize=(8,16),legend=True, rotate_xticklab=False,
                 legend_col_title=None, legend_size_title=None):
    '''
    Plots a dotplot.
    parameters:
        * df - dataframe of 4 columns, x, y, size_col, val_col
        * x, y, size_col, val_col - which columns to use
        * normalize_size,normalize_col - whether to normilize each row to the maximal value of the row
        * sort - sort the rows?
        * sort_method - if sort is True, how to sort? possible values are "sum","std","mean"
        * ax - matplotlib Axes, if provided
        * figsize, cmap, legend, xlab, ylab, title, legend_col_title, 
        legend_size_title, rotate_xticklab - cosmetic parameters
    '''
    color_data = df.pivot(index=y, columns=x, values=val_col)
    size_data  = df.pivot(index=y, columns=x, values=size_col)
    
    if normalize_col:
        color_data = color_data.div(color_data.max(axis=1), axis=0)
    if normalize_size:
        size_data  = size_data.div(size_data.max(axis=1), axis=0)

    if sort:
        if sort_method == "sum":
            color_data["delta"] = color_data.sum(axis=1, skipna=True)
        elif sort_method == "mean":
            color_data["delta"] = color_data.mean(axis=1, skipna=True)
        elif sort_method == "std":
            color_data["delta"] = color_data.std(axis=1, skipna=True)
        else:
            raise ValueError(f"Invalid sort_method: {sort_method}. "
                             "Choose from ['sum','mean','std']")
        color_data = color_data.sort_values("delta", ascending=False)
        size_data  = size_data.loc[color_data.index, :]
        del color_data["delta"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    row_labels, col_labels  = list(color_data.index), list(color_data.columns)

    xvals, yvals, colors, sizes = [], [], [], []
    for i, row_name in enumerate(row_labels):
        for j, col_name in enumerate(col_labels):
            xvals.append(j)
            yvals.append(i)
            c_val = color_data.loc[row_name, col_name]
            colors.append(c_val)
            s_val = size_data.loc[row_name, col_name]
            sizes.append(np.nan_to_num(s_val, nan=0))

    all_sizes_arr = np.array(sizes)
    current_max = np.nanmax(all_sizes_arr)
    if current_max > 0:
        sizes_normed = all_sizes_arr / current_max
    else:
        sizes_normed = all_sizes_arr  # if all zero/NaN

    # scale up to user-requested maximum size
    scatter_sizes = [max_dot_size * s for s in sizes_normed]

    if isinstance(cmap, list):
        cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap)

    sc = ax.scatter(xvals, yvals,c=colors,
        s=scatter_sizes,cmap=cmap,edgecolors="none")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90 if rotate_xticklab else 0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)

    if legend:
        # Shrink main axis to free space on the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])

        # Create a new Axes in top half for colorbar
        cbar_ax = fig.add_axes([box.x0 + box.width*0.85, box.y0 + box.height*0.5, 
            0.03, box.height*0.45])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        if legend_col_title:
            cbar.set_label(legend_col_title)

        if current_max > 0:
            # Example fractions of original data's range
            fraction_values = [0.25, 0.50, 0.75, 1.00]
            # Convert fraction -> actual data scale
            actual_sizes = [fraction * current_max for fraction in fraction_values]
            # Human-readable labels
            size_labels = [f"{v:.2g}" for v in actual_sizes]
            
            # Convert fraction -> scatter circle area
            size_legend_scaled = [max_dot_size * f for f in fraction_values]

            # Make dummy scatter patches to display in the legend
            legend_patches = [plt.scatter([], [], s=s, color="gray",alpha=0.8) 
                              for s in size_legend_scaled]
            # Place them in the bottom half
            ax.legend(legend_patches,size_labels,title=legend_size_title,loc="upper left",
                bbox_to_anchor=(1.02, 0.45),frameon=True,
                labelspacing=2,handletextpad=1.5 ,borderpad=1.5)

    return ax
