# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:29:05 2024

@author: royno
"""

import os
import numpy as np
import pandas as pd
# import geopandas as gpd
# import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colormaps
import seaborn as sns
from adjustText import adjust_text
import plotly.express as px
from subprocess import Popen, PIPE
# from matplotlib import colormaps


POINTS_PER_INCH = 72
MAX_BARS = 30 # in barplot
PAD_CONSTANT = 0.3 # padding of squares in scatterplot
DEFAULT_COLOR ='None' # for plotting categorical
chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'

class PlotVizium:
    def __init__(self, vizium_instance):
        self.main = vizium_instance
        self.current_ax = None
    
    def save(self, filename:str, fig=None, ax=None, open_file=False, format='png', dpi=300):
        '''
        saves a figure or ax. 
        parameters:
            * filename - name of plot
            * fig (optional) - plt.Figure object to save.
            * ax - ax to save. if not passed, will use self.current_ax
            * open_file - open the file?
            * format - format of file
        '''
        path = f"{self.main.path_output}/{self.main.name}_{filename}.{format}"
        if isinstance(fig, pd.DataFrame):
            fig.to_csv(path.replace(".png",".csv"))
            return path
        if fig is None:
            if ax is None:
                if self.current_ax is None:
                    raise ValueError(f"No ax present in {self.main.name}")
                ax = self.current_ax
            fig = ax.get_figure()
        
        fig.savefig(path, format=format, dpi=dpi, bbox_inches='tight')
        if open_file:
            os.startfile(path)
        return path
    
    
    def __get_dot_size(self, adjusted_microns_per_pixel:float):
        '''gets the size of spots, depending on adjusted_microns_per_pixel'''
        bin_size_pixels = self.main.json['bin_size_um'] / adjusted_microns_per_pixel 
        dpi = plt.gcf().get_dpi()
        # dpi = mpl.rcParams['figure.dpi']
        points_per_pixels = POINTS_PER_INCH / dpi
        dot_size = bin_size_pixels * points_per_pixels 
        return dot_size
    
    def spatial(self, what=None, image=True, ax=None, title=None, cmap="viridis", 
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
            
        xlim, ylim, adjusted_microns_per_pixel = self.main.crop(xlim, ylim)
        size = self.__get_dot_size(adjusted_microns_per_pixel)
        if pad:
            size *= PAD_CONSTANT
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)


        if image: # Plot image
            # ax.imshow(self.image_cropped, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])
            ax.imshow(self.main.image_cropped)

        if what: 
            values = self.main.get(what, cropped=True)
            if np.issubdtype(values.dtype, np.number):  # Filter values that are 0
                mask = values > 0
            else:
                mask = [True for _ in values]   # No need for filtering
            values = values[mask]
            x = self.main.pixel_x[mask]
            y = self.main.pixel_y[mask]
            
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
            
        height, width = self.main.image_cropped.shape[:2]  
        set_axis_ticks(ax, width, adjusted_microns_per_pixel, axis='x')
        set_axis_ticks(ax, height, adjusted_microns_per_pixel, axis='y')    
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)     
        
        # Save figure:
        self.current_ax = ax
        if save:
            self.save(f"{(what + '_') if what else ''}SPATIAL")
        return ax
    
    def hist(self, what, bins=20, xlim=None, title=None, ylab=None,xlab=None,ax=None,
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
        self.main.crop() # resets adata_cropped to full image
        to_plot = pd.Series(self.main.get(what, cropped=True))
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = plot_histogram(to_plot,bins=bins,xlim=xlim,title=title,figsize=figsize,
                       cmap=cmap,color=color,ylab=ylab,xlab=xlab,ax=ax)            
        self.current_ax = ax
        if save:
            self.save(f"{what}_HIST")
        return ax
    
    def __call__(self):
        pass
    
    def __repr__(self):
        s = f"Plots available for [{self.main.name}]:\n\tsave(), spatial(), hist()"
        if self.main.SC:
            s += "\n\nand for sc:\n\tsave(), spatial(), hist(), cells(), umpa()"
        return s
    
    


class PlotSC:
    def __init__(self, sc_instance):
        self.main = sc_instance
        self.current_ax = None
    
    def save(self):
        pass
    
    def spatial(self):
        pass

    def hist(self):
        pass
    
    def cells(self):
        pass
    
    def umap(self):
        pass
    
    def __repr__(self):
        s = f"Plots available for [{self.main.name}].sc:\n\tsave(), spatial(), hist(), cells(), umpa()"
        return s


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
    
def plot_scatter_signif(df,x_col,y_col,genes=None,text=True,figsize=(8,8),size=10,legend=False,
                    ax=None,xlab=None,ylab=None,out_path=None,color="blue",color_genes="red",x_line=None,y_line=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    df['type'] = ""
    
    sns.scatterplot(data=df[df['type'] == ''], x=x_col, y=y_col,s=size,legend=legend,
                    ax=ax,color=color,edgecolor=None)
    if y_line is not None:
        ax.axhline(y=y_line if y_line is not True else 0,color="k",linestyle="--")
    if x_line is not None:
        ax.axvline(x=x_line if x_line is not True else 0,color="k",linestyle="--")
    if genes:
        df.loc[df['gene'].isin(genes),"type"] = "selected"
        subplot = df[df['type'] != '']
        if not subplot.empty:
            sns.scatterplot(data=subplot, x=x_col, y=y_col,color=color_genes,
                            s=size,legend=False,ax=ax,edgecolor="k")
            if text:
                texts = [ax.text(
                    subplot[x_col].iloc[i], 
                    subplot[y_col].iloc[i], 
                    subplot['gene'].iloc[i],
                    color=color_genes,
                    fontsize=14,
                    ) for i in range(len(subplot))]
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                            force_text=(0.6, 0.6),ax=ax)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    if out_path:
        if not out_path.endswith(".png"):
            out_path += ".png"
        plt.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    return ax

def plot_scatter_html(df,x,y,save_path,text="gene",color=None,size=None,xlab=None,ylab=None,title=None,open_fig=True,legend_title=None):

    def open_html(html_file,chrome_path=chrome_path):
        process = Popen(['cmd.exe', '/c', chrome_path, html_file], stdout=PIPE, stderr=PIPE)

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


def plot_pie(series, figsize=(4,4),title=None,ax=None,cmap="Set1",capitalize=True):
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
        handles.append(
            plt.Line2D([0], [0],
                marker='o',
                color=facecolor,
                label=category,
                markersize=15,
                linestyle='None'))

    legend = ax.legend(
        handles=handles,
        title=title,
        loc='center',
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=ax.transAxes,
        frameon=False)
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










