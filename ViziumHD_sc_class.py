# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:57:45 2024

@author: royno
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import anndata as ad

import ViziumHD_utils
import ViziumHD_class
import ViziumHD_plot
import SingleCell_utils


def _count_apical(series):
        return (series == 'apical').sum()

def _count_basal(series):
    return (series == 'basal').sum()


class SingleCell:
    def __init__(self, vizium_instance, input_df, input_type="Ofra_int", columns=None):
        '''
        input_df (str or df) - pd.DataFrame or path to csv file produced with the Groovy pipeline. 
                                if it's an anndata, it will skip initialization, and just store the anndata.
        columns (list) - which columns from the CSV to add to the metadata (aggregate from spots)?
                         if None, will use all columns of the CSV.
                         example:

        '''
        self.viz = vizium_instance
        self.path_output = self.viz.path_output + "/single_cell"
        # self.plot = ViziumHD_plot.plotSC(self)
        if isinstance(input_df, str):
            print("[Reading CSV]")
            input_df = pd.read_csv(input_df, sep="\t")
        

        if isinstance(input_df, ad._core.anndata.AnnData): 
            self.adata = input_df
        else:
            if columns is None: 
                columns = ['Object ID']
#  ['Object ID','cell_y_um','cell_x_um','area_nuc_um2','number_spots_nuc','number_spots_nuc','area_cell_um2']
            self.adata = SingleCell_utils._aggregate_spots(self.viz.adata,
                                                       input_df, columns,
                                                       custom_agg={"apicome":[_count_apical, _count_basal]}) 

    def __init_img(self):
        pass
     
       
    def sc_transfer_meta(self, what:str):
        '''transfers metadata assignment from the single-cell to the spots'''
        pass
    
    def crop(self):
        pass
        
    
    def export_h5(self, path=None):
        if not path:
            path = f"{self.path_output}/{self.viz.name}_viziumHD_cells.h5ad"
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
    
    def update(self):
        '''updates the methods in the instance'''
        ViziumHD_utils.update_instance_methods(self)
        ViziumHD_utils.update_instance_methods(self.plot)
        self.__init_img()
    
    def head(self, n=5):
        return self.adata.obs.head(n) 
