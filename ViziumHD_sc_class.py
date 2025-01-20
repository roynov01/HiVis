# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:57:45 2024

@author: royno
"""

import numpy as np
import pandas as pd
import anndata as ad
import os
import scipy.io
from copy import deepcopy
import scanpy as sc
import geopandas as gpd
import warnings
import re


import ViziumHD_utils
# import ViziumHD_class
import ViziumHD_plot
import SingleCell_utils


# def _count_apical(series):
#         return (series == 'apical').sum()

# def _count_basal(series):
#     return (series == 'basal').sum()


# def new_from_segmentation(vizium_instance, input_df, columns=None, custom_agg=None, sep="\t"):
#     '''
#     vizium_instance - ViziumHD object
#     input_df (str or df or anndata) - single cell metadata, should include columns: ['Object ID','Name','in_nucleus','in_cell']
#             either pd.DataFrame or str, path to csv file produced with the Groovy pipeline. 
#                             if it's an anndata, it will skip initialization, and just store the anndata.
#     columns (list) - which columns from the CSV to add to the metadata (aggregate from spots)?
#                      example: ['cell_y_um','cell_x_um','area_nuc_um2','number_spots_nuc','number_spots_nuc','area_cell_um2']
#     custom_agg (dict) - {str:func} or {str:[func,func]}. Used for metadata aggregation.
#                         example {'apicome':[_count_apical, _count_basal]}
#     sep (str) - passed to read_csv if input_df is a path of CSV
#     '''
#     if isinstance(input_df, str):
#         ViziumHD_utils.validate_exists(input_df)
#         print("[Reading CSV]")
#         input_df = pd.read_csv(input_df, sep=sep)
#     if columns is None: 
#         columns = []
#     for col in ["Object ID","pxl_row_in_fullres", "pxl_col_in_fullres", "pxl_col_in_lowres", "pxl_row_in_lowres",
#                 "pxl_col_in_highres", "pxl_row_in_highres", "um_x", "um_y", "nUMI"]:
#         if col not in columns:
#             columns += [col]
#     adata_sc = SingleCell_utils._aggregate_spots_cells(vizium_instance.adata,
#                                                input_df, columns,
#                                                custom_agg=custom_agg) 
#     adata_sc.obs.rename(columns={"nUMI":"nUMI_avg"})
#     adata_sc.obs["nUMI"] = adata_sc.obs["nUMI_avg"] * adata_sc.obs["spot_count"]
#     return SingleCell(vizium_instance, adata_sc)

# def new_from_annotations(vizium_instance, group_col, columns=None, custom_agg=None):
#     '''
#     vizium_instance - ViziumHD object
#     input_df (str or df or anndata) - single cell metadata, should include columns: ['Object ID','Name','in_nucleus','in_cell']
#             either pd.DataFrame or str, path to csv file produced with the Groovy pipeline. 
#                             if it's an anndata, it will skip initialization, and just store the anndata.
#     columns (list) - which columns from the CSV to add to the metadata (aggregate from spots)?
#                      example: ['cell_y_um','cell_x_um','area_nuc_um2','number_spots_nuc','number_spots_nuc','area_cell_um2']
#     custom_agg (dict) - {str:func} or {str:[func,func]}. Used for metadata aggregation.
#                         example {'apicome':[_count_apical, _count_basal]}
#     sep (str) - passed to read_csv if input_df is a path of CSV
#     '''
#     if columns is None: 
#         columns = []
#     for col in ["pxl_row_in_fullres", "pxl_col_in_fullres", "pxl_col_in_lowres", "pxl_row_in_lowres",
#                 "pxl_col_in_highres", "pxl_row_in_highres", "um_x", "um_y", "nUMI"]:
#         if col not in columns:
#             columns += [col]
#     adata_sc = SingleCell_utils._aggregate_spots_annotations(vizium_instance.adata,
#                                                    group_col, columns,
#                                                    custom_agg=custom_agg) 
#     adata_sc.obs.rename(columns={"nUMI":"nUMI_avg"},inplace=True)
#     adata_sc.obs["nUMI"] = adata_sc.obs["nUMI_avg"] * adata_sc.obs["spot_count"]
#     return SingleCell(vizium_instance, adata_sc)

class SingleCell:
    def __init__(self, vizium_instance, adata_sc, geojson_cells_path=None):
        '''
        vizium_instance - ViziumHD object
        adata_sc - anndata of single cells
        geojson_path - path of geojson, exported cells
        '''
        if not isinstance(adata_sc, ad._core.anndata.AnnData): 
            raise ValueError("Adata must be Anndata object")
        self.adata = adata_sc
        self.viz = vizium_instance
        self.path_output = self.viz.path_output + "/single_cell"
        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        self.plot = ViziumHD_plot.PlotSC(self)
        self.adata_cropped = None
        if geojson_cells_path:
            self.import_geometry(geojson_cells_path,object_type="cell")


    def import_geometry(self, geojson_path, object_type="cell"):
        gdf = gpd.read_file(geojson_path)
        gdf = gdf[gdf["objectType"] == object_type]
        gdf = gdf.loc[:,["id","geometry"]]
        gdf.rename(columns={"id":self.adata.obs.index.name},inplace=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry column does not contain geometry")
            gdf["geometry"] = gdf["geometry"].astype(str)
        gdf = gdf.set_index(self.adata.obs.index.name)
        self.adata.obs = self.adata.obs.join(gdf,how="left")
        
    
    def merge(self, adata, obs=None, var=None, umap=True, pca=True, hvg=True):
        if not obs:
            obs = []
        elif isinstance(obs, str):
            obs = [obs]
        if umap and "X_umap" in adata.obsm:
            self.adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
            umap_coords = adata.obsm['X_umap']
            adata.obs['UMAP_1'] = umap_coords[:, 0]
            adata.obs['UMAP_2'] = umap_coords[:, 1]
            
            obs += ['UMAP_1','UMAP_2']
        if pca and "X_pca" in adata.obsm:
            self.adata.obsm['X_pca'] = adata.obsm['X_pca'].copy()
        if hvg and 'highly_variable' in adata.var.columns:
            if not var:
                var = 'highly_variable'
            else:
                if 'highly_variable' not in var:
                    var += ['highly_variable']
        if obs:
            
            existing_columns = [col for col in obs if col in self.adata.obs.columns]
            if existing_columns:
                self.adata.obs.drop(columns=existing_columns, inplace=True)
            self.adata.obs = self.adata.obs.join(adata.obs[obs])
        if var:
            if isinstance(var, str):
                var = [var]
            existing_columns = [col for col in var if col in self.adata.var.columns]
            if existing_columns:
                self.adata.var.drop(columns=existing_columns, inplace=True)
            self.adata.var = self.adata.var.join(adata.var[var])
            
                
    def get(self, what, cropped=False):
        adata = self.adata_cropped if cropped else self.adata
        if isinstance(what, str):  # Easy access to data or metadata arrays
            if what in adata.obs.columns:  # Metadata
                column_data = adata.obs[what]
                if column_data.dtype.name == 'category':
                    return column_data.astype(str).values
                return column_data.values
            elif what in adata.var.index:  # A gene
                return np.array(adata[:, what].X.todense().ravel()).flatten()
            elif what in adata.var.columns:  # Gene metadata
                column_data = adata.var[what]
                if column_data.dtype.name == 'category':
                    return column_data.astype(str).values
                return column_data.values
            obs_cols_lower = adata.obs.columns.str.lower()
            if what.lower() in obs_cols_lower:
                col_name = adata.obs.columns[obs_cols_lower.get_loc(what.lower())]
                column_data = adata.obs[col_name]
                if column_data.dtype.name == 'category':
                    return column_data.astype(str).values
                return column_data.values
            elif self.viz.organism == "mouse" and (what.lower().capitalize() in adata.var.index):
                return np.array(adata[:, what.lower().capitalize()].X.todense()).flatten()
            elif self.viz.organism == "human" and (what.upper() in adata.var.index):
                return np.array(adata[:, what.upper()].X.todense()).flatten()
            var_cols_lower = adata.var.columns.str.lower()
            if what.lower() in var_cols_lower:
                col_name = adata.var.columns[var_cols_lower.get_loc(what.lower())]
                column_data = adata.var[col_name]
                if column_data.dtype.name == 'category':
                    return column_data.astype(str).values
                return column_data.values
        else:
            # Create a new SingleCell object based on adata subsetting
            return self.subset(what)
        
    def subset(self, what=(slice(None), slice(None))):
        adata = self.adata[what].copy()
        return SingleCell(self.viz, adata)
    
    def __getitem__(self, what):
        '''get a vector from data (a gene) or metadata (from obs or var). or subset the object.'''
        item = self.get(what, cropped=False)
        if item is None:
            raise KeyError(f"[{what}] isn't in data or metadatas")
        return item
     
    def pseudobulk(self, by=None):
        if by is None:
            pb = self.adata.X.mean(axis=0).A1
            return pd.Series(pb, index=self.adata.var_names)
    
        expr_df = pd.DataFrame(self.adata.X.A,
                               index=self.adata.obs_names,
                               columns=self.adata.var_names)
        
        group_key = self.adata.obs[by]
        return expr_df.groupby(group_key).mean().T
    
    def sync_metadata_to_spots(self, what: str):
        '''transfers metadata assignment from the single-cell to the spots'''
        if what not in self.adata.obs:
            raise KeyError(f"'{what}' does not exist in SC.adata.obs.")
        cell_id_col = self.adata.obs.index.name
        if cell_id_col not in self.viz.adata.obs.columns:
            raise KeyError(f"'{cell_id_col}' does not exist in ViziumHD.adata.obs.")
        mapping = self.adata.obs[what]
        self.viz.adata.obs[what] = self.viz.adata.obs[cell_id_col].map(mapping)
        
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
        # s = f"SingleCell[{self.viz.name}]"
        s = self.__str__()
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
        else:
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
    
    def update(self):
        '''updates the methods in the instance'''
        ViziumHD_utils.update_instance_methods(self)
        ViziumHD_utils.update_instance_methods(self.plot)
        # self.plot._init_img()
    
    def head(self, n=5):
        return self.adata.obs.head(n) 
    
    @property
    def name(self):
        return self.viz.name + "_sc"
    
    def copy(self):
        return deepcopy(self)
    
    def export_to_matlab(self, path=None):
        var_names = self.adata.var_names.to_numpy()  
        if 'X_umap' in self.adata.obsm:
            self.adata.obs['UMAP_1'] = self.adata.obsm['X_umap'][:, 0]  
            self.adata.obs['UMAP_2'] = self.adata.obsm['X_umap'][:, 1]  
            
        obs = self.adata.obs.copy()
        def remove_non_ascii(d):
            return {re.sub(r'[^\x00-\x7F]+', '_', k): v for k, v in d.items()}
        
        obs = obs.to_dict(orient='list')  
        obs = remove_non_ascii(obs)

        if not path:
            path = f"{self.path_output}/matlab"
            if not os.path.exists(path):
                os.makedirs(path)
            path = f"{path}/{self.name}.mat"
        scipy.io.savemat(path, {"genes": var_names, "mat": self.adata.X,"metadata":obs})
        # self.adata.obs.to_csv(path.replace(".mat","metadata.csv"))
        
