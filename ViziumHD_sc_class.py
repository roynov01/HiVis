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
import geopandas as gpd
import warnings
import re
from shapely.affinity import scale
import gc

import ViziumHD_utils
import ViziumHD_plot


class SingleCell:
    def __init__(self, vizium_instance, adata_sc, geojson_cells_path=None):
        '''
        Creates a new instance that is linked to a ViziumHD object and has a PlosSC object.
        parameters:
            * vizium_instance - ViziumHD object
            * adata_sc - anndata of single cells
            * geojson_path - path of geojson, exported cells
        '''
        if not isinstance(adata_sc, ad._core.anndata.AnnData): 
            raise ValueError("Adata must be Anndata object")
        adata_sc = adata_sc[adata_sc.obs["pxl_col_in_fullres"].notna(),:].copy()
        if adata_sc.shape[0] == 0:
            raise ValueError("Filtered AnnData object is empty. No valid rows remain.")
        
        scalefactor_json = vizium_instance.json
        adata_sc.obs["pxl_col_in_lowres"] = adata_sc.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
        adata_sc.obs["pxl_row_in_lowres"] = adata_sc.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
        adata_sc.obs["pxl_col_in_highres"] = adata_sc.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
        adata_sc.obs["pxl_row_in_highres"] = adata_sc.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
        adata_sc.obs["um_x"] = adata_sc.obs["pxl_col_in_fullres"] * scalefactor_json["microns_per_pixel"]
        adata_sc.obs["um_y"] = adata_sc.obs["pxl_row_in_fullres"] * scalefactor_json["microns_per_pixel"]
        
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
        '''
        Adds "geometry" column to self.adata.obs, based on Geojson exported from Qupath.
        parameters:
            * geojson_path - path to geojson file
            * object_type - which "objectType" to merge from the geojson
        '''
        if isinstance(geojson_path,str):
            gdf = gpd.read_file(geojson_path)
        elif isinstance(geojson_path,gpd.GeoDataFrame):
            gdf = geojson_path
        gdf = gdf[gdf["objectType"] == object_type]
        gdf = gdf.loc[:,["id","geometry"]]
        gdf.rename(columns={"id":self.adata.obs.index.name},inplace=True)

        microns_per_pixel = self.viz.json["microns_per_pixel"]
        gdf["geometry"] = gdf["geometry"].apply(lambda geom: scale(geom, xfact=microns_per_pixel, yfact=microns_per_pixel, origin=(0, 0)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.wkt)
        gdf = gdf.set_index(self.adata.obs.index.name)
    
        if "geometry" in self.adata.obs.columns:
            print("Geometry column already exists, overwriting...")
            del self.adata.obs["geometry"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry column does not contain geometry")
            self.adata.obs = self.adata.obs.join(gdf,how="left")
        
    
    def merge(self, adata, obs=None, var=None, umap=True, pca=True, hvg=True):
        '''
        Merge info from an anndata to self.adata.
        parameters:
            * adata - anndata where to get the values from
            * obs - single string or list of obs to merge
            * var - single string or list of var to merge
            * umap - add umap to OBSM, and UMAP coordinates to obs?
            * pca - add PCA to OBSM?
            * hvg - add highly variable genes to vars?
        '''
        if not obs:
            obs = []
        elif isinstance(obs, str):
            obs = [obs]
        if umap and "X_umap" in adata.obsm:
            if self.adata.shape[0] == adata.shape[0]:
                self.adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
            else:
                print("Cant add UMAP to obsm, size of adatas don't match")
            umap_coords = adata.obsm['X_umap']
            adata.obs['UMAP_1'] = umap_coords[:, 0]
            adata.obs['UMAP_2'] = umap_coords[:, 1]
            
            obs += ['UMAP_1','UMAP_2']
        if pca and "X_pca" in adata.obsm:
            if self.adata.shape[0] == adata.shape[0]:
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
            
                
    def get(self, what, cropped=False, geometry=False):
        '''
        get a vector from data (a gene) or metadata (from obs or var). or subset the object.
        parameters:
            * what - if string, will get data or metadata. 
                     else, will return a new SingleCell object that is spliced.
                     the splicing is passed to the self.adata.
            * cropped - get the data from the adata_cropped after crop() or plotting methods?
            * geometry - include only cells which have geometry
        '''
        adata = self.adata_cropped if cropped else self.adata
        if geometry and self.plot.geometry is not None:
            adata = adata[adata.obs.index.isin(self.plot.geometry.index)]
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
        '''
        Transfers metadata assignment from the single-cell to the spots.
        what - obs column name to pass to ViziumHD object
        '''
        if what not in self.adata.obs:
            raise KeyError(f"'{what}' does not exist in SC.adata.obs.")
        cell_id_col = self.adata.obs.index.name
        if cell_id_col not in self.viz.adata.obs.columns:
            raise KeyError(f"'{cell_id_col}' does not exist in ViziumHD.adata.obs.")
        mapping = self.adata.obs[what]
        self.viz.adata.obs[what] = self.viz.adata.obs[cell_id_col].map(mapping)
        
    def export_h5(self, path=None):
        '''
        Exports the adata. path - path to save the h5 file
        '''
        print(f"SAVING [{self.name}]")
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
        _ = gc.collect()
    
    def head(self, n=5):
        return self.adata.obs.head(n) 
    
    @property
    def name(self):
        return self.viz.name + "_sc"
    
    def copy(self):
        return deepcopy(self)
    
    def export_to_matlab(self, path=None):
        '''exports gene names, data (sparse matrix) and metadata to a .mat file'''
        var_names = self.adata.var_names.to_numpy()  
        if 'X_umap' in self.adata.obsm:
            self.adata.obs['UMAP_1'] = self.adata.obsm['X_umap'][:, 0]  
            self.adata.obs['UMAP_2'] = self.adata.obsm['X_umap'][:, 1]  
            
        obs = self.adata.obs.copy()
        obs["Cell_ID"] = obs.index.tolist()
        
        # Shorten long column names in obs
        def shorten_col_names(columns, max_len=28):
            seen_names = {}
            rename_dict = {}
            for col in columns:
                if len(col) > max_len:
                    base_name = col[:max_len]  
                    count = seen_names.get(base_name, 0)
                    new_name = f"{base_name}_{count}"
                    seen_names[base_name] = count + 1
                    rename_dict[col] = new_name
            return rename_dict
        
        rename_dict = shorten_col_names(obs.columns)
        obs = obs.rename(columns=rename_dict)
        
        def remove_non_ascii(d):
            return {re.sub(r'[^\x00-\x7F]+', '_', k): v for k, v in d.items()}
        
        obs = obs.to_dict(orient='list')  
        obs = remove_non_ascii(obs)

        if not path:
            path = f"{self.path_output}/matlab"
            if not os.path.exists(path):
                os.makedirs(path)
            path = f"{path}/{self.name}.mat"
        print("[Saving mat file]")
        scipy.io.savemat(path, {"genes": var_names, "mat": self.adata.X,"metadata":obs})
        self.adata.obs.to_csv(path.replace(".mat","metadata.csv"))
        
