# -*- coding: utf-8 -*-
"""
Aggregation of spots from HiVis
"""

from copy import deepcopy
import warnings
import re
import gc
import os
import numpy as np
import anndata as ad
from shapely.affinity import scale
import scipy.io
import geopandas as gpd

from . import HiVis_plot
from . import HiVis_analysis
from . import HiVis_utils

class Aggregation:
    '''
    Stores data of HiVis that have been aggregated (for example to single-cells). \
    Enables plotting via Aggregation.plot. Each instance is linked to a HiViz object. 
    '''
    
    def __init__(self, hiviz_instance, adata_agg, name, geojson_agg_path=None):
        '''
        Creates a new instance that is linked to a HiViz object.
        
        Parameters:
            * hiviz_instance (HiViz) - HiViz object
            * adata_agg (ad.AnnData) - anndata of aggregations
            * name (str) - name of object
            * geojson_path (str) - path of geojson, exported annotations
        '''
        if not isinstance(adata_agg, ad._core.anndata.AnnData): 
            raise ValueError("Adata must be Anndata object")
        if not "pxl_col_in_fullres" in adata_agg.obs.columns or not "pxl_row_in_fullres" in adata_agg.obs.columns:
            raise ValueError("Anndata.obs must include [pxl_col_in_fullres, pxl_row_in_fullres ]")
        adata_agg = adata_agg[adata_agg.obs["pxl_col_in_fullres"].notna(),:].copy()
        if adata_agg.shape[0] == 0:
            raise ValueError("AnnData object is empty columns")
        
        HiVis_utils.add_spatial_keys(hiviz_instance, adata_agg, name) # add obsm["spatial"] and uns["spatial"]
        
        scalefactor_json = hiviz_instance.json

        adata_agg.obs["pxl_col_in_lowres"] = adata_agg.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
        adata_agg.obs["pxl_row_in_lowres"] = adata_agg.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
        adata_agg.obs["pxl_col_in_highres"] = adata_agg.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
        adata_agg.obs["pxl_row_in_highres"] = adata_agg.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
        adata_agg.obs["um_x"] = adata_agg.obs["pxl_col_in_fullres"] * scalefactor_json["microns_per_pixel"]
        adata_agg.obs["um_y"] = adata_agg.obs["pxl_row_in_fullres"] * scalefactor_json["microns_per_pixel"]
        
        self.adata = adata_agg
        self.viz = hiviz_instance
        self.name = name 
        self.path_output = self.viz.path_output + f"/{self.name}"
        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        self.plot = HiVis_plot.PlotAgg(self)
        self.analysis = HiVis_analysis.AnalysisAgg(self)
        self.adata_cropped = None
        self.tree = None
        
        if geojson_agg_path:
            self.import_geometry(geojson_agg_path,object_type="cell")


    def import_geometry(self, geojson_path, object_type="cell"):
        '''
        Adds "geometry" column to self.adata.obs, based on Geojson exported from Qupath.
        
        Parameters:
            * geojson_path (str) - path to geojson file (can also be a geodataframe)
            * object_type (str) - which "objectType" to merge from the geojson
        '''
        print("[Importing geometry]")

        if isinstance(geojson_path,str):
            gdf = gpd.read_file(geojson_path)
        elif isinstance(geojson_path,gpd.GeoDataFrame):
            gdf = geojson_path
        if gdf.shape[0] != self.shape[0]:
            print(f"Number of shapes ({gdf.shape[0]}) isn't the same as the number of cells in adata ({self.shape[0]})")
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
        
    
    def merge(self, adata, obs=None, var=None, layer=None ,umap=True, pca=True, hvg=True):
        '''
        Merge info from an anndata to self.adata, in case genes have been filtered.
        
        Parameters:
            * adata (ad.AnnData) - anndata where to get the values from
            * obs - single string or list of obs to merge
            * var - single string or list of var to merge
            * layer - single string or list of layers to merge
            * umap (bool) - add umap to OBSM, and UMAP coordinates to obs
            * pca (bool) - add PCA to OBSM
            * hvg (bool) - add highly variable genes to vars
        '''
        
        if not obs:
            obs = []
        elif isinstance(obs, str):
            obs = [obs]
        if isinstance(var, str):
            var = [var]
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
        if layer is not None:
            if layer in adata.layers:
                if self.adata.shape[0] == adata.shape[0]:
                    self.adata.layers[layer] = adata.layers[layer].copy()
                else:
                    print(f"Can't add layer {layer} to self.adata.layers, size of adatas don't match")
            else:
                print(f"Layer {layer} not found in the provided adata.")
        if obs:
            existing_columns = [col for col in obs if col in self.adata.obs.columns]
            if existing_columns:
                self.adata.obs.drop(columns=existing_columns, inplace=True)
            self.adata.obs = self.adata.obs.join(adata.obs[obs], how="left")
            
            HiVis_utils._convert_bool_columns_to_float(self.adata.obs)
                        
        if var:
            existing_columns = [col for col in var if col in self.adata.var.columns]
            if existing_columns:
                self.adata.var.drop(columns=existing_columns, inplace=True)
            self.adata.var = self.adata.var.join(adata.var[var], how="left")
            
            HiVis_utils._convert_bool_columns_to_float(self.adata.var)
        
            
    def get(self, what, cropped=False, geometry=False, layer=None):
        '''
        get a vector from data (a gene) or metadata (from obs or var). or subset the object.
        
        Parameters:
            * what - if string, will get data or metadata. \
                     else, will return a new Aggregation object that is spliced. \
                     the splicing is passed to the self.adata.
            * cropped (bool) - get the data from the adata_cropped after plotting spatial
            * geometry (bool) - include only objects which have geometry
            
        **Returns**: either np.array of data or, if subsetting, a new Aggregation instance
        '''
        
        adata = self.adata_cropped if cropped else self.adata
        if geometry and self.plot.geometry is not None:
            adata = adata[adata.obs.index.isin(self.plot.geometry.index)]
        if isinstance(what, str):  # Easy access to data or metadata arrays
            if what in adata.obs.columns:  # Metadata from OBS
                column_data = adata.obs[what]
                if column_data.dtype.name == 'category':
                    return column_data.astype(str).values
                return column_data.values
            elif what in adata.var.index:  # A gene
                gene_data = adata[:, what].X if layer is None else adata[:, what].layers[layer]
                return np.array(gene_data.todense().ravel()).flatten()
            elif what in adata.var.columns:  # Gene metadata from VAR
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
                gene_name = what.lower().capitalize()
                gene_data = adata[:, gene_name].X if layer is None else adata[:, gene_name].layers[layer]
                return  np.array(gene_data.todense().ravel()).flatten()
            elif self.viz.organism == "human" and (what.upper() in adata.var.index):
                gene_name = what.lower().upper()
                gene_data = adata[:, gene_name].X if layer is None else adata[:, gene_name].layers[layer]
                return  np.array(gene_data.todense().ravel()).flatten()
            var_cols_lower = adata.var.columns.str.lower()
            if what.lower() in var_cols_lower:
                col_name = adata.var.columns[var_cols_lower.get_loc(what.lower())]
                column_data = adata.var[col_name]
                if column_data.dtype.name == 'category':
                    return column_data.astype(str).values
                return column_data.values
        else:
            # Create a new Aggregation object based on adata subsetting
            return self.subset(what)
        
    def subset(self, what=(slice(None), slice(None))):
        '''
        Create a new Aggregation object based on adata subsetting.
        **Returns** new Aggregation instance
        '''
        what = tuple(idx.to_numpy() if hasattr(idx, "to_numpy") else idx for idx in what)
        adata = self.adata[what].copy()
        adata.var = adata.var.loc[:,~adata.var.columns.str.startswith(("cor_","exp_"))]
        for layer in self.adata.layers.keys():
            adata.layers[layer] = self.adata.layers[layer][what].copy()
        return Aggregation(self.viz, adata, name=self.name)
    
    def __getitem__(self, what):
        '''get a vector from data (a gene) or metadata (from obs or var). or subset the object.'''
        item = self.get(what, cropped=False)
        if item is None:
            raise KeyError(f"[{what}] isn't in data or metadatas")
        return item
    
    def __contains__(self, what):
        if (what in self.adata.obs) or (what in self.adata.var):
            return True
        return False
     
    def sync(self, what: str):
        '''
        Transfers metadata assignment from the Aggregation to the spots.
        
        Parameters:
            * what (str) - obs column name to pass to HiViz object
        '''
        if what not in self.adata.obs:
            raise KeyError(f"'{what}' does not exist in agg.adata.obs.")
        agg_id_col = self.adata.obs.index.name
        if agg_id_col not in self.viz.adata.obs.columns:
            raise KeyError(f"'{agg_id_col}' does not exist in HiViz.adata.obs.")
        mapping = self.adata.obs[what]
        self.viz.adata.obs[what] = self.viz.adata.obs[agg_id_col].map(mapping)
        
    def export_h5(self, path=None):
        '''
        Exports the adata. 
        
        Parameters:
            * path (str) - path to save the h5 file. If None, will save to path_output
            
        **Returns** path where the file was saved (str)
        '''
        print(f"SAVING [{self.name}]")
        if not path:
            path = f"{self.path_output}/{self.name}.h5ad"
        self.adata.write(path)
        return path
    
   
    @property
    def shape(self):
        return self.adata.shape
    
    def __str__(self):
        s = f"# Aggregation # {self.name} #\n\n"
        s += f"# Parent: {self.viz.name} #\n"
        s += f"\tSize: {self.adata.shape[0]} x {self.adata.shape[1]}\n"
        s += '\nobs: '
        s += ', '.join(list(self.adata.obs.columns))
        if not self.adata.var.columns.empty:
            s += '\n\nvar: '
            s += ', '.join(list(self.adata.var.columns))
        layers = list(self.adata.layers.keys())
        if layers:
            s += '\n\nlayers: '
            s += ', '.join(layers)
        return s
    
    def __repr__(self):
        # s = f"Aggregation[{self.name}]"
        s = self.__str__()
        return s
    
    
    def combine(self, other):
        '''
        Combines two Aggregation objects into a single adata.
        '''
        return self + other
    
    def __add__(self, other):
        '''Combines two Aggregation objects into a single adata'''
        if not isinstance(other, type(self)):
            raise ValueError("Addition supported only for Aggregation class")
        self.adata.obs["source_"] = self.name
        other.adata.obs["source_"] = other.name if other.name != self.name else f"{self.name}_1"
        adata = ad.concat([self.adata, other.adata], join='outer')
        print("To add the combined anndata to HiVis object, use HiVis.add_agg(adata_agg, name)")
        return adata
    
    def __delitem__(self, key):
        '''Deletes metadata'''
        if isinstance(key, str):
            if key in self.adata.obs:
                del self.adata.obs[key]
            elif key in self.adata.var:
                del self.adata.var[key]
            else:
                raise KeyError(f"'{key}' not found in adata.obs")
        else:
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
    
    def __setitem__(self, key, value):
        if len(value) == self.adata.shape[0]:
            self.adata.obs[key] = value
        elif len(value) == self.adata.shape[1]:
            self.adata.var[key] = value
        else:
            raise ValueError("Values must be in the length of OBS or VAR")
    
    def update(self):
        '''
        Updates the methods in the instance. 
        Should be used after modifying the source code in the class
        '''
        HiVis_utils.update_instance_methods(self)
        HiVis_utils.update_instance_methods(self.plot)
        _ = gc.collect()
    
    def head(self, n=5):
        '''**Returns** Aggregation.adata.obs.head(n), where n is number of rows'''
        return self.adata.obs.head(n) 
    
    @property
    def columns(self):
        return self.adata.obs.columns.copy()
    
    def copy(self, new_name=None, new_out_path=False, full=False):
        '''
        Creates a deep copy of the instance
        if new_name is specified, renames the object and changes the path_output.
        If full is False, the name will be added to the current (previous) name.
        
        **Returns** new Aggregation instance
        '''
        new = deepcopy(self)
        new.viz = self.viz
        gc.collect()
        new = deepcopy(self)
        if new_name:
            new.rename(new_name, new_out_path=new_out_path, full=full)
        return new
    
    def rename(self, new_name: str, new_out_path=True, full=False):
        '''
        Renames the object and changes the path_output.
        If full is False, the name will be added to the current (previous) name
        '''
        if full:
            self.name = new_name
        else:
            self.name = f"{self.viz.name}_{new_name}"
        if new_out_path:
            self.path_output = self.viz.path_output + f"/{new_name}"
    
    def export_to_matlab(self, path=None, layer=None):
        '''
        Exports gene names, data (sparse matrix) and metadata to a .mat file.
        
        Parameters:
            * path (str) - path of mat file to save to
            * layer - which layer from the adata to use
            
        **Returns** path of mat file
        '''
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
        mat = self.adata.X if layer is None else self.adata.layers[layer]
        scipy.io.savemat(path, {"genes": var_names, "mat":mat ,"metadata":obs})
        self.adata.obs.to_csv(path.replace(".mat","metadata.csv"))
        return path
        
