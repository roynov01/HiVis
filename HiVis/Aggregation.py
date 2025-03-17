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
import pandas as pd
import anndata as ad
from shapely.affinity import scale
from scipy.stats import mode
import scipy.io
from scipy.spatial import cKDTree
from tqdm import tqdm
import geopandas as gpd

from . import HiVis_plot
from . import HiVis_utils

class Aggregation:
    def __init__(self, hiviz_instance, adata_agg, name, geojson_agg_path=None):
        '''
        Creates a new instance that is linked to a HiViz object.
        parameters:
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
            raise ValueError("Filtered AnnData object is empty. No valid rows remain.")
        
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
        self.adata_cropped = None
        self.tree = None
        
        if geojson_agg_path:
            self.import_geometry(geojson_agg_path,object_type="cell")


    def import_geometry(self, geojson_path, object_type="cell"):
        '''
        Adds "geometry" column to self.adata.obs, based on Geojson exported from Qupath.
        parameters:
            * geojson_path (str) - path to geojson file
            * object_type (str) - which "objectType" to merge from the geojson
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
        Merge info from an anndata to self.adata, in case genes have been filtered.
        parameters:
            * adata (as.AnnData) - anndata where to get the values from
            * obs - single string or list of obs to merge
            * var - single string or list of var to merge
            * umap (bool) - add umap to OBSM, and UMAP coordinates to obs?
            * pca (bool) - add PCA to OBSM?
            * hvg (bool) - add highly variable genes to vars?
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
            self.adata.obs = self.adata.obs.join(adata.obs[obs], how="left")
        if var:
            if isinstance(var, str):
                var = [var]
            existing_columns = [col for col in var if col in self.adata.var.columns]
            if existing_columns:
                self.adata.var.drop(columns=existing_columns, inplace=True)
            self.adata.var = self.adata.var.join(adata.var[var], how="left")
            
                
    def get(self, what, cropped=False, geometry=False):
        '''
        get a vector from data (a gene) or metadata (from obs or var). or subset the object.
        parameters:
            * what - if string, will get data or metadata. \
                     else, will return a new Aggregation object that is spliced. \
                     the splicing is passed to the self.adata.
            * cropped (bool) - get the data from the adata_cropped after crop() or plotting methods?
            * geometry (bool) - include only objects which have geometry
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
            # Create a new Aggregation object based on adata subsetting
            return self.subset(what)
        
    def subset(self, what=(slice(None), slice(None))):
        '''
        Create a new Aggregation object based on adata subsetting.
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
     
    def pseudobulk(self, by=None,layer=None):
        '''
        Returns the gene expression for each group in a single obs.
        If "by" is None, will return the mean expression of every gene.
        Else, will return a dataframe, each column is a value in "by" (for example cluster), rows are genes.
        layer - which layer in adata to use.
        '''
        if layer is None:
            x = self.adata.X
        else:
            if layer not in self.adata.layers:
                raise KeyError(f"Layer '{layer}' not found in self.adata.layers. Available layers: {list(self.adata.layers.keys())}")
            x = self.adata.layers[layer]
        
        if by is None:
            pb = x.mean(axis=0).A1
            return pd.Series(pb, index=self.adata.var_names)
    
        expr_df = pd.DataFrame(x.toarray(),
                               index=self.adata.obs_names,
                               columns=self.adata.var_names)
        
        group_key = self.adata.obs[by]
        return expr_df.groupby(group_key, observed=True).mean().T
    
    
    def smooth(self, what, radius, method="median", new_col_name=None, **kwargs):
        '''
        Applies median smoothing to the specified column in adata.obs using spatial neighbors.
        parameters:
            * what (str) - what to smooth. either a gene name or column name from self.adata.obs
            * radius (float) - in microns
            * method - ["mode","median", "mean", "gaussian", "log"]
            * new_col_name (str) - Optional custom name for the output column.
            **kwargs - Additional parameters for specific methods (e.g., sigma for gaussian, offset for log).
        '''
        coords = self.adata.obs[['um_x', 'um_y']].values

        if self.tree is None:
            # Build a KDTree for fast neighbor look-up.
            print("Building coordinate tree")
            self.tree = cKDTree(coords)
        
        values = self[what]
        if len(values) != self.adata.shape[0]:
            raise ValueError(f"{what} not in adata.obs or a gene name")
            
        if isinstance(values[0], str):
            if method != "mode":
                raise ValueError("Smoothing on string columns is only supported using the 'mode' method.")
    
        smoothed_values = []
        
        if method == "log":
            offset = kwargs.get("offset", 1.0)
            if np.min(values) < -offset:
                raise ValueError(f"Negative values detected in '{what}'. Log smoothing requires all values >= {-offset}.")
        elif method == "gaussian":
            sigma = kwargs.get("sigma", radius / 2)
    
        # Iterate through each object's coordinates, find neighbors, and compute the median.
        for i, point in enumerate(tqdm(coords, desc=f"{method} filtering '{what}' in radius {radius}")):
            # Find all neighbors within the given radius.
            indices = self.tree.query_ball_point(point, radius)
            if not indices:
                # Assign the original value or np.nan if no neighbor is found.
                new_val = values[i]
            neighbor_values = values[indices]
            
            if method == "median":
                new_val = np.median(neighbor_values)
            elif method == "mean":
                new_val = np.mean(neighbor_values)
            elif method == "mode":
                if isinstance(neighbor_values[0], str):
                    unique_vals, counts = np.unique(neighbor_values, return_counts=True)
                    new_val = unique_vals[np.argmax(counts)] 
                else:
                    new_val = mode(neighbor_values).mode
            elif method == "gaussian":
                # Calculate distances to neighbors.
                distances = np.linalg.norm(coords[indices] - point, axis=1)
                
                # Compute Gaussian weights.
                weights = np.exp(- (distances**2) / (2 * sigma**2))
                new_val = np.sum(neighbor_values * weights) / np.sum(weights)
            elif method == "log":
                # Apply a log1p transform to handle zero values; add an offset if necessary.
                offset = kwargs.get("offset", 1.0)
                # It is assumed that neighbor_values + offset > 0.
                new_val = np.expm1(np.median(np.log1p(neighbor_values + offset))) - offset
            else:
                raise ValueError(f"Unknown smoothing method: {method}")

            smoothed_values.append(new_val)

        if not new_col_name:
            new_col_name = f'{what}_smooth_r{radius}'
        self.adata.obs[new_col_name] = smoothed_values
        
    def noise_mean_curve(self, plot=False, layer=None, signif_thresh=0.95, inplace=False, **kwargs):
        '''
        Generates a noise-mean curve of the data.
        Parameters:
            * plot (bool) - plot the curve?
            * layer (str) - which layer in the anndata to use
            * signif_thresh (float) - for plotting, add text for genes in this residual precentile
            * inplace (bool) - add the mean_expression, cv and residuals to VAR?
        '''
        return HiVis_utils.noise_mean_curve(self.adata, plot=plot,layer=layer,
                                               signif_thresh=signif_thresh,inplace=inplace, **kwargs)
        
    
    def cor(self, what, self_corr_value=None, normilize=True, layer: str = None, inplace=False):
        '''
        Calculates gene(s) correlation.
        Parameters:
            * what (str or list) - if str, computes Spearman correlation of a given gene with all genes. \
                                    if list, will compute correlation between all genes in the list
            * self_corr_value - replace the correlation of the gene with itself by this value
            * normilize (bool) - normilize expression before computing correlation?
            * layer (str) - which layer in the anndata to use
            * inplace (bool) - add the correlation to VAR?
        '''
        if isinstance(what, str):
            x = self[what]
            return HiVis_utils.cor_gene(self.adata, x, what, self_corr_value, normilize, layer, inplace)
        return HiVis_utils.cor_genes(self.adata, what, self_corr_value, normilize, layer)

    def sync(self, what: str):
        '''
        Transfers metadata assignment from the Aggregation to the spots.
        what - obs column name to pass to HiViz object
        '''
        if what not in self.adata.obs:
            raise KeyError(f"'{what}' does not exist in agg.adata.obs.")
        agg_id_col = self.adata.obs.index.name
        if agg_id_col not in self.viz.adata.obs.columns:
            raise KeyError(f"'{agg_id_col}' does not exist in HiViz.adata.obs.")
        mapping = self.adata.obs[what]
        self.viz.adata.obs[what] = self.viz.adata.obs[agg_id_col].map(mapping)
        print("Columns in agg.adata.obs:", self.adata.obs.columns)
        print("Index name in agg.adata.obs:", self.adata.obs.index.name)
        print("Columns in HiViz.adata.obs:", self.viz.adata.obs.columns)
        print("Unique values in HiViz index column:", self.viz.adata.obs[self.adata.obs.index.name].unique())
        print("Mapping index (agg keys):", self.adata.obs.index.unique())
        
    def export_h5(self, path=None):
        '''
        Exports the adata. 
        * path (str) - path to save the h5 file. If None, will save to path_output
        '''
        print(f"SAVING [{self.name}]")
        if not path:
            path = f"{self.path_output}/{self.name}.h5ad"
        self.adata.write(path)
        return path
    
    def dge(self, column, group1, group2=None, method="wilcox", two_sided=False,
            umi_thresh=0, inplace=False, layer=None):
        '''
        Runs differential gene expression analysis between two groups.
        Values will be saved in self.var: expression_mean, log2fc, pval
        parameters:
            * column - which column in obs has the groups classification
            * group1 - specific value in the "column"
            * group2 - specific value in the "column". \
                       if None,will run agains all other values, and will be called "rest"
            * method - either "wilcox" or "t_test"
            * two_sided - if one sided, will give the pval for each group, \
                          and the minimal of both groups (which will also be FDR adjusted)
            * umi_thresh - use only spots with more UMIs than this number
            * expression - function F {mean, mean, max} F(mean(group1),mean(group2))
            * inplace - modify the adata.var with log2fc, pval and expression columns?
            * layer (str) - which layer in the anndata to use
        '''
        alternative = "two-sided" if two_sided else "greater"
        df = HiVis_utils.dge(self.adata, column, group1, group2, umi_thresh,layer=layer,
                     method=method, alternative=alternative, inplace=inplace)
        if group2 is None:
            group2 = "rest"
        df = df[[f"pval_{column}",f"log2fc_{column}",group1,group2]]
        df.rename(columns={f"log2fc_{column}":"log2fc"},inplace=True)
        if not two_sided:
            df[f"pval_{group1}"] = 1 - df[f"pval_{column}"]
            df[f"pval_{group2}"] = df[f"pval_{column}"]
            df["pval"] = df[[f"pval_{group1}",f"pval_{group2}"]].min(axis=1)
        else:
            df["pval"] = df[f"pval_{column}"]
        del df[f"pval_{column}"]
        df["qval"] = HiVis_utils.p_adjust(df["pval"])
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
        '''Combines two Aggregation objects into a single adata'''
        return self + other
    
    def __add__(self, other):
        '''Combines two Aggregation objects into a single adata'''
        if not isinstance(other, (Aggregation,Aggregation.Aggregation)):
            raise ValueError("Addition supported only for Aggregation class")
        self.adata.obs["source_"] = self.name
        other.adata.obs["source_"] = other.name if other.name != self.name else f"{self.name}_1"
        adata = ad.concat([self.adata, other.adata], join='outer')
        del self.adata.obs["source_"]
        del other.adata.obs["source_"]
        return adata
    
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
        '''Updates the methods in the instance. Should be used after modifying the source code in the class'''
        HiVis_utils.update_instance_methods(self)
        HiVis_utils.update_instance_methods(self.plot)
        _ = gc.collect()
    
    def head(self, n=5):
        return self.adata.obs.head(n) 
    
    @property
    def columns(self):
        return self.adata.obs.columns.copy()
    
    def copy(self):
        '''Creates a deep copy of the instance'''
        new = deepcopy(self)
        new.viz = self.viz
        gc.collect()
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
        
