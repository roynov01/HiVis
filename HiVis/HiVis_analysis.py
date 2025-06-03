# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:24:36 2025

@author: royno
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mode, zscore
from scipy.spatial import cKDTree
from tqdm import tqdm
from shapely.strtree import STRtree
from shapely import wkt, from_wkt
from shapely.geometry import Point
import matplotlib.pyplot as plt

from . import HiVis_utils



class AnalysisVisium:
    '''Handles all analysis functions for HiVis object'''
    def __init__(self, viz_instance):
        self.main = viz_instance
    
    def qc(self, save=False,figsize=(8, 8)):
        '''
        Plots basic QC (spatial, nUMI, mitochondrial %)
        
        Parameters:
            * save (bool) - save the plot in HiVis.path_output
        '''
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2,nrows=2, figsize=figsize)
        if self.main.json is not None:
            ax0 = self.main.plot.spatial(title=self.main.name, ax=ax0)
        ax1 = self.main.plot.hist("mito_percent_log10", title="Mitochondrial content per spot", xlab="log10(Mito %)",ax=ax1)
        ax2 = self.main.plot.hist("nUMI_log10", title="Number of UMIs per spot", xlab="log10(UMIs)",ax=ax2)
        ax3 = self.main.plot.hist("nUMI_gene_log10", title="Number of UMIs per gene", xlab="log10(UMIs)",ax=ax3)
        plt.tight_layout()
        if save:
            self.main.plot.save(figname="QC", fig=fig)
    
    def pseudobulk(self, by=None, layer=None):
        '''
        Sums the gene expression for each group in a single obs.
        
        Parameters:
            
            * by (str) - return a dataframe, each column is a value in "by" (for example cluster), rows are genes. \
            If None, will return the mean expression of every gene. 
            * layer (str) - which layer in adata to use.
            
        **Returns** the gene expression for each group (pd.DataFrame)
        '''
        if layer is None:
            x = self.main.adata.X
        else:
            if layer not in self.main.adata.layers:
                raise KeyError(f"Layer '{layer}' not found in self.main.adata.layers. Available layers: {list(self.main.adata.layers.keys())}")
            x = self.main.adata.layers[layer]
        if by is None:
            pb = x.mean(axis=0).A1
            return pd.Series(pb, index=self.main.adata.var_names)
        
        unique_groups = self.main.adata.obs[by].unique()
        unique_groups = unique_groups[pd.notna(unique_groups)]

        n_groups = len(unique_groups)
        n_genes = self.main.adata.n_vars  
        result = np.zeros((n_groups, n_genes))
        for i, group in enumerate(unique_groups):
            mask = (self.main.adata.obs[by] == group).values
            if mask.sum() == 0: 
                continue
            group_sum = x[mask].sum(axis=0)  
            group_mean = group_sum / mask.sum() 
            result[i, :] = group_mean.A1     
        return pd.DataFrame(result.T, index=self.main.adata.var_names, columns=unique_groups)
    
    def noise_mean_curve(self, layer=None, inplace=False, poly_deg=4):
        '''
        Generates a noise-mean curve of the data and calculates residuals.
        
        Parameters:
            * layer - which layer in the AnnData to use
            * inplace (bool) - add the mean_expression, cv and residuals to VAR
            * poly_deg (int) - degree of polynomial fit. if 1, will be a linear model
            
        **Returns** dataframe with expression, CV and residuals of each gene (pd.DataFrame). 
        '''
        return HiVis_utils.noise_mean_curve(self.main.adata,layer=layer,inplace=inplace, poly_deg=poly_deg)
    
    def cor(self, what, self_corr_value=None, normilize=True, layer: str = None, inplace=False):
        '''
        Calculates gene(s) correlation.
        
        Parameters:
            * what (str or list) - if str, computes Spearman correlation of a given gene with all genes. \
                                    if list, will compute correlation between all genes in the list
            * self_corr_value - replace the correlation of the gene with itself by this value
            * normalize (bool) - normilize expression before computing correlation
            * layer (str) - which layer in the AnnData to use
            * inplace (bool) - add the correlation to VAR
            
        **Returns** dataframe of spearman correlation between genes (pd.DataFrame)
        '''
        if isinstance(what, str):
            x = self.main[what]
            return HiVis_utils.cor_gene(self.main.adata, x, what, self_corr_value, normilize, layer, inplace)
        return HiVis_utils.cor_genes(self.main.adata, what, self_corr_value, normilize, layer)
    

    def score_genes(self, gene_list:list, score_name:str, z_normilize=False):
        '''
        Assigns score for each bin, based on a list of genes.
        
        Parameters:
            * gene_list (list) - list of genes
            * score_name (str) - name of column that will store the score in self.main.adata.obs
            * z_normilize (bool) - Z transform the score values
        
        **returns** score values (pd.Series)
        '''
        sc.tl.score_genes(self.main.adata, gene_list=gene_list, score_name=score_name)
        if z_normilize:
            self.main.adata.obs[score_name] = zscore(self.main.adata.obs[score_name])
        return self.main.adata.obs[score_name]


    def smooth(self, what, radius, method="mean", new_col_name=None, layer=None, **kwargs):
        r'''
        Applies median smoothing to the specified column in adata.obs using spatial neighbors.
        
        Parameters:
            * what (str) - what to smooth. either a gene name or column name from self.main.adata.obs
            * radius (float) - in microns
            * method - ["mode", "median", "mean", "gaussian", "log"]
            * new_col_name (str) - Optional custom name for the output column
            * layer (str) - which layer in the AnnData to use
            * \**kwargs - Additional Parameters for specific methods (e.g., sigma for gaussian, offset for log).
            
        **returns** smoothed values (pd.Series)
        '''
        coords = self.main.adata.obs[['um_x', 'um_y']].values

        if self.main.tree is None:
            # Build a KDTree for fast neighbor look-up.
            print("Building coordinate tree")
            self.main.tree = cKDTree(coords)
        
        values = self.main.get(what,layer=layer)
        if len(values) != self.main.adata.shape[0]:
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
        for i, point in enumerate(tqdm(coords, desc=f"'{method} smoothing '{what}' in radius {radius}")):
            # Find all neighbors within the given radius.
            indices = self.main.tree.query_ball_point(point, radius)
            if not indices:
                # Assign the original value or np.nan if no neighbor is found.
                new_val = values[i]
            neighbor_values = values[indices]
            
            if method == "median":
                new_val = np.nanmedian(neighbor_values)
            elif method == "mean":
                new_val = np.nanmean(neighbor_values)
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
        self.main.adata.obs[new_col_name] = smoothed_values
        return smoothed_values
    
    
    def dge(self, column, group1, group2=None, method="fisher_exact", two_sided=False,
            umi_thresh=0, inplace=False, layer=None):
        '''
        Runs differential gene expression analysis between two groups.
        Values will be saved in self.main.var: expression_mean, log2fc, pval
        
        Parameters:
            * column (str) - which column in obs has the groups classification
            * group1 (str) - specific value in the "column"
            * group2 (str) - specific value in the "column". \
                       if None, will run against all other values, and will be called "rest"
            * method (str) - one of ["fisher_exact", "wilcox", "t_test"]
            * two_sided (bool) - if one sided, will give the pval for each group, \
                          and the minimal of both groups (which will also be FDR adjusted)
            * umi_thresh (int) - use only spots with more UMIs than this number
            * inplace (bool) - modify the adata.var with log2fc, pval and expression columns
            * layer (str) - which layer in the AnnData to use
            
        **Returns** the DGE results (pd.DataFrame)
        '''
        alternative = "two-sided" if two_sided else "greater"
        df = HiVis_utils.dge(self.main.adata, column, group1, group2, umi_thresh,layer=layer,
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
            cols_to_drop = [col for col in var.columns if col in self.main.adata.var.columns]
            self.main.adata.var.drop(columns=cols_to_drop,inplace=True)
            self.main.adata.var = self.main.adata.var.join(var, how="left")
        return df
    
        
    def compute_distances(self, agg_name, dist_col_name=None, nearest_col_name=None):
        '''
        Compute distances of each bin too the nearest aggregation.

        Parameters:
            * agg_name (str) - name of agg
            * dist_col_name (str) - Name of column to save distance to. default is dist_to_{agg_name}
            * nearest_col_name (str) -  Name of column to save the closest aggregation name
        '''
        if agg_name not in self.main.agg:
            raise ValueError(f"{agg_name} is not a valid aggregation in {self.main.name}")

        dist_col_name = dist_col_name if dist_col_name is not None else f"dist_to_{agg_name}"
        nearest_col_name = nearest_col_name if nearest_col_name is not None else f"nearest_{agg_name}"

        target_geoms = self.main.agg[agg_name].adata.obs.geometry.apply(wkt.loads).tolist()
        tree = STRtree(target_geoms)

        x_coords = self.main["um_x"]
        y_coords = self.main["um_y"]
        n_pts = len(x_coords)
        distances = [0.0] * n_pts
        nearest_obs_ids = [None] * n_pts

        for i, (x, y) in enumerate(tqdm(zip(x_coords, y_coords), total=n_pts, desc=f"Computing distance to {agg_name}")):
            pt = Point(x, y)
            nearest_index = tree.nearest(pt)
            nearest_geom = target_geoms[nearest_index] 
            distances[i] = pt.distance(nearest_geom)
            nearest_obs_ids[i] = self.main.agg[agg_name].adata.obs.index[nearest_index]

        self.main.adata.obs[dist_col_name] = distances
        self.main.adata.obs[nearest_col_name] = nearest_obs_ids

    

class AnalysisAgg:
    '''Handles all analysis functions for Aggregation object'''
    def __init__(self, agg_instance):
        self.main = agg_instance

    def pseudobulk(self, by=None,layer=None):
        '''
        Sums the gene expression for each group in a single obs.
        
        Parameters:
            * by (str) - return a dataframe, each column is a value in "by" (for example cluster), rows are genes. \
            If None, will return the mean expression of every gene. 
            * layer (str) - which layer in adata to use.
            
        **Returns** the gene expression for each group (pd.DataFrame)
        '''
        if layer is None:
            x = self.main.adata.X
        else:
            if layer not in self.main.adata.layers:
                raise KeyError(f"Layer '{layer}' not found in self.main.adata.layers. Available layers: {list(self.main.adata.layers.keys())}")
            x = self.main.adata.layers[layer]
        
        if by is None:
            pb = x.mean(axis=0).A1
            return pd.Series(pb, index=self.main.adata.var_names)
    
        expr_df = pd.DataFrame(x.toarray(),
                               index=self.main.adata.obs_names,
                               columns=self.main.adata.var_names)
        
        group_key = self.main.adata.obs[by]
        return expr_df.groupby(group_key, observed=True).mean().T
    
    
    def smooth(self, what, radius, method="mean", new_col_name=None, layer=None, **kwargs):
        '''
        Applies median smoothing to the specified column in adata.obs using spatial neighbors.
        
        Parameters:
            * what (str) - what to smooth. either a gene name or column name from self.main.adata.obs
            * radius (float) - in microns
            * method - ["mode", "median", "mean", "gaussian", "log"]
            * new_col_name (str) - Optional custom name for the output column
            * layer (str) - which layer in the AnnData to use
            * \**kwargs - Additional Parameters for specific methods (e.g., sigma for gaussian, offset for log).
            
        **returns** smoothed values (pd.Series)
        '''
        coords = self.main.adata.obs[['um_x', 'um_y']].values

        if self.main.tree is None:
            # Build a KDTree for fast neighbor look-up.
            print("Building coordinate tree")
            self.main.tree = cKDTree(coords)
        
        values = self.main.get(what,layer=layer)
        if len(values) != self.main.adata.shape[0]:
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
        for i, point in enumerate(tqdm(coords, desc=f"'{method}' smoothing '{what}' in radius {radius}")):
            # Find all neighbors within the given radius.
            indices = self.main.tree.query_ball_point(point, radius)
            if not indices:
                # Assign the original value or np.nan if no neighbor is found.
                new_val = values[i]
            neighbor_values = values[indices]
            
            if method == "median":
                new_val = np.nanmedian(neighbor_values)
            elif method == "mean":
                new_val = np.nanmean(neighbor_values)
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
        self.main.adata.obs[new_col_name] = smoothed_values
        return smoothed_values
    
    def score_genes(self, gene_list:list, score_name:str, z_normilize=False):
        '''
        Assigns score for each object, based on a list of genes.
        
        Parameters:
            * gene_list (list) - list of genes
            * score_name (str) - name of column that will store the score in self.main.adata.obs
            * z_normilize (bool) - Z transform the score values
        
        **returns** score values (pd.Series)
        '''
        sc.tl.score_genes(self.main.adata, gene_list=gene_list, score_name=score_name)
        if z_normilize:
            self.main.adata.obs[score_name] = zscore(self.main.adata.obs[score_name])
        return self.main.adata.obs[score_name]
        
    def noise_mean_curve(self, layer=None,inplace=False,poly_deg=4):
        '''
        Generates a noise-mean curve of the data.
        
        Parameters:
            * layer (str) - which layer in the AnnData to use
            * inplace (bool) - add the mean_expression, cv and residuals to VAR
            * poly_deg (int) - degree of polynomial fit. if 1, will be a linear model.
            
        **Returns** dataframe with expression, CV and residuals of each gene (pd.DataFrame).
        '''
        return HiVis_utils.noise_mean_curve(self.main.adata,layer=layer,inplace=inplace,poly_deg=poly_deg)
        
    
    def cor(self, what, self_corr_value=None, normilize=True, layer: str = None, inplace=False):
        '''
        Calculates gene(s) correlation.
        
        Parameters:
            * what (str or list) - if str, computes Spearman correlation of a given gene with all genes. \
                                    if list, will compute correlation between all genes in the list
            * self_corr_value - replace the correlation of the gene with itself by this value
            * normalize (bool) - normilize expression before computing correlation
            * layer (str) - which layer in the AnnData to use
            * inplace (bool) - add the correlation to VAR
            
        **Returns** dataframe of spearman correlation between genes (pd.DataFrame)
        '''
        if isinstance(what, str):
            x = self.main[what]
            return HiVis_utils.cor_gene(self.main.adata, x, what, self_corr_value, normilize, layer, inplace)
        return HiVis_utils.cor_genes(self.main.adata, what, self_corr_value, normilize, layer)


    def dge(self, column, group1, group2=None, method="wilcox", two_sided=False,
            umi_thresh=0, inplace=False, layer=None):
        '''
        Runs differential gene expression analysis between two groups.
        Values will be saved in self.main.var: expression_mean, log2fc, pval
        
        Parameters:
            * column (str) - which column in obs has the groups classification
            * group1 (str) - specific value in the "column"
            * group2 (str) - specific value in the "column". \
                       if None, will run against all other values, and will be called "rest"
            * method (str) - one of ["fisher_exact", "wilcox", "t_test"]
            * two_sided (bool) - if one sided, will give the pval for each group, \
                          and the minimal of both groups (which will also be FDR adjusted)
            * umi_thresh (int) - use only spots with more UMIs than this number
            * inplace (bool) - modify the adata.var with log2fc, pval and expression columns
            * layer (str) - which layer in the AnnData to use
            
        **Returns** the DGE results (pd.DataFrame)
        '''
        alternative = "two-sided" if two_sided else "greater"
        df = HiVis_utils.dge(self.main.adata, column, group1, group2, umi_thresh,layer=layer,
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
            cols_to_drop = [col for col in var.columns if col in self.main.adata.var.columns]
            self.main.adata.var.drop(columns=cols_to_drop,inplace=True)
            self.main.adata.var = self.main.adata.var.join(var, how="left")
        return df
    
    
    def compute_distances(self, target_agg, dist_col_name=None, nearest_col_name=None):
        '''
        Compute distances of each object to the nearest object of another Aggregation.
        
        Parameters:
            * target_agg - either Aggregation object, or a name (str) that is the key in self.main.viz.agg
            * dist_col_name (str) - Name of column to save distance to. default is dist_to_{agg_name}
            * nearest_col_name (str) -  Name of column to save the closest aggregation name
        '''
        if isinstance(target_agg, str):
            target_agg = self.main.viz.agg[target_agg]
        if target_agg is self:
            raise ValueError("target_agg should be either another Agg object or key in HiVis.agg")

        target_name = target_agg.name
        dist_col_name = dist_col_name    or f"dist_to_{target_name}"
        nearest_col_name = nearest_col_name or f"nearest_{target_name}"

        def to_geoms(series):
            if series.dtype == "object" and isinstance(series.iloc[0], str):
                return series.map(from_wkt)
            return series

        cur_geoms = to_geoms(self.main.adata.obs.geometry)
        target_geoms = to_geoms(target_agg.adata.obs.geometry)

        # build STRtree for fast nearest-neighbour look-ups 
        target_list = target_geoms.tolist()
        tree  = STRtree(target_list)
        distances = [0.0]  * len(cur_geoms)
        nearest_ids = [None] * len(cur_geoms)
        
        
        # id2idx = {id(g): i for i, g in enumerate(target_list)}  # map geom → row index
        # for i, poly in enumerate(tqdm(cur_geoms, desc=f"Distances → {target_name}")):
        #     nearest_geom = tree.nearest(poly)
        #     distances[i] = poly.distance(nearest_geom)
        #     nearest_ids[i] = target_agg.adata.obs.index[id2idx[id(nearest_geom)]]


        for i, poly in enumerate(tqdm(cur_geoms, desc=f"Distances → {target_name}")):
            nearest_idx   = int(tree.nearest(poly))    
            nearest_geom  = target_list[nearest_idx]    
            distances[i]  = poly.distance(nearest_geom)
            nearest_ids[i]= target_agg.adata.obs.index[nearest_idx]




        self.main.adata.obs[dist_col_name] = distances
        self.main.adata.obs[nearest_col_name] = nearest_ids



