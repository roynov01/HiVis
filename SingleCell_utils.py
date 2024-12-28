# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:49:15 2024

@author: royno
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from scipy.sparse import lil_matrix
import os

# import ViziumHD_utils
# import ViziumHD_class
# import ViziumHD_plot
import ViziumHD_sc_class



def _aggregate_spots(adata, sc_metadata, columns, custom_agg=None):
    if "Object ID" not in columns:
        columns += "Object ID"
    for w in ["Object ID","Name","in_nucleus","in_cell"]:
        if w not in sc_metadata.columns:
            raise ValueError("columns of df must include ['Object ID','Name','in_nucleus','in_cell']")
    spots_only = sc_metadata.loc[sc_metadata['Classification']=='Spot',['Name','in_nucleus','in_cell']]
    cells_only = sc_metadata.loc[sc_metadata['Classification']=='Cell',columns]
    cells_only.rename(columns={'Object ID': 'Cell_ID'},inplace=True)
    _split_name(spots_only, adata)
    
    custom_agg = custom_agg if custom_agg else {}
    _aggregate_meta(adata, cells_only, user_aggregations=custom_agg)
    
    adata_sc = _aggregate_data(cells_only, adata)
    return adata_sc


def _split_name(spots_only, adata):
    def helper(s):
        if '__' in s:
            parts = s.split('__')
            if len(parts) == 2:
                return parts[0], parts[1]
        return s, np.nan
    tqdm.pandas(desc="[Splitting name column]")
    spots_only[['Spot_ID', 'Cell_ID']] = spots_only['Name'].progress_apply(lambda s: pd.Series(helper(s)))
    spots_only = spots_only.set_index("Spot_ID")

    del spots_only['Name']
    
    adata.obs['Spot_ID'] = adata.obs.index
    print("[Adding cells to the spots-adata]")
    adata.obs = adata.obs.join(spots_only,how='left')
    # adata.obs = adata.obs.join(spots_only,on="Spot_ID",how='left')
# 


def _aggregate_meta(adata, cells_only, user_aggregations=None):
    def _guess_default_aggregator(series, default_numeric=np.median, default_categorical=custom_mode):
        """Guess aggregator based on numeric vs. non-numeric dtype."""
        if pd.api.types.is_numeric_dtype(series):
            return default_numeric
        else:
            return default_categorical

    def _prepare_aggregations(df, user_aggregations=None,
                              default_numeric=np.median,
                              default_categorical=custom_mode):
        """
        Build a dict of {column_name: function or [functions]}.
        1. Use user_aggregations if given.
        2. Otherwise, guess aggregator based on column type.
        """
        if user_aggregations is None:
            user_aggregations = {}
        final_aggregations = {}
        for col in df.columns:
            if col in user_aggregations:
                final_aggregations[col] = user_aggregations[col]
            else:
                final_aggregations[col] = _guess_default_aggregator(
                    df[col],default_numeric,default_categorical)
        return final_aggregations

    def _aggregate_with_progress(grouped, aggregations):
        """
        Apply the aggregations to each group with a progress bar.
        :param grouped: A pandas groupby object
        :param aggregations: Dict of {col_name: function or list_of_functions}
        :return: DataFrame with one row per group
        """
        results = []        
        for name, group in tqdm(grouped, total=len(grouped), desc="Aggregating spots metadata"):
            row_result = {}
            for col_name, funcs in aggregations.items():
                if not isinstance(funcs, list):
                    funcs = [funcs]
                for func in funcs:
                    agg_value = func(group[col_name])
                    # If multiple functions, store separately using function name
                    if len(funcs) > 1:
                        fun_name = func.__name__.lstrip("_")
                        row_result[f"{col_name}_{fun_name}"] = agg_value
                    else:
                        row_result[col_name] = agg_value
            
            row_result['Cell_ID'] = name
            results.append(row_result)
        return pd.DataFrame(results)
    
    # Prepare final aggregator dictionary
    final_aggregations = _prepare_aggregations(
        adata.obs, user_aggregations=user_aggregations,default_numeric=np.median,
        default_categorical=custom_mode)

    # Perform aggregation
    cells_only2 = _aggregate_with_progress(adata.obs.groupby('Cell_ID'), final_aggregations)

    # Merge results   
    cells_only = pd.merge(cells_only, cells_only2, on='Cell_ID', how='left')


def _aggregate_data(cells_only, adata, output_dir=None, name=''):
    if 'Cell_ID' in cells_only.columns:
        cells_only.set_index('Cell_ID', inplace=True)
    adata_filtered = adata[(adata.obs['in_cell'] == 1)]
    
    # Split into nucleus/cytoplasm subsets
    adata_nuc = adata_filtered[adata_filtered.obs['in_nucleus'] == 1].copy()
    adata_cyto = adata_filtered[adata_filtered.obs['in_nucleus'] == 0].copy()
    ind_dict_nuc = adata_nuc.obs.groupby(by=['Cell_ID']).indices
    ind_dict_cyto = adata_cyto.obs.groupby(by=['Cell_ID']).indices

    # Find cell ids that have both nucleus and cytoplasm
    cells_ids = np.intersect1d(list(ind_dict_nuc.keys()), list(ind_dict_cyto.keys()))
    num_genes = adata_filtered.shape[1]
    num_cells = len(cells_ids)
    
    # Preallocate sparse matrices
    nucleus_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    cyto_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    
    # Aggregate the spots in nucleus and in cytoplasm for each cell
    for i, cell in enumerate(tqdm(cells_ids, desc='Aggregating spots expression')): 
        nucleus_data[i, :] = adata_nuc[ind_dict_nuc[cell],:].X.sum(axis=0) 
        cyto_data[i, :] = adata_cyto[ind_dict_cyto[cell],:].X.sum(axis=0) 
    
    # Convert to sparse
    nucleus_data = nucleus_data.tocsr()
    cyto_data = cyto_data.tocsr()
    cell_data = nucleus_data + cyto_data

    # Create anndata
    obs_df = cells_only.loc[cells_ids].copy()  # ensures same order as in the loop above
    adata_sc = sc.AnnData(X=cell_data,obs=obs_df,var=adata_filtered.var)
    adata_sc.layers["nuc"] = nucleus_data
    adata_sc.layers["cyto"] = cyto_data

    return adata_sc


def custom_mode(series):
    mode_series = series.mode()
    if not mode_series.empty:
        return mode_series.iloc[0]
    return np.nan    












