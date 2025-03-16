# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 09:59:38 2025

@author: royno
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import anndata as ad
from scipy.sparse import lil_matrix

import HiVis_utils



def new_adata(adata, aggregate_by, aggregation_func, obs2agg=None, **aggregation_kwargs):
    '''
    Creates a new AnnData by aggregating both expression and metadata from the HiVis.adata.
    Parameters:
        * adata - AnnData of HiVis object
        * aggregate_by - Column in obs by which to group spots
        * aggregation_func - Function that aggregates expression data (X, layers)
        * obs2agg - what obs to aggregate from the HiVis? 
                    Can be a list of column names. numeric columns will be summed, categorical will be the mode.
                    Can be a dict specifying the aggregation function. 
                        examples: {"value_along_axis":np.median} or {"value_along_axis":[np.median,np.mean]}
        * aggregation_kwargs - extra arguments for the aggregation_func
    '''
    if obs2agg is None:
        obs2agg = {}
    if isinstance(obs2agg, list):
        obs2agg = {k:None for k in set(obs2agg)}
    obs2agg["pxl_row_in_fullres"] = np.mean
    obs2agg["pxl_col_in_fullres"] = np.mean
    obs2agg["nUMI"] = np.sum    
        
    meta_df, meta_ids = _aggregate_meta(adata=adata,aggregate_by=aggregate_by,custom_agg=obs2agg)

    expr_data, expr_ids, layers, other_results = aggregation_func(adata, group_col=aggregate_by,**aggregation_kwargs)

    meta_df = meta_df.reindex(expr_ids)

    adata_agg = ad.AnnData(X=expr_data,obs=meta_df,var=pd.DataFrame(index=adata.var_names))

    if layers:
        for layer_name, layer_data in layers.items():
            adata_agg.layers[layer_name] = layer_data

    return adata_agg, other_results


def split_stardist(input_df):
    df = input_df.copy()
    print("[Splitting name column and merging metadata]")
    split_names = df['Name'].str.split('__', n=1, expand=True)
    split_names.columns = ['Spot_ID', 'Cell_ID']
    split_names['Cell_ID'] = split_names['Cell_ID'].fillna(value=np.nan)
    df[['Spot_ID', 'Cell_ID']] = split_names
    cols = ['in_nucleus', 'in_cell', 'Cell_ID', 'Spot_ID']
    spots_only = df.loc[input_df['Classification']=='Spot',cols]
    spots_only = spots_only.set_index("Spot_ID")
        
    cells_only = input_df.loc[input_df['Classification']=='Cell']
    cells_only = cells_only.set_index("Cell_ID")
    # vizium_instance.adata.obs = vizium_instance.adata.obs.join(cells_only,how='left',on="Cell_ID")
    
    return spots_only, cells_only


def _aggregate_meta(adata, aggregate_by, custom_agg):
    '''
    Helper function for "new_adata". Aggregates metadata.
    '''
    def custom_mode(series):
        mode_series = series.mode()
        if not mode_series.empty:
            return mode_series.iloc[0]
        return np.nan
    
    def _guess_aggregator(series):
        if pd.api.types.is_numeric_dtype(series):
            return np.sum
        else:
            return custom_mode
        
    columns = list(custom_agg.keys())
    df = adata.obs[columns + [aggregate_by]].copy()
    
    agg_dict = {}
    for col in df.columns:
        if col == aggregate_by:
            continue
        if col in custom_agg and custom_agg[col]:
            agg_dict[col] = custom_agg[col]
        else:
            agg_dict[col] = _guess_aggregator(df[col])

    grouped = df.groupby(aggregate_by, sort=False)
    group_results, group_order = [], []
    for group_val, sub_df in tqdm(grouped, total=len(grouped), desc="Aggregating metadata"):
        row = {}
        for col_name, funcs in agg_dict.items():
            # Each aggregator could be a single function or a list of functions
            if not isinstance(funcs, list):
                funcs = [funcs]
            for func in funcs:
                result = func(sub_df[col_name])
                if len(funcs) > 1:
                    # e.g. col_name_mean, col_name_std, etc.
                    func_name = getattr(func, "__name__", "func")
                    row[f"{col_name}_{func_name}"] = result
                else:
                    row[col_name] = result
        row["spot_count"] = len(sub_df)
        # Keep track of the group value
        row[aggregate_by] = group_val
        group_results.append(row)
        group_order.append(group_val)
    updated_obs = pd.DataFrame(group_results).set_index(aggregate_by)
    return updated_obs, group_order


def _aggregate_data_stardist(adata, group_col="Cell_ID", in_cell_col="in_cell",nuc_col="in_nucleus"):
    '''
    Helper function that can be used for as "aggregation_func" in new_adata().
    Aggregates expression data based on processed dataframe from 
    Ofras pipeline that uses Stardist + extension. (version used in small intestine).    
    '''
    adata_filtered = adata[(adata.obs[in_cell_col] == 1)]
    
    # Split into nucleus/cytoplasm subsets
    print("[Splitting data to nuc/cyto]")
    adata_nuc = adata_filtered[adata_filtered.obs[nuc_col] == 1].copy()
    adata_cyto = adata_filtered[adata_filtered.obs[nuc_col] == 0].copy()
    ind_dict_nuc = adata_nuc.obs.groupby(by=[group_col]).indices
    ind_dict_cyto = adata_cyto.obs.groupby(by=[group_col]).indices

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
    print("[Converting to sparse matrices]")
    nucleus_data = nucleus_data.tocsr()
    cyto_data = cyto_data.tocsr()
    
    cell_data = nucleus_data + cyto_data
    layers = {"nuc":nucleus_data}

    return cell_data, cells_ids, layers, None


def merge_cells(cells_only,  adata, additional_obs)    :
    additional_obs += ["Cell_ID"]
    additional_obs = list(set(additional_obs))
    additional_obs = cells_only.columns[cells_only.columns.isin(additional_obs)]
    adata.obs = adata.obs.join(cells_only[additional_obs],how="left", on="Cell_ID")
    