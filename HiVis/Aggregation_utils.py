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
    

def split_spots_cells(input_df):
    df = input_df.copy()
    print("[Splitting name column and merging metadata]")
    split_names = df['Name'].str.split('__', n=1, expand=True)
    split_names.columns = ['Spot_ID', 'Cell_ID']
    split_names['Cell_ID'] = split_names['Cell_ID'].fillna(value=np.nan)
    
    # Assign spots with more than one cell to the first one.
    split_names['Cell_ID'] = split_names['Cell_ID'].apply(
        lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else x
    )
    
    df[['Spot_ID', 'Cell_ID']] = split_names
    cols = ['Cell_ID', 'Spot_ID']
    cols = ['Cell_ID', 'Spot_ID','InNuc']
    spots_only = df.loc[input_df['Object type']=='Tile',cols]
    spots_only = spots_only.set_index("Spot_ID")
        
    cells_only = input_df.loc[input_df['Object type']!='Tile']
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


def _aggregate_data_cells(adata, group_col="Cell_ID"):
    '''
    Aggregates expression data for all spots inside each cell,
    disregarding whether a spot is in the nucleus or cytoplasm.
    '''

    # Group the spots by cell id
    ind_dict = adata.obs.groupby(by=[group_col]).indices
    cells_ids = list(ind_dict.keys())
    num_cells = len(cells_ids)
    num_genes = adata.shape[1]
    
    # Preallocate a sparse matrix for the aggregated cell data
    cell_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    
    # Sum all spots for each cell
    for i, cell in enumerate(tqdm(cells_ids, desc='Aggregating spots expression')):
        cell_data[i, :] = adata[ind_dict[cell], :].X.sum(axis=0)
    
    cell_data = cell_data.tocsr()
    
    layers = {}
    
    return cell_data, cells_ids, layers, None



def _aggregate_data_annotations(adata, group_col):
    '''
    Aggregates expression data for all spots inside each annotation,
    '''
    # Group the spots by group_col
    ind_dict = adata.obs.groupby(by=[group_col]).indices
    cells_ids = list(ind_dict.keys())
    num_cells = len(cells_ids)
    num_genes = adata.shape[1]
    
    # Preallocate a sparse matrix for the aggregated cell data
    cell_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    
    # Sum all spots for each cell
    for i, cell in enumerate(tqdm(cells_ids, desc='Aggregating spots expression')):
        cell_data[i, :] = adata[ind_dict[cell], :].X.sum(axis=0)
    
    cell_data = cell_data.tocsr()
    
    layers = {}
    
    return cell_data, cells_ids, layers, None



def merge_cells(cells_only,  adata, additional_obs, id_col="Cell_ID"):
    # additional_obs = list(set(additional_obs + [id_col]))
    
    additional_obs = [c for c in additional_obs if c in cells_only.columns]
    if id_col in cells_only.columns:
        cells_only = cells_only.set_index(id_col)
    adata.obs = adata.obs.join(cells_only[additional_obs],how="left", on=id_col)
    

# def add_spatial_keys(hivis_obj, adata, name):
#     """
#     Adds spatial keys to the AnnData object to make it Scanpy/Squidpy spatial plot compatible.
    
#     Parameters:
#         * hivis_obj (HiVis) - that has images and scalefactors json
#         * adata (AnnData) - AnnData object to which spatial keys will be added.
#         * name (str) - name of adata, will be concatinated to hivis_obj.name
    
#     **Returns:** The updated AnnData object.
#     """
#     required_cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
#     if not all(col in adata.obs.columns for col in required_cols):
#         raise ValueError("Missing required spatial coordinate columns in adata.obs")
    
#     adata.obsm["spatial"] = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
        
#     adata.uns["spatial"] = {
#         name: {"images": {"hires": hivis_obj.image_highres,"lowres": hivis_obj.image_lowres},
#             "scalefactors": hivis_obj.json,
#             "metadata": hivis_obj.properties}}
    
#     return adata
