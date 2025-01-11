# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from scipy.sparse import lil_matrix, csr_matrix


# def new_adata(vizium_instance, aggregate_by, aggregation_func, 
#               columns=None, custom_agg=None, additional_obs=None, *aggregation_params):
#     adata = vizium_instance.adata
#     if columns is None: 
#         columns = []
#     for col in ["pxl_row_in_fullres", "pxl_col_in_fullres", "pxl_col_in_lowres", "pxl_row_in_lowres",
#                 "pxl_col_in_highres", "pxl_row_in_highres", "um_x", "um_y", "nUMI"]:
#         if col not in columns:
#             columns += [col]
#     meta_df = _aggregate_meta(adata,aggregate_by,columns,custom_agg)
#     if additional_obs:
#         meta_df = meta_df.join(additional_obs, on=aggregate_by, how="left")
#     expr_data, group_ids, layers = aggregation_func(adata, aggregate_by, *aggregation_params)    
#     meta_df = meta_df.reindex(group_ids)    
#     adata_sc = sc.AnnData(X=expr_data, obs=meta_df,var=pd.DataFrame(index=adata.var_names))        
#     if layers:
#         for layer_name, layer_data in layers.items():
#             adata_sc.layers[layer_name] = layer_data
#     adata_sc.obs.rename(columns={"nUMI":"nUMI_spots_avg"},inplace=True)
#     if "nUMI_spots_avg" in adata_sc.obs.columns and "spot_count" in adata_sc.obs.columns:
#         adata_sc.obs["nUMI"] = adata_sc.obs["nUMI_spots_avg"] * adata_sc.obs["spot_count"]
#     return adata_sc

# def _aggregate_meta(adata, aggregate_by, columns, custom_agg):
#     def custom_mode(series):
#         mode_series = series.mode()
#         if not mode_series.empty:
#             return mode_series.iloc[0]
#         return np.nan   
    
#     df = adata.obs[columns + [aggregate_by]].copy()

#     def _guess_aggregator(series):
#         if pd.api.types.is_numeric_dtype(series):
#             return np.median
#         else:
#             return custom_mode

#     if custom_agg is None:
#         custom_agg = {}
#     agg_dict = {}
#     for col in df.columns:
#         if col == aggregate_by:
#             continue
#         if col in custom_agg:
#             agg_dict[col] = custom_agg[col]
#         else:
#             agg_dict[col] = _guess_aggregator(df[col])

#     grouped = df.groupby(aggregate_by, sort=False)
#     group_results = []
#     for group_val, sub_df in tqdm(grouped, total=len(grouped), desc="Aggregating metadata"):
#         row = {}
#         for col_name, funcs in agg_dict.items():
#             if not isinstance(funcs, list):
#                 funcs = [funcs]
#             for func in funcs:
#                 result = func(sub_df[col_name])
#                 # If multiple funcs for the same column, store as colname_funcname
#                 if len(funcs) > 1:
#                     func_name = getattr(func, "__name__", "func")
#                     row[f"{col_name}_{func_name}"] = result
#                 else:
#                     row[col_name] = result
                    
#         row['spot_count'] = len(sub_df) # Add spots count in each group
#         # Keep track of which group (index)
#         row[aggregate_by] = group_val
#         group_results.append(row)
#     updated_obs = pd.DataFrame(group_results).set_index(aggregate_by)
#     return updated_obs

# def _aggregate_data_annotations(adata, group_col):
#     obs = adata.obs
#     unique_groups = obs[group_col].dropna().unique()  
#     group_to_indices = {}
#     for i, val in enumerate(obs[group_col]):
#         if pd.isna(val):
#             continue  
#         group_to_indices.setdefault(val, []).append(i)
#     n_genes = adata.shape[1]
#     n_groups = len(unique_groups)
#     out_matrix = lil_matrix((n_groups, n_genes), dtype=adata.X.dtype)
#     for idx, gval in enumerate(tqdm(unique_groups, desc="Aggregating expression")):
#         row_indices = group_to_indices[gval]
#         group_sum = adata.X[row_indices].sum(axis=0)
#         group_sum = np.ravel(group_sum)
#         out_matrix[idx, :] = group_sum

#     return out_matrix.tocsr(), unique_groups, None







def new_adata(vizium_instance,aggregate_by,aggregation_func,columns=None,
    custom_agg=None,additional_obs=None,*aggregation_params):
    """
    Creates a new AnnData (adata_sc) by aggregating both expression and metadata
    from an existing AnnData (vizium_instance.adata).
    
    :param vizium_instance:  An object holding an AnnData at `.adata`.
    :param aggregate_by:     Column in obs by which to group spots.
    :param aggregation_func: Function that aggregates expression data (X, layers).
    :param columns:          List of obs columns to aggregate (defaults to coords, etc.).
    :param custom_agg:       Dict specifying custom aggregators for specific columns
                             (e.g., {"um_x": np.mean, "um_y": np.mean}).
    :param additional_obs:   Optional DataFrame to join on aggregate_by.
    :param aggregation_params: Extra arguments for the aggregation_func.
    :return:                 Aggregated AnnData object.
    """
    adata = vizium_instance.adata

    default_cols = ["pxl_row_in_fullres", "pxl_col_in_fullres",
        "pxl_col_in_lowres",  "pxl_row_in_lowres",
        "pxl_col_in_highres", "pxl_row_in_highres",
        "um_x", "um_y", "nUMI"]
    
    if columns is None:
        columns = default_cols
    else:
        for col in default_cols:
            if col not in columns:
                columns.append(col)

    meta_df, meta_ids = _aggregate_meta(adata=adata,aggregate_by=aggregate_by,
                        columns=columns,custom_agg=custom_agg)
    
    # Optionally join additional_obs (e.g., cell size etc)
    if additional_obs is not None:
        meta_df = meta_df.join(additional_obs, on=aggregate_by, how="left")

    expr_data, expr_ids, layers = aggregation_func(adata, aggregate_by, *aggregation_params)

    meta_df = meta_df.reindex(expr_ids)

    adata_sc = sc.AnnData(X=expr_data,obs=meta_df,var=pd.DataFrame(index=adata.var_names))

    if layers:
        for layer_name, layer_data in layers.items():
            adata_sc.layers[layer_name] = layer_data

    adata_sc.obs.rename(columns={"nUMI": "nUMI_spots_avg"}, inplace=True)
    if "nUMI_spots_avg" in adata_sc.obs and "spot_count" in adata_sc.obs:
        adata_sc.obs["nUMI"] = adata_sc.obs["nUMI_spots_avg"] * adata_sc.obs["spot_count"]

    return adata_sc


def _aggregate_meta(adata, aggregate_by, columns, custom_agg=None):
    def custom_mode(series):
        mode_series = series.mode()
        if not mode_series.empty:
            return mode_series.iloc[0]
        return np.nan
    
    def _guess_aggregator(series):
        if pd.api.types.is_numeric_dtype(series):
            return np.mean
        else:
            return custom_mode
        
    if custom_agg is None:
        custom_agg = {}
    df = adata.obs[columns + [aggregate_by]].copy()
    agg_dict = {}
    for col in df.columns:
        if col == aggregate_by:
            continue
        if col in custom_agg:
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


def _aggregate_data_annotations(adata, group_col):
    obs = adata.obs
    unique_groups = obs[group_col].dropna().unique()
    group_to_indices = {}
    for i, val in enumerate(obs[group_col]):
        if pd.isna(val):
            continue
        group_to_indices.setdefault(val, []).append(i)
    n_genes = adata.shape[1]
    n_groups = len(unique_groups)
    out_matrix = lil_matrix((n_groups, n_genes), dtype=adata.X.dtype)
    for idx, gval in enumerate(tqdm(unique_groups, desc="Aggregating expression")):
        row_indices = group_to_indices[gval]
        group_sum = adata.X[row_indices].sum(axis=0)
        # group_sum = np.ravel(group_sum)
        out_matrix[idx, :] = group_sum
    return out_matrix.tocsr(), unique_groups, None

# m1_sub = m1[(m1.adata.obs.um_x<1000) & (m1.adata.obs.um_y>5000),:]
# m1_sub.adata.obs["lipid_id_c"] = m1_sub.adata.obs["lipid_id"].copy()

# adata_sc = new_adata(m1_sub, "lipid_id", _aggregate_data_annotations, 
#               columns=["lipid_id_c"], custom_agg=None, additional_obs=None)

# m1_sub.SC = ViziumHD_sc_class.SingleCell(m1_sub,adata_sc)
# m1_sub.SC.plot.spatial("lipid_id_c",size=50,cmap="tab20",legend=False)



# fig, ax = plt.subplots(figsize=(12,12))
# m1_sub.SC.update()
# ax = m1_sub.SC.plot.spatial(image=False,what="lipid_id_c",ax=ax,cmap="tab20",
#                        img_resolution="full",size=200,legend=False)
# m1_sub.plot.spatial("lipid_id_c",legend=False,img_resolution="full",
#                           cmap=["black","black"],alpha=0.5,ax=ax,exact=True)
# plt.show()



