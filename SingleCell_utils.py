# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:49:15 2024

@author: royno
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from scipy.sparse import lil_matrix, csr_matrix


def custom_mode(series):
    mode_series = series.mode()
    if not mode_series.empty:
        return mode_series.iloc[0]
    return np.nan   

def _aggregate_spots_cells(adata, sc_metadata, columns, custom_agg=None):
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
    _aggregate_meta_cells(adata, cells_only, user_aggregations=custom_agg)
    
    adata_sc = _aggregate_data_cells(cells_only, adata)
    return adata_sc

def _split_name(spots_only, adata):
    print("[Splitting name column]")
    split_names = spots_only['Name'].str.split('__', n=1, expand=True)
    split_names.columns = ['Spot_ID', 'Cell_ID']  # Assign column names
    # Replace empty second parts with NaN
    split_names['Cell_ID'] = split_names['Cell_ID'].fillna(value=np.nan)
    spots_only[['Spot_ID', 'Cell_ID']] = split_names
    spots_only = spots_only.set_index("Spot_ID")
    del spots_only['Name']
    adata.obs['Spot_ID'] = adata.obs.index
    print("[Adding cells to the spots-adata]")
    adata.obs = adata.obs.join(spots_only,how='left')

def _aggregate_meta_cells(adata, cells_only, user_aggregations=None):
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
            row_result['spot_count'] = len(group)
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

def _aggregate_data_cells(cells_only, adata, output_dir=None, name=''):
    if 'Cell_ID' in cells_only.columns:
        cells_only.set_index('Cell_ID', inplace=True)
    adata_filtered = adata[(adata.obs['in_cell'] == 1)]
    
    # Split into nucleus/cytoplasm subsets
    print("[Splitting data to nuc/cyto]")
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
    print("[Converting to sparse matrices and creating anndata]")
    nucleus_data = nucleus_data.tocsr()
    cyto_data = cyto_data.tocsr()
    cell_data = nucleus_data + cyto_data

    # Create anndata
    obs_df = cells_only.loc[cells_ids].copy()  # ensures same order as in the loop above
    adata_sc = sc.AnnData(X=cell_data,obs=obs_df,var=adata_filtered.var)
    adata_sc.layers["nuc"] = nucleus_data
    adata_sc.layers["cyto"] = cyto_data

    return adata_sc

def _aggregate_spots_annotations(adata, group_col="lipid_id", columns=None, custom_agg=None):
    """
    Aggregate an AnnData object based on a column in `adata.obs`.
    E.g., group by 'villus_id' and create one row per unique villus.

    Parameters
    ----------
    adata : AnnData
        Original AnnData with spot-level data in .X and metadata in .obs.
    group_col : str
        The column in adata.obs used to define groups (default: 'villus_id').
    columns : list or None
        Which columns in adata.obs to aggregate. Defaults to empty list.
    custom_agg : dict or None
        Optional dict specifying custom aggregation for certain columns.
        E.g. { "columnA": np.mean, "columnB": lambda x: ','.join(x) }

    Returns
    -------
    AnnData
        A new AnnData with aggregated expression and aggregated metadata.
        Each row (obs) corresponds to one unique value of `group_col`.
    """
    if columns is None:
        columns = []
    if not isinstance(columns, list):
        raise ValueError("`columns` must be a list of column names.")
    if group_col not in adata.obs.columns:
        raise ValueError(f"'{group_col}' not found in adata.obs columns.")
    if group_col not in columns:
        columns.append(group_col)
        

    # 2) Aggregate metadata (returns a DataFrame with one row per group_col)
    meta_df = _aggregate_meta_annotations(adata=adata,group_col=group_col,
        columns=columns,user_aggregations=custom_agg)

    # 3) Aggregate expression data (returns a 2D array or sparse matrix + list of group IDs)
    expr_data, group_ids = _aggregate_data_annotations(adata=adata,group_col=group_col)

    # 4) Build a new AnnData
    #    - obs = aggregated metadata (indexed by the same group IDs order)
    #    - X   = aggregated expression data
    #    - var = original genes
    aggregated_obs = meta_df.reindex(group_ids)  # ensure same order
    new_adata = sc.AnnData(X=expr_data, obs=aggregated_obs,var=adata.var.copy())

    return new_adata

def _aggregate_meta_annotations(adata, group_col, columns, user_aggregations=None):
    """
    Group `adata.obs` by `group_col` and compute aggregated metadata for each group.

    Parameters
    ----------
    adata : AnnData
        Original AnnData with metadata in .obs.
    group_col : str
        The column used to define groups.
    columns : list
        Which columns to aggregate (must include group_col).
    user_aggregations : dict or None
        A mapping of {col_name: aggregator_function or [functions]}.
        If not provided, numeric columns default to median, others to custom_mode.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by the unique values of `group_col`.
        Each row is the aggregated result for that group.
    """
    # Subset the obs to relevant columns
    df = adata.obs[columns].copy()

    # Prepare final aggregator dictionary
    def _guess_aggregator(series):
        """Default aggregator for columns without a user-specified aggregator."""
        if pd.api.types.is_numeric_dtype(series):
            return np.median
        else:
            return custom_mode

    if user_aggregations is None:
        user_aggregations = {}
    agg_dict = {}
    for col in df.columns:
        if col == group_col:
            # We won't apply aggregator to the grouping column itself
            continue
        if col in user_aggregations:
            agg_dict[col] = user_aggregations[col]
        else:
            agg_dict[col] = _guess_aggregator(df[col])

    # Group by group_col
    grouped = df.groupby(group_col)

    # Apply aggregations to each group with a progress bar
    group_results = []
    for group_val, sub_df in tqdm(grouped, total=len(grouped), desc="Aggregating metadata"):
        row = {}
        for col_name, funcs in agg_dict.items():
            if not isinstance(funcs, list):
                funcs = [funcs]
            for func in funcs:
                result = func(sub_df[col_name])
                # If multiple funcs for the same column, store as colname_funcname
                if len(funcs) > 1:
                    func_name = getattr(func, "__name__", "func")
                    row[f"{col_name}_{func_name}"] = result
                else:
                    row[col_name] = result
                    
        row['spot_count'] = len(sub_df)

        # Keep track of which group (index)
        row[group_col] = group_val
        group_results.append(row)

    aggregated_df = pd.DataFrame(group_results).set_index(group_col)
    return aggregated_df

def _aggregate_data_annotations(adata, group_col):
    """
    Aggregate the expression data in `adata.X` by summing (or otherwise combining)
    all spots/rows that share the same value of `group_col`.

    Parameters
    ----------
    adata : AnnData
        The AnnData with spot-level expression in .X and grouping info in .obs.
    group_col : str
        The column in adata.obs used to define groups.

    Returns
    -------
    (X_agg, group_ids) : (csr_matrix or np.ndarray, list-like)
        X_agg is the aggregated expression matrix with shape [n_groups, n_genes].
        group_ids is the list of group_col values in the order used by the rows of X_agg.
    """
    # 1) Find the unique groups & row indices
    #    We'll build a mapping of {group_val -> list of row indices}
    obs = adata.obs
    unique_groups = obs[group_col].dropna().unique()  # skip NaNs if needed
    group_to_indices = {}
    for i, val in enumerate(obs[group_col]):
        if pd.isna(val):
            continue  
        group_to_indices.setdefault(val, []).append(i)

    # 2) Summation across rows for each group
    #    We'll create a new matrix with shape [n_groups, n_genes]
    n_genes = adata.shape[1]
    n_groups = len(unique_groups)
    # Using a sparse matrix in case adata.X is large
    out_matrix = csr_matrix((n_groups, n_genes), dtype=adata.X.dtype)

    # If adata.X is sparse, we can sum slices directly; if it's dense, convert or handle carefully
    # We'll do a row-by-row sum here with a progress bar
    for idx, gval in enumerate(tqdm(unique_groups, desc="Aggregating expression")):
        row_indices = group_to_indices[gval]
        # Sum across all rows in that group
        # If adata.X is sparse, sum(axis=0) stays sparse. Otherwise, we might want to convert to dense
        group_sum = adata.X[row_indices].sum(axis=0)
        # group_sum might be a matrix; ensure it's 1D
        out_matrix[idx, :] = group_sum

    # 3) Return the aggregated matrix + the list of group IDs
    return out_matrix, unique_groups









