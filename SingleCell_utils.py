# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from scipy.sparse import lil_matrix, csr_matrix
from scipy.stats import multinomial
import ViziumHD_utils

def new_adata(vizium_instance,aggregate_by,aggregation_func,columns=None,
    custom_agg=None,additional_obs=None,**aggregation_kwargs):
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
    # if additional_obs:
    #     meta_df = meta_df.join(additional_obs, on=aggregate_by, how="left")

    expr_data, expr_ids, layers, other_results = aggregation_func(adata, group_col_cell=aggregate_by,**aggregation_kwargs)

    meta_df = meta_df.reindex(expr_ids)

    adata_sc = sc.AnnData(X=expr_data,obs=meta_df,var=pd.DataFrame(index=adata.var_names))

    if layers:
        for layer_name, layer_data in layers.items():
            adata_sc.layers[layer_name] = layer_data

    adata_sc.obs.rename(columns={"nUMI": "nUMI_spots_avg"}, inplace=True)
    if "nUMI_spots_avg" in adata_sc.obs and "spot_count" in adata_sc.obs:
        adata_sc.obs["nUMI"] = adata_sc.obs["nUMI_spots_avg"] * adata_sc.obs["spot_count"]

    return adata_sc, other_results


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
    return out_matrix.tocsr(), unique_groups, None, None



def _aggregate_data_stardist(adata, group_col="Cell_ID", in_cell_col="in_cell",nuc_col="in_nucleus"):
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
    layers = {"nuc":nucleus_data, "cyto":cyto_data}

    return cell_data, cells_ids, layers, None




def _aggregate_data_two_nuclei(adata, cells_nuc, 
    group_col_cell="Cell_ID",group_col_nuc="Nuc_ID",in_cell_col="InCell",
    nuc_col="InNuc"):
    
    # Filter only spots that are inside a cell
    adata_filtered = adata[adata.obs[in_cell_col] == 1].copy()
    
    # Subset: nucleus spots vs cytoplasm spots
    adata_nuc = adata_filtered[adata_filtered.obs[nuc_col] == 1].copy()
    adata_cyto = adata_filtered[adata_filtered.obs[nuc_col] == 0].copy()
    
    # Build dictionary of indices, but now grouped by Nuc_ID for nuclear subset
    nuc_dict = adata_nuc.obs.groupby(group_col_nuc).indices
    cyto_dict = adata_cyto.obs.groupby(group_col_cell).indices
    
    # Determine cell IDs we want to process
    cells_ids = cells_nuc.index.tolist()
    num_cells = len(cells_ids)
    num_genes = adata_filtered.shape[1]
    
    # Allocate sparse matrices
    nucleus_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    cyto_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    
    # Preallocate double nucleated nucs and cells for analysis
    double_nuc_cells = cells_nuc.index[~cells_nuc["nuc2"].isna() & 
                                       cells_nuc["nuc2"].isin(nuc_dict.keys())]
    num_nucs_in_2nuc_cells = len(double_nuc_cells) * 2
    nucs_2nuc_cells = lil_matrix((num_nucs_in_2nuc_cells, num_genes), dtype=np.float32)
    cells_2nucs_list = [None for _ in range(num_nucs_in_2nuc_cells)]
    nucs_2nucs_list = [None for _ in range(num_nucs_in_2nuc_cells)]
    nuc_index = 0
    
    for i, cell in enumerate(tqdm(cells_ids, desc='Aggregating spots expression')):
        # 1) Aggregate cytoplasm for this cell 
        if cell in cyto_dict:
            cyto_data[i, :] = adata_cyto[cyto_dict[cell], :].X.sum(axis=0)
        
        # 2) Handle nucleus data
        nuc1_id = cells_nuc.loc[cell, "nuc1"]
        nuc2_id = cells_nuc.loc[cell, "nuc2"]
        
        # Single nucleus
        if pd.notna(nuc1_id) and (nuc1_id in nuc_dict):
            nuc1_expr = adata_nuc[nuc_dict[nuc1_id], :].X.sum(axis=0)
        
        # No nuclei
        else: 
            nuc1_expr = 0
        
        # Two nuclei
        if pd.notna(nuc2_id) and (nuc2_id in nuc_dict):
            nuc2_expr = adata_nuc[nuc_dict[nuc2_id], :].X.sum(axis=0)
            
            # Add both nuclei for 2nuc analysis
            nucs_2nucs_list[nuc_index] = nuc1_id
            cells_2nucs_list[nuc_index] = cell
            nucs_2nuc_cells[nuc_index,:] = nuc1_expr
            nuc_index += 1
            nucs_2nucs_list[nuc_index] = nuc2_id
            cells_2nucs_list[nuc_index] = cell
            nucs_2nuc_cells[nuc_index,:] = nuc2_expr
            nuc_index += 1
        else:
            nuc2_expr = 0
        # Total nucleus expression is expression in both nuclei
        nucleus_data[i, :] = nuc1_expr + nuc2_expr
    
    # Calculate which genes are enriched in one nuclei out of both
    genes = np.array(adata.var_names)
    df = pd.DataFrame(nucs_2nuc_cells.tocsr().toarray(),index=nucs_2nucs_list,columns=genes)
    df["Cell_ID"] = cells_2nucs_list
    nuc_cell_dict = df.groupby("Cell_ID").indices
    del df["Cell_ID"]
    
    
    cyto_data = cyto_data.tocsr()
    nucleus_data = nucleus_data.tocsr()
    cell_data = nucleus_data + cyto_data
    layers = {"nuc":nucleus_data, "cyto": cyto_data}
    
    return cell_data, cells_ids, layers, {"nuc_by_genes":df,"nuc_cell_dict":nuc_cell_dict}



def adata_to_nuc_by_gene(adata):
    
    
    
    
    return df


def processcells_2nucs(df, nuc_cell_dict, n_perm=10, func=np.mean):
    def calculate_val(nuc1, nuc2):
        nucs_ratio = (nuc1 + 1) / (nuc2 + 1)  # Add pseudonumber and calculate ratio
        return np.abs(np.log2(nucs_ratio))  
    
    genes = df.columns
    n_genes = len(genes)
    n_cells = len(nuc_cell_dict)
    
    # Calculate abs(log2(nuc1/nuc2)) for each gene
    real_result = np.full((n_cells, n_genes), np.nan)
    expression = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    for ind_cell, (cell, (nuc1_ind, nuc2_ind)) in enumerate(nuc_cell_dict.items()):
        counts_nuc1 = df.iloc[nuc1_ind] 
        counts_nuc2 = df.iloc[nuc2_ind]
        real_result[ind_cell] = calculate_val(counts_nuc1, counts_nuc2)
        expression[ind_cell] = (counts_nuc1 + counts_nuc2) / 2
        
    # Get single statistic
    real_value = func(real_result, axis=0) 
    
    n_reads_genes = df.sum(axis=0).values  # total reads for each gene
    n_reads_nucs = df.sum(axis=1).values  # total reads for each nucleus
    weights = n_reads_nucs / n_reads_nucs.sum()     
    
    # Build random multinomial distribution
    simulations = np.zeros((n_perm, n_genes), dtype=np.float32)
    for i in tqdm(range(n_perm),desc="Running simulations with multinomial distribution"):
        random_matrix = np.zeros((df.shape[0], df.shape[1]), dtype=int)
        
        # For each gene, sample from a multinomial
        for gene_idx in range(n_genes):
            gene_exp = n_reads_genes[gene_idx]
            if gene_exp > 0:
                # Distribute gene_exp reads across the rows (nuclei)
                random_counts = multinomial.rvs(n=gene_exp, p=weights)
                random_matrix[:, gene_idx] = random_counts
        
        # For each "cell" (pair of nuclei), compute ratio again
        rand_result = np.full((n_cells, n_genes), np.nan)
        for ind_cell, (cell, (nuc1_idx, nuc2_idx)) in enumerate(nuc_cell_dict.items()):
            rand_nuc1 = random_matrix[nuc1_ind]
            rand_nuc2 = random_matrix[nuc2_ind]
            rand_result[ind_cell] = calculate_val(rand_nuc1, rand_nuc2)
        simulations[i,:] = func(rand_result, axis=0) 

    above_rand = np.sum(simulations >= real_value, axis=0)
    above_rand[above_rand == 0] = 1  # Pval can't be lower than 1/n_perm
    pvals = above_rand / n_perm
    
    results = pd.Series(real_value, index=genes, name="statistic")
    results = pd.DataFrame(results)
    results["pval"] = pvals
    results["qval"] = ViziumHD_utils.p_adjust(results["pval"])
    mean_expr = df.mean(axis=0)  # Series with the same index as 'genes'
    results["expression_mean"] = mean_expr.values
    
    return results

# def _aggregate_data_liver(adata, cells_nuc, group_col="Cell_ID", in_cell_col="in_cell",
#                           nuc_col="in_nucleus"):
#     cells_wo_nuc = set(cells_nuc.index[cells_nuc["nuc1"].isna()])
#     cells2nuc = set(cells_nuc.index[~cells_nuc["nuc2"].isna()])
#     cells_with_nuc = set(cells_nuc.index) - (cells_wo_nuc | cells2nuc)

#     nuc1_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
#     nuc2_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
#     nucs_ratio = lil_matrix((num_cells, num_genes), dtype=np.float32)
#     cyto_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    
#     for i, cell in enumerate(tqdm(cells_ids, desc='Aggregating spots expression')): 
#         cyto_data[i, :] = adata_cyto[ind_dict_cyto[cell],:].X.sum(axis=0) 
#         if cell in cells_with_nuc:
#             nuc1 = adata_nuc[ind_dict_nuc[cell],:].X.sum(axis=0)
#             nuc1_data[i, :] = nuc1
#             if cell in cells2nuc:
#                 nuc2 = adata_nuc[ind_dict_nuc[cell],:].X.sum(axis=0)
#                 nuc2_data[i, :] = nuc2
#                 ratio = nuc1 + 1 / nuc2 + 1 # ratio and pseudonumber
#                 nucs_ratio[i, :] = np.abs(np.log2(ratio))
#         else: # No nucleus
#             pass
            
#     print("[Converting to sparse matrices]")
#     nuc1_data = nuc1_data.tocsr()
#     nuc2_data = nuc2_data.tocsr()
    
#     nucleus_data = nuc1_data + nuc2_data
#     cyto_data = cyto_data.tocsr()
    
#     cell_data = nucleus_data + cyto_data
#     layers = {"nuc":nucleus_data, "cyto":cyto_data}

#     return cell_data, cells_ids, layers

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



    

