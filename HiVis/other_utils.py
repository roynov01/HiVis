# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:45:31 2025

@author: royno
"""
import pandas as pd
import numpy as np

def matnorm(df, axis="col"):
    '''
    Normalizes a dataframe or matrix by the sum of columns or rows.
    
    Parameters:
    - df: np.ndarray, sparse matrix, or pandas DataFrame
    - axis: "col" for column-wise normalization, "row" for row-wise normalization
    
    Returns:
    - Normalized matrix of the same type as input
    '''
    if isinstance(df, pd.Series):
        return df.div(df.sum())

    if isinstance(df, (np.ndarray, np.matrix)):
        df = np.asarray(df)  # Convert matrix to array if needed
        axis_num = 1 if axis == "row" else 0
        sums = df.sum(axis=axis_num, keepdims=True)
        sums[sums == 0] = 1  # Avoid division by zero
        return df / sums

    if isinstance(df, pd.DataFrame):
        if axis == "row":
            row_sums = df.sum(axis=1).to_numpy().reshape(-1, 1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return df.div(row_sums, axis=0).astype(np.float32)
        else:
            col_sums = df.sum(axis=0)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            return df.div(col_sums, axis=1).astype(np.float32)
    import scipy.sparse as sp

    if sp.isspmatrix_csr(df):
        if axis == "row":
            row_sums = np.array(df.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            diag_inv = sp.diags(1 / row_sums)
            return diag_inv.dot(df)  # Normalize rows
        else:
            col_sums = np.array(df.sum(axis=0)).ravel()
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            diag_inv = sp.diags(1 / col_sums)
            return df.dot(diag_inv)  # Normalize columns

    raise ValueError("df is not a supported type (list, numpy array, sparse matrix, or dataframe)")


def find_markers(exp_df, celltypes=None, ratio_thresh=2, exp_thresh=0,
                 chosen_fun="max",other_fun="max",ignore=None):
    '''
    Finds markers of celltype/s based on signature matrix:
        * exp_df - dataframe, index are genes, columns are celltypes
        * celltypes (str or list) - column name/names of the chosen celltype/s
        * ratio_thresh - ratio is chosen/other 
        * exp_thresh - process genes that are expressed above X in the chosen celltype/s
        * chosen_fun, other_fun - either "mean" or "max"
        * ignore (list) - list of celltypes to ignore in the "other" group
    '''
    if "gene" in exp_df.columns:
        exp_df.index = exp_df["gene"]
        del exp_df["gene"]
    if celltypes is None:
        print(exp_df.columns)
        return
    exp_df = matnorm(exp_df)
    chosen = exp_df[celltypes]
    if isinstance(celltypes, str):
        celltypes = [celltypes]
    other_names = exp_df.columns[~exp_df.columns.isin(celltypes)]
    if chosen_fun == "max":
        chosen = chosen.max(axis=1)
    elif chosen_fun == "mean":
        chosen = chosen.mean(axis=1)

    if ignore:
        other_names = [name for name in other_names if name not in ignore]
    other = exp_df[other_names]
    if other_fun == "max":
        other = other.max(axis=1)
    elif other_fun == "mean":
        other = other.mean(axis=1)
    
    pn = chosen[chosen>0].min()
    markers_df = pd.DataFrame({"chosen_cell":chosen,"other":other},index=exp_df.index.copy())
    markers_df = markers_df.loc[markers_df["chosen_cell"] >= exp_thresh]
    markers_df["ratio"] = (chosen+pn) / (other+pn)
    markers_df["gene"] = markers_df.index
    genes = markers_df.index[markers_df["ratio"] >= ratio_thresh].tolist()

    return genes, markers_df


def fix_excel_gene_dates(df, handle_duplicates="mean"):
    """
    Fixes gene names in a DataFrame that Excel auto-converted to dates.
        * df - DataFrame containing gene names either in a column named "gene" or in the index.
        * handle_duplicates (str): How to handle duplicates after conversion. Options: "mean" or "first".
    """
    date_to_gene = {
        "1-Mar": "MARCH1", "2-Mar": "MARCH2", "3-Mar": "MARCH3", "4-Mar": "MARCH4", "5-Mar": "MARCH5",
        "6-Mar": "MARCH6", "7-Mar": "MARCH7", "8-Mar": "MARCH8", "9-Mar": "MARCH9",
        "1-Sep": "SEPT1", "2-Sep": "SEPT2", "3-Sep": "SEPT3", "4-Sep": "SEPT4", "5-Sep": "SEPT5",
        "6-Sep": "SEPT6", "7-Sep": "SEPT7", "8-Sep": "SEPT8", "9-Sep": "SEPT9",
        "10-Sep": "SEPT10", "11-Sep": "SEPT11", "12-Sep": "SEPT12", "15-Sep": "SEPT15",
        "10-Mar": "MARCH10", "11-Mar": "MARCH11"
    }  
    
    if 'gene' in df.columns:
        df['gene'] = df['gene'].replace(date_to_gene)  # Replace values in 'gene' column
    else:
        df.index = df.index.to_series().replace(date_to_gene)  # Replace values in the index
    if 'gene' in df.columns:
        df = df.set_index('gene')
    if handle_duplicates == "mean":
        df = df.groupby(df.index).mean()
    elif handle_duplicates == "first":
        df = df[~df.index.duplicated(keep='first')]
    if 'gene' in df.columns or 'gene' in df.index.names:
        df = df.reset_index()
    
    return df


def inspect_df(df, col, n_rows=2):
    '''samples a df and return few rows (n_rows)
    from each unique value of col'''
    subset_df = df.groupby(col).apply(lambda x: x.sample(min(len(x), n_rows))).reset_index(drop=True)
    return subset_df



def merge_geojsons(geojsons_files, filename_out):
    '''
    Combine geopandas to one file.
    parameters:
        * geojsons_files - list of file paths
        * filename_out - name of the combined file. ends with .shp
    '''
    import geopandas as gpd
    gdfs = [gpd.read_file(file) for file in geojsons_files]
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry')
    if not filename_out.endswith(".shp"):
        filename_out += ".shp"
    combined_gdf.to_file(filename_out, driver="GPKG")

def pca(df, k_means=None, first_pc=1, title="PCA", number_of_genes=20):
    """
    Performs PCA on a dataframe, optionally applies k-means clustering, and generates plots.

    Parameters:
    - df: DataFrame with genes as rows and samples as columns.
    - k_means: Number of clusters for k-means clustering. If None, clustering is not performed.
    - first_pc: The first principal component to display.
    - title: Title for the PCA plot.
    - number_of_genes: Number of variable genes to plot.

    Returns:
    - A dictionary with PCA plot, elbow plot, PCA DataFrame, variance explained,
      and silhouette scores (if k_means is provided).
    """
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import numpy as np
    
    if first_pc >= df.shape[1]:
        raise ValueError(f"No PC {first_pc +1} possible in a data of {df.shape[1]} samples")

    # Filter out genes with zero expression
    df_filtered = df.loc[df.sum(axis=1) > 0]

    # Perform PCA
    pca_object = PCA()
    pca_result = pca_object.fit_transform(df_filtered.T)

    var_explained = pca_object.explained_variance_ratio_
    elbow_data = pd.DataFrame({
        'PC': np.arange(1, len(var_explained) + 1),
        'Variance': var_explained
    })

    pca_data = pd.DataFrame(pca_result, columns=[f'PC{ i +1}' for i in range(pca_result.shape[1])])
    pca_data['sample'] = df.columns

    x = f'PC{first_pc}'
    y = f'PC{first_pc + 1}'
    xlab = f'PC {first_pc} ({var_explained[first_pc - 1] * 100:.2f}%)'
    ylab = f'PC {first_pc + 1} ({var_explained[first_pc] * 100:.2f}%)'

    # K-means clustering
    if k_means is not None:
        if k_means >= pca_data.shape[0]:
            raise ValueError("k_means needs to be lower than number of samples")
        kmeans = KMeans(n_clusters=k_means, random_state=0).fit(pca_data.iloc[:, first_pc - 1: first_pc + 1])
        pca_data['cluster'] = kmeans.labels_.astype(str)
        silhouette_avg = silhouette_score(pca_data.iloc[:, first_pc - 1:first_pc + 1], kmeans.labels_)
    else:
        pca_data['cluster'] = "1"
        silhouette_avg = None

    # Plot PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_data, x=x, y=y, hue='cluster', palette='Set1', s=100)
    for i in range(pca_data.shape[0]):
        plt.text(pca_data.loc[i, x], pca_data.loc[i, y], pca_data.loc[i, 'sample'])
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if silhouette_avg is not None:
        # plt.suptitle(f'Silhouette mean score: {silhouette_avg:.2f}')
        plt.text(0.5, -0.1, f'Silhouette mean score: {silhouette_avg:.2f}', ha='center', va='center',
                 transform=plt.gca().transAxes)
    pca_plot = plt.gcf()

    # Plot Elbow
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=elbow_data, x='PC', y='Variance', marker='o')
    plt.title('Elbow Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    elbow_plot = plt.gcf()

    # Variable genes plot
    var_genes = pd.DataFrame(pca_object.components_.T, index=df_filtered.index,
                             columns=[f'PC{ i +1}' for i in range(pca_result.shape[1])])

    def plot_variable_genes(pc, num_genes):
        df_pc = var_genes[[pc]].sort_values(by=pc)
        top_bottom_genes = pd.concat([df_pc.head(num_genes), df_pc.tail(num_genes)])
        plt.figure(figsize=(10, 7))
        sns.barplot(x=top_bottom_genes.index, y=top_bottom_genes[pc], color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='black')
        plt.title(f'Top and Bottom {num_genes} Genes for {pc}')
        plt.tight_layout()
        return plt.gcf()

    genes1_plot = plot_variable_genes(x, number_of_genes)
    genes2_plot = plot_variable_genes(y, number_of_genes)

    result = {
        'plot_pca': pca_plot,
        'plot_elbow': elbow_plot,
        'pca_df': pca_data,
        'variance': elbow_data,
        'var_genes': var_genes,
        'plot_genes': [genes1_plot, genes2_plot]
    }

    if silhouette_avg is not None:
        result['silhouette'] = silhouette_avg
    print(f'keys in output: {list(result.keys())}')
    return result



def gsea(gene_list=None, log2fc_ratio=None, geneset="hallmark", organism="mouse", nperm=1000, seed=42):
    '''
    Run GSEA (prerank) on list of genes 
    
    Parameters:
        * geneset (str) - can be a .gmt file, or one of:
            "hallmark"
            "kegg"
            "KEGG_2021_Human"
            "GO_Molecular_Function_2025"
            "GO_Cellular_Component_2025"
            "GO_Biological_Process_2025"
            for full list: gp.get_library_name(organism='Human')
        * organism (str) - human or mouse
        * nperm (int) - reduce for speed, increase for accuracy
    '''
    import gseapy as gp
    org = organism.capitalize()
    available = gp.get_library_name(organism=org)
    if (gene_list is None):
        print(f"Available datasets for {org}:")
        _ = [print(i) for i in available]
        return
        
    if len(gene_list) != len(log2fc_ratio):
        raise ValueError("gene_list and log2fc_ratio must be the same length")
    rnk = pd.DataFrame({'gene': gene_list, 'score': list(log2fc_ratio)})
    rnk.dropna(inplace=True)
    rnk.set_index('gene', inplace=True)

    
    lib = geneset
    
    if geneset.lower() == "hallmark":
        lib = "MSigDB_Hallmark_2020"
    elif geneset.lower() == "kegg":
        # this will pick e.g. "KEGG_2019_Rat" if you passed organism="rat"
        lib = f"KEGG_2019_{org}"
    else:
        
        if (lib not in available) or lib.endswith(".gmt"):
            raise ValueError(f"Library '{lib}' not found for organism '{org}'")
    print(f"GSEA for {org}, dataset: {lib}")
    
    gsea_res = gp.prerank(rnk=rnk, gene_sets=lib, outdir=None, permutation_num=nperm, seed=seed)
    return gsea_res


def plot_gsea_dotplot(df, title="GSEA",q_thresh=0.25,
                      ax=None,directions=["upregulated","downregulated"],
                      colors=["red","blue"], grid=False, figsize=(10,6)):
    """Dotplot of GSEA results."""
    import seaborn as sns
    import gseapy as gp
    import matplotlib.pyplot as plt
    
    if isinstance(df,gp.Prerank):
        df = df.res2d
    df = df.copy()
    df = df.loc[df["FDR q-val"] <= q_thresh]
    if df.empty:
        raise ValueError("no entries to plot")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)   



    df = df.sort_values("NES",ascending=False)
    df["Term"] = pd.Categorical(df["Term"], categories=df["Term"], ordered=True)
    df["direction"] = directions[0]
    df.loc[df["NES"]<0,"direction"] = directions[1]
    

    sns.set(style="whitegrid" if grid else "white")
    palette = {directions[0]: colors[0],directions[1]: colors[1]}
    df["Gene %"] = df["Gene %"].str.rstrip('%').astype(float)

    sns.scatterplot(data=df, x="NES", y="Term", hue="direction", size="Gene %",
                    palette={k: palette[k] for k in df["direction"].unique()}, 
                    sizes=(40,400), ax=ax)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel("NES")
    ax.set_ylabel(None)

    ax.grid(grid)
    return ax

def plot_gsea_pathway(gsea_res,pathway=None,q_thresh=0.05, figsize=(3,4),legend_kws=None):
    '''
    Plot pathway(s) hits
    Parameters:
        * gsea_res - results of gsea()
        * pathway - name of pathway or a list of names. If None, will plot all significant, then provide q_thresh
        * legend_kws (dict) - for example {'loc': (1.2, 0)}
    '''
    if pathway is None:
        pathway = gsea_res.res2d.loc[gsea_res.res2d["FDR q-val"] <= q_thresh,"Term"]
    elif isinstance(pathway, str):
        pathway = [pathway]

    ax = gsea_res.plot(terms=pathway,legend_kws=legend_kws,show_ranking=False,figsize=figsize)
    return ax


def plot_dotplot(df, x, y, size_col, val_col,
                 normalize_size=False, normalize_col=False, sort=True, sort_method="sum",
                 ax=None, xlab=None, ylab=None, title=None,max_dot_size=100, 
                 cmap="coolwarm", figsize=(8,16),legend=True, rotate_xticklab=False,
                 legend_col_title=None, legend_size_title=None):
    '''
    Plots a dotplot.
    Parameters:
        * df - dataframe of 4 columns, x, y, size_col, val_col
        * x, y, size_col, val_col - which columns to use
        * normalize_size,normalize_col - whether to normilize each row to the maximal value of the row
        * sort - sort the rows
        * sort_method - if sort is True, how to sort. possible values are "sum","std","mean"
        * ax - matplotlib Axes, if provided
        * figsize, cmap, legend, xlab, ylab, title, legend_col_title, 
        legend_size_title, rotate_xticklab - cosmetic Parameters
    '''
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    color_data = df.pivot(index=y, columns=x, values=val_col)
    size_data  = df.pivot(index=y, columns=x, values=size_col)
    
    if normalize_col:
        color_data = color_data.div(color_data.max(axis=1), axis=0)
    if normalize_size:
        size_data  = size_data.div(size_data.max(axis=1), axis=0)

    if sort:
        if sort_method == "sum":
            color_data["delta"] = color_data.sum(axis=1, skipna=True)
        elif sort_method == "mean":
            color_data["delta"] = color_data.mean(axis=1, skipna=True)
        elif sort_method == "std":
            color_data["delta"] = color_data.std(axis=1, skipna=True)
        else:
            raise ValueError(f"Invalid sort_method: {sort_method}. "
                             "Choose from ['sum','mean','std']")
        color_data = color_data.sort_values("delta", ascending=False)
        size_data  = size_data.loc[color_data.index, :]
        del color_data["delta"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    row_labels, col_labels  = list(color_data.index), list(color_data.columns)

    xvals, yvals, colors, sizes = [], [], [], []
    for i, row_name in enumerate(row_labels):
        for j, col_name in enumerate(col_labels):
            xvals.append(j)
            yvals.append(i)
            c_val = color_data.loc[row_name, col_name]
            colors.append(c_val)
            s_val = size_data.loc[row_name, col_name]
            sizes.append(np.nan_to_num(s_val, nan=0))

    all_sizes_arr = np.array(sizes)
    current_max = np.nanmax(all_sizes_arr)
    if current_max > 0:
        sizes_normed = all_sizes_arr / current_max
    else:
        sizes_normed = all_sizes_arr  # if all zero/NaN

    # scale up to user-requested maximum size
    scatter_sizes = [max_dot_size * s for s in sizes_normed]

    if isinstance(cmap, list):
        cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap)

    sca = ax.scatter(xvals, yvals,c=colors,
        s=scatter_sizes,cmap=cmap,edgecolors="none")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90 if rotate_xticklab else 0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)

    if legend:
        # Shrink main axis to free space on the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])

        # Create a new Axes in top half for colorbar
        cbar_ax = fig.add_axes([box.x0 + box.width*0.85, box.y0 + box.height*0.5, 
            0.03, box.height*0.45])
        cbar = fig.colorbar(sca, cax=cbar_ax)
        if legend_col_title:
            cbar.set_label(legend_col_title)

        if current_max > 0:
            # Example fractions of original data's range
            fraction_values = [0.25, 0.50, 0.75, 1.00]
            # Convert fraction -> actual data scale
            actual_sizes = [fraction * current_max for fraction in fraction_values]
            # Human-readable labels
            size_labels = [f"{v:.2g}" for v in actual_sizes]
            
            # Convert fraction -> scatter circle area
            size_legend_scaled = [max_dot_size * f for f in fraction_values]

            # Make dummy scatter patches to display in the legend
            legend_patches = [plt.scatter([], [], s=s, color="gray",alpha=0.8) 
                              for s in size_legend_scaled]
            # Place them in the bottom half
            ax.legend(legend_patches,size_labels,title=legend_size_title,loc="upper left",
                bbox_to_anchor=(1.02, 0.45),frameon=True,
                labelspacing=2,handletextpad=1.5 ,borderpad=1.5)

    return ax

