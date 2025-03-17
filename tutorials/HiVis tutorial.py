# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 15:55:36 2025

@author: royno
"""

import os
import sys
os.chdir(r"C:\Users\royno.WISMAIN\Documents\GitHub\HiVis\tutorials")

# sys.path.append(os.path.abspath(".."))





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from HiVis import HiVis, HiVis_utils

import importlib

importlib.reload(HiVis_utils.HiVis_plot)
importlib.reload(HiVis.Aggregation_utils)

importlib.reload(HiVis_utils)
importlib.reload(HiVis)






#%%
path_image_fullres = r"outs\Visium_HD_Mouse_Small_Intestine_tissue_image.btf"
path_input_data = r"outs\binned_outputs\square_002um"
path_output = r"output"
properties = {"organism":"mouse",
              "organ":"Small intestine",
              "cancer": False,
              "source":"10X"}
si = HiVis.new(path_image_fullres, 
               path_input_data, 
               path_output,
               name="mouse_intestine",  
               properties=properties)


classifier_path = r"qupath\pixel_classifier.tif"
classifier_name = "muscle_villi_classifier"

mask_values = si.add_mask(classifier_path, classifier_name, plot=False)
values = {0:"immune", 1:"lumen", 2:"muscle", 3:"tissue"}
si.update_meta(classifier_name, values)


annotation_path = r"qupath\annotation.geojson"
annotation_name = "intestine_part"

mask_values = si.add_annotations(annotation_path, annotation_name)


high_expressed = si["nUMI_gene"] > 10000
si_subset = si[:, high_expressed]
si_subset.rename("highly_expressed_genes") # otherwise it will be called "subset"

segmentation_path = "qupath/stardist_results.csv"
segmentation = pd.read_csv(segmentation_path, sep="\t")
segmentation.rename(columns={"InCell":"in_cell", "InNuc":'in_nucleus',"Object ID":"Cell_ID"}, inplace=True)



#%%
# del si_subset["Cell_ID"];del si_subset["in_cell"];del si_subset["in_nucleus"];

si_subset.agg_stardist(input_df=segmentation, name="SC", obs2add=["Cell: Area µm^2","Eosin: Mean"],
                         obs2agg=["mito_sum","muscle_villi_classifier"])


print(si_subset.agg["SC"])
#%%
import scanpy as sc
adata_sc = si_subset.agg["SC"].adata.copy()

sc.pp.normalize_total(adata_sc, target_sum=1e4)
adata_sc.var["expression_mean"] = np.array(adata_sc.X.mean(axis=0)).flatten()/1e4
adata_sc.var["expression_max"] =  np.array(adata_sc.X.max(axis=0).toarray()).flatten()/1e4
sc.pp.log1p(adata_sc)
sc.pp.highly_variable_genes(adata_sc, n_top_genes=1000)
adata_sc_full = adata_sc.copy()
adata_sc = adata_sc[:,(adata_sc.var['highly_variable'] == True) ].copy()
sc.pp.scale(adata_sc, max_value=10)
sc.tl.pca(adata_sc, svd_solver="arpack")
sc.pl.pca_variance_ratio(adata_sc, log=True)
sc.pp.neighbors(adata_sc, n_neighbors=15, n_pcs=20) 
sc.tl.umap(adata_sc)
sc.tl.leiden(adata_sc,resolution=0.7,random_state=0,n_iterations=2,directed=False)
sc.pl.umap(adata_sc,color="leiden",size=5)
si_subset.agg["SC"].merge(adata_sc,obs="leiden",umap=True)
#%%
cmap = "tab10"
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
si_subset.agg["SC"].plot.spatial("leiden", xlim=[0,400], ylim=[3300,3700],size=15, ax=axes[0], axis_labels=False,cmap=cmap)
si_subset.agg["SC"].plot.umap("leiden",size=5, ax=axes[1],cmap=cmap)
si_subset.agg["SC"].plot.hist("leiden", ax=axes[2], ylab="Cells count",cmap=cmap,cropped=True)
plt.tight_layout()
#%%
si_subset.agg["SC"].sync("leiden")
np.unique(si_subset["leiden"])
#%%
clust1_dge = si_subset.agg.dge("leiden", group1="1") # we can also specify group2
clust1_dge2 = si_subset.agg.dge("leiden", group1="1", group2="5") 

sc_pb = si_subset.agg["SC"].pseudobulk("leiden")



