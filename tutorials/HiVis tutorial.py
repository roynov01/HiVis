# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 15:55:36 2025

@author: royno
"""

import os
import sys
os.chdir(r"C:\Users\royno.WISMAIN\Documents\GitHub\HiVis\tutorials")

sys.path.append(os.path.abspath(".."))

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



#%%

segmentation_path = "qupath/stardist_results.csv"
segmentation = pd.read_csv(segmentation_path, sep="\t")
segmentation.rename(columns={"InCell":"in_cell", "InNuc":'in_nucleus',"Object ID":"Cell_ID"}, inplace=True)


si_subset.agg_stardist(segmentation, name="SC", obs2add=["Cell: Area µm^2","Eosin: Mean"],
                         obs2agg=["mito_sum","muscle_villi_classifier"])




