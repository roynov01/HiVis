# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:29:56 2026

@author: royno
"""

import matplotlib.pyplot as plt
from HiVis import HiVis



# Data was downloaded from: https://spateo-release.readthedocs.io/en/latest/tutorials/notebooks/1_cell_segmentation/stain_segmentation.html

path_output = r"output"
image_file = r"SS200000135IL-D1.ssDNA.tif"
transcripts_file = r"SS200000135TL_D1_all_bin1.txt.gz"

fluorescence = {"ssDNA": "white"}
name = "stereoseq"
bin_size_um = 5
microns_per_pixel = 0.5 

brain = HiVis.new_stereoseq(
    path_transcripts=transcripts_file,
    path_image=image_file,
    bin_size_um=bin_size_um,  
    name=name,
    path_output=path_output,
    fluorescence=fluorescence,
    microns_per_pixel=microns_per_pixel,
    flip_img=True
)

# ax = brain.plot.spatial("nUMI",exact=True,img_resolution="full",save=True,title="",show_zeros=1,scalebar={"text":False})
# brain.plot.spatial(exact=True,img_resolution="full",legend=False,save=True,scalebar={"text":False})


