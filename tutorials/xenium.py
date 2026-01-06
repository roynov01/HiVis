# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 13:46:41 2025

@author: royno
"""


import numpy as np
import pandas as pd

from HiVis import HiVis
import  matplotlib.pyplot as plt






#%%
if __name__ == "__main__":
    path = r"Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs"
    
    fluorescence = {"DAPI":"white"}
    properties = None
    name = "xenium"
    bin_size_um = 10
    path_output = r"output"

    X = HiVis.new_xenium(path, bin_size_um, name, path_output, fluorescence, properties=None, downscale_factor=4)
    
    fig, axes = plt.subplots(2,1,figsize=(6,6))
    xlim, ylim= [5000,7000], [1500,2500]
    X.plot.spatial(ax=axes[0],img_resolution="full",legend=0, title="XENIUM (human ductal adenocarcinoma)",xlim=xlim,ylim=ylim)
    X.plot.spatial("nUMI",image=False,exact=True,ax=axes[1],title=" ",scalebar={"color":"white"},
                   xlim=xlim,ylim=ylim,cmap="hot",show_zeros=1)
    plt.tight_layout()
    plt.savefig(r"spatial.pdf")
