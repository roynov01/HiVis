from HiVis import HiVis
import  matplotlib.pyplot as plt


path = r"region_1"
name = "merfish"
path_output = r"output"
fluorescence = {"DAPI":"green","PolyT":"red"}
ML = HiVis.new_merfish(path, name=name, path_output=path_output, fluorescence=fluorescence, 
                 bin_size_um=20, downscale_factor=4)



fig, axes = plt.subplots(1,2,figsize=(8,6))
ML.plot.spatial(ax=axes[0],img_resolution="full",legend=True, title="MERFISH (human liver)")
ML.plot.spatial("nUMI",image=False,exact=True,ax=axes[1],title=" ")
plt.tight_layout()
plt.savefig(r"spatial.pdf")
