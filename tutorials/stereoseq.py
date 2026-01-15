from HiVis import HiVis

# Brain StereoSeq data was downloaded from: https://spateo-release.readthedocs.io/en/latest/tutorials/notebooks/1_cell_segmentation/stain_segmentation.html
# The data is from Chen et al. 2022 (PMID: 35512705), processed by the STOmics SAW software, available from Qiu et al. 2024 (PMID: 39532097).

path_output = r"output"
image_file = r"SS200000135IL-D1.ssDNA.tif"
transcripts_file = r"SS200000135TL_D1_all_bin1.txt.gz"

fluorescence = {"ssDNA": "white"}
name = "stereoseq"
bin_size_um = 5
microns_per_pixel = 0.5 

brain = HiVis.new_stereoseq(
    path_transcripts=transcripts_file, path_image=image_file, bin_size_um=bin_size_um, name=name,
    path_output=path_output, fluorescence=fluorescence, microns_per_pixel=microns_per_pixel, flip_img=True)
