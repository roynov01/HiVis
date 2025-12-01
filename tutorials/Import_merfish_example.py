# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:25:58 2025

@author: royno
"""

import os
import json
import numpy as np
import pandas as pd
import tifffile
import anndata as ad
from scipy import sparse
from HiVis import HiVis

def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    microns_per_pixel = float(manifest["microns_per_pixel"])
    return manifest, microns_per_pixel


def get_microns_per_pixel(manifest_path):
    manifest, microns_per_pixel = load_manifest(manifest_path)
    return microns_per_pixel


def bin_transcripts_to_anndata(transcripts_csv, manifest_path, bin_size_um=2.0, z_plane=0):
    """Load"""
    manifest, microns_per_pixel = load_manifest(manifest_path)
    bbox_microns = manifest["bbox_microns"]
    x_min, y_min, x_max, y_max = bbox_microns

    df = pd.read_csv(transcripts_csv)

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("Expected columns 'global_x' and 'global_y' in transcripts file.")
    if "gene" not in df.columns:
        raise ValueError("Expected column 'gene' in transcripts file.")

    if "global_z" in df.columns:
        df = df[df["global_z"] == z_plane]

    df["x_um"] = df["global_x"]
    df["y_um"] = df["global_y"]

    df["x_rel_um"] = df["x_um"] - x_min
    df["y_rel_um"] = df["y_um"] - y_min

    df = df[(df["x_rel_um"] >= 0) & (df["x_rel_um"] <= (x_max - x_min)) & (df["y_rel_um"] >= 0) & (df["y_rel_um"] <= (y_max - y_min))]

    df["bin_x"] = (df["x_rel_um"] / bin_size_um).astype(int)
    df["bin_y"] = (df["y_rel_um"] / bin_size_um).astype(int)

    grouped = df.groupby(["bin_x", "bin_y", "gene"]).size().reset_index(name="count")

    matrix = grouped.pivot_table(index=["bin_x", "bin_y"], columns="gene", values="count", fill_value=0)
    matrix = matrix.sort_index(axis=0)
    matrix = matrix.sort_index(axis=1)

    bin_indices = np.array(matrix.index.tolist())
    bin_x = bin_indices[:, 0]
    bin_y = bin_indices[:, 1]

    x_center_rel_um = (bin_x + 0.5) * bin_size_um
    y_center_rel_um = (bin_y + 0.5) * bin_size_um
    um_x = x_center_rel_um + x_min
    um_y = y_center_rel_um + y_min

    pxl_x = x_center_rel_um / microns_per_pixel
    pxl_y = y_center_rel_um / microns_per_pixel

    pxl_col_in_fullres = pxl_x
    pxl_row_in_fullres = pxl_y

    obs_index = pd.Index(["bin_x{}_y{}".format(i, j) for i, j in zip(bin_x, bin_y)], name="bin_id")
    obs = pd.DataFrame({"um_x": um_x, "um_y": um_y, "pxl_row_in_fullres": pxl_row_in_fullres, "pxl_col_in_fullres": pxl_col_in_fullres}, index=obs_index)

    var = pd.DataFrame(index=matrix.columns)

    X = sparse.csr_matrix(matrix.values)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def load_mosaic_images(manifest_path, images_dir, stains=("DAPI", "PolyT"), z_plane=0):
    manifest, microns_per_pixel = load_manifest(manifest_path)
    available = manifest.get("mosaic_files", [])

    image_paths = []
    for stain in stains:
        found_path = None
        for entry in available:
            if entry.get("stain") == stain and int(entry.get("z", 0)) == z_plane:
                found_path = os.path.join(images_dir, entry["file_name"])
                break
        if found_path is None:
            raise ValueError("Could not find mosaic file for stain '{}' at z={}".format(stain, z_plane))
        image_paths.append(found_path)

    channels = []
    for path in image_paths:
        img = tifffile.imread(path)
        if img.ndim == 2:
            img2d = img
        elif img.ndim == 3:
            img2d = img[0]
        else:
            raise ValueError("Unexpected image shape {} for file {}".format(img.shape, path))
        channels.append(img2d)

    if len(channels) == 0:
        raise ValueError("No images loaded for stains {}".format(stains))

    stacked = np.stack(channels, axis=-1)
    return stacked, microns_per_pixel


if __name__ == "__main__":
    base_dir = r"region_0"
    bin_size_um = 20
    
    transcripts_csv = os.path.join(base_dir, "detected_transcripts.csv")
    manifest_path = os.path.join(base_dir, "images/manifest.json")
    images_dir = os.path.join(base_dir, "images")



    adata = bin_transcripts_to_anndata(transcripts_csv, manifest_path, bin_size_um=bin_size_um, z_plane=0)
    print(adata)

    img, microns_per_pixel = load_mosaic_images(manifest_path, images_dir, stains=("DAPI", "PolyT"), z_plane=0)
    print("Image shape:", img.shape)
    
    # Rescale the image if its too large
    from skimage.measure import block_reduce
    scale = 0.25
    down_factor = int(1/scale)
    down = block_reduce(img, block_size=(down_factor, down_factor, 1), func=np.mean)
    
    adata.obs["pxl_row_in_fullres"] = adata.obs["pxl_row_in_fullres"] / down_factor
    adata.obs["pxl_col_in_fullres"] = adata.obs["pxl_col_in_fullres"] / down_factor
    
    microns_per_pixel_down = microns_per_pixel * down_factor

    
    # flip the adata if image is flipped
    # adata.obs["pxl_row_in_fullres"] = down.shape[0] - adata.obs["pxl_row_in_fullres"]
    
    high_res_scale, low_res_scale = 0.25,0.01
    
    res = HiVis.HiVis_utils.create_rescaled_images(down, high_res_scale=high_res_scale, low_res_scale=low_res_scale)
    high_res_image, low_res_image, high_res_scale, low_res_scale = res
    
    high_res_image = HiVis.HiVis_utils.fluorescence_to_RGB(high_res_image, colors=["green","red"])
    low_res_image = HiVis.HiVis_utils.fluorescence_to_RGB(low_res_image, colors=["green","red"])

    

    scalefactor_json = {"microns_per_pixel":microns_per_pixel_down,
                        "bin_size_um":bin_size_um,
                        "tissue_hires_scalef": high_res_scale,
                        "tissue_lowres_scalef": low_res_scale}
    
    
    HiVis.HiVis_utils._edit_adata(adata, scalefactor_json, "MT-")
    
    
    ML = HiVis.HiVis(adata, down, high_res_image, low_res_image, scalefactor_json, 
                 name="merfish", path_output=r"X:\roy\viziumHD\analysis\Python\version_11\merfish\output",
                 properties={"organism":"human"}, agg=None, fluorescence={"DAPI":"green","polyT":"red"})
