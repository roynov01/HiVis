# -*- coding: utf-8 -*-
"""
HD Integrated Visium Interactive Suite (HiVis)
"""
# General libraries
import os
import dill
import gc
import warnings
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Data libraries
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt, affinity
from scipy import sparse
import anndata as ad

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import tifffile

from . import HiVis_utils
from .Aggregation import Aggregation
from . import HiVis_plot
from . import HiVis_analysis
from . import other_utils
from . import Aggregation_utils

Image.MAX_IMAGE_PIXELS = 1063425001 # Enable large images loading


def load(filename, directory=''):
    '''
    loads an instance from a pickle format, that have been saved via HiVis.save()
    
    Parameters:
        * filename (str)- full path of pkl file, or just the filename if directory is specified
        
    **Returns** HiVis instance
    '''
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"
    if directory:
        filename = f"{directory}/{filename}"
    HiVis_utils.validate_exists(filename)
    with open(filename, "rb") as f:
        instance = dill.load(f)
    return instance

def new_visiumHD(path_image_fullres:str, path_input_data:str, path_output:str,
             name:str, crop_images=True, properties: dict = None, on_tissue_only=True,min_reads_in_spot=1,
             min_reads_gene=10, fluorescence=False, plot_qc=True):
    '''
    - Loads images, data and metadata.
    - Initializes the connection from the data and metadata to the images coordinates
    - Adds basic QC to the metadata (nUMI, mitochondrial %)
    
    Parameters:
        * path_input_fullres_image (str) - path of the full resolution microscopy image
        * path_input_data (str) - folder with outs of the Visium. Typically square_002um \
                            (with h5 files and with folders filtered_feature_bc_matrix, spatial)
        * path_output (str) - path where to save plots and files
        * name (str) - name of the instance
        * crop_images (bool) - crop the regions outside of the spots cover area
        * properties (dict) - can be any metadata, such as organism, organ, sample_id. \
            Organism is recomended (i.e human or mouse) for case-sensitivity correction and QC of mitochondrial reads.
        * on_tissue_only (bool) - remove spots that are not classified as "on tissue"
        * min_reads_in_spot (int) - filter out spots with less than X UMIs
        * min_reads_gene (int) - filter out gene that is present in less than X spots
        * fluorescence - either False for H&E, or a dict of channel names and colors. color can be None. Example {"DAPI":"blue"}
        * plot_qc (bool) - plot QC when object is being created
        
    **Returns** HiVis instance
    '''
    # Validate paths of metadata and images
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    path_image_highres = path_input_data + "/spatial/tissue_hires_image.png"
    path_image_lowres = path_input_data + "/spatial/tissue_lowres_image.png"
    json_path = path_input_data + "/spatial/scalefactors_json.json"
    metadata_path = path_input_data + "/spatial/tissue_positions.parquet"
    HiVis_utils.validate_exists([path_image_fullres,path_image_highres,path_image_lowres,json_path,metadata_path])
    
    # Load images
    image_fullres, image_highres, image_lowres = HiVis_utils.load_images(path_image_fullres, path_image_highres, path_image_lowres)
    
    # Load scalefactor_json
    with open(json_path) as file:
        scalefactor_json = json.load(file)
    
   # Load data + metadata
    adata = HiVis_utils._import_data(metadata_path, path_input_data, path_image_fullres, on_tissue_only)
    
    if crop_images:
        # Crop images
        adata, image_fullres, image_highres, image_lowres = HiVis_utils._crop_images_permenent(
            adata, image_fullres, image_highres, image_lowres, scalefactor_json)
        
        # Save cropped images
        path_image_fullres_cropped = path_image_fullres.replace("." + path_image_fullres.split(".")[-1], "_cropped.tif")
        path_image_highres_cropped = path_image_highres.replace("." + path_image_highres.split(".")[-1], "_cropped.tif")
        path_image_lowres_cropped = path_image_lowres.replace("." + path_image_lowres.split(".")[-1], "_cropped.tif")
        HiVis_utils._export_images(path_image_fullres_cropped, path_image_highres_cropped, 
                                      path_image_lowres_cropped,image_fullres,
                                      image_highres, image_lowres,um_per_pxl=scalefactor_json["microns_per_pixel"])
        cols = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres','pxl_col_in_fullres']        
        csv_path = Path(path_image_fullres_cropped).parent / f"{name}_tissue_positions_cropped.csv"
        adata.obs[cols].to_csv(csv_path, index=True, index_label="barcode")
    
    if fluorescence:
        HiVis_utils._measure_fluorescence(adata, image_fullres, list(fluorescence.keys()), scalefactor_json["spot_diameter_fullres"])

    # Add QC (nUMI, mito %) and unit transformation
    mito_name_prefix = "MT-" if properties.get("organism") == "human" else "mt-"
    HiVis_utils._edit_adata(adata, scalefactor_json, mito_name_prefix)

    # Filter low quality spots and lowly expressed genes
    adata = adata[adata.obs["nUMI"] >= min_reads_in_spot, adata.var["nUMI_gene"] >= min_reads_gene].copy()

    return HiVis(adata, image_fullres, image_highres, image_lowres, scalefactor_json, 
                    name, path_output, properties, agg=None, fluorescence=fluorescence, plot_qc=plot_qc)

def new_merfish(path, bin_size_um, name, path_output, fluorescence, properties=None, downscale_factor=4):
    '''
    Load MERFISH data into a HiVis object.

    Parameters:
        * path (str or pathlib.Path) - Path to the Xenium output directory. Must contain experiment.xenium and a morphology_focus subdirectory with OME-TIFF morphology images.
        * bin_size_um (float) - Spatial bin size in micrometers used to aggregate transcript counts into a grid.
        * name (str) - Name to assign to the returned HiVis object.
        * path_output (str or pathlib.Path) - Output directory passed to the HiVis object for saving results/artifacts.
        * fluorescence (dict) - Dict of channel names and colors. color can be None. Example {"DAPI":"blue"}
        * properties (dict, optional) - Metadata passed to the HiVis object.
        * downscale_factor (int) - Downscaling factor applied when generating the downscaled image (default: 4).
    
    Returns:
        HiVis instance.
    '''
    def load_manifest(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        microns_per_pixel = float(manifest["microns_per_pixel"])
        return manifest, microns_per_pixel

    def get_microns_per_pixel(manifest_path):
        manifest, microns_per_pixel = load_manifest(manifest_path)
        return microns_per_pixel

    def merfish_transcripts_to_anndata(transcripts_csv, manifest_path, bin_size_um=2.0, z_plane=0):
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
        adata = adata[:,~adata.var_names.str.startswith("Blank")].copy()
        return adata


    def load_merfish_images(manifest_path, images_dir, stains, z_plane=0):
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
    
    transcripts_csv = os.path.join(path, "detected_transcripts.csv")
    manifest_path = os.path.join(path, "images/manifest.json")
    images_dir = os.path.join(path, "images")

    high_res_scale = 0.25
    low_res_scale = 0.01

    adata = merfish_transcripts_to_anndata(transcripts_csv, manifest_path, bin_size_um=bin_size_um, z_plane=0)

    img, microns_per_pixel = load_merfish_images(manifest_path, images_dir, 
                                                list(fluorescence.keys()), z_plane=0)
    
    # Rescale the image if its too large
    downscaled_img, high_res_image, low_res_image, microns_per_pixel_down = HiVis_utils.rescale_img_and_adata(adata,
                                                                        microns_per_pixel, img, down_factor=downscale_factor, 
                                                                        fluorescence=fluorescence,
                                                                        high_res_scale=high_res_scale, low_res_scale=low_res_scale)
        
    scalefactor_json = {"microns_per_pixel":microns_per_pixel_down,
                        "bin_size_um":bin_size_um,
                        "spot_diameter_fullres": bin_size_um/microns_per_pixel_down,
                        "tissue_hires_scalef": high_res_scale,
                        "tissue_lowres_scalef": low_res_scale}
    if properties is None:
        properties = {}
    mito_name_prefix = "MT-" if properties.get("organism") == "human" else "mt-"
    HiVis_utils._edit_adata(adata, scalefactor_json, mito_name_prefix)
    
    return HiVis(adata, downscaled_img, high_res_image, low_res_image, scalefactor_json, 
                 name=name, path_output=path_output,properties=properties, agg=None, fluorescence=fluorescence)


def new_xenium(path, bin_size_um, name, path_output, fluorescence, properties=None, downscale_factor=1):
    '''
    Load 10x Genomics Xenium data into a HiVis object.
    
    This function:
    - Reads Xenium transcript points via spatialdata_io.xenium and aggregates them into a binned count matrix (AnnData) using bin_size_um.
    - Loads Xenium morphology OME-TIFF images from the morphology_focus directory and stacks them into a single (Y, X, C) array.

    Parameters:
        * path (str or pathlib.Path) - Path to the Xenium output directory. Must contain experiment.xenium and a morphology_focus subdirectory with OME-TIFF morphology images.
        * bin_size_um (float) - Spatial bin size in micrometers used to aggregate transcript counts into a grid.
        * name (str) - Name to assign to the returned HiVis object.
        * path_output (str or pathlib.Path) - Output directory passed to the HiVis object for saving results/artifacts.
        * fluorescence - either False for H&E, or a dict of channel names and colors. color can be None. Example {"DAPI":"blue"}
        * properties (dict, optional) - Metadata passed to the HiVis object.
        * downscale_factor (int) - Downscaling factor applied when generating the downscaled image.
    
    Returns:
        HiVis instance.
    '''
    from spatialdata_io import xenium
    def load_xenium_image(xenium_outs_path):
        xenium_outs_path = Path(xenium_outs_path)
        morph_path = xenium_outs_path / "morphology_focus"
        if not morph_path.exists():
            raise FileNotFoundError(f"Cannot find morphology_focus in {xenium_outs_path}")

        # 1. microns-per-pixel from experiment.xenium
        meta_path = xenium_outs_path / "experiment.xenium"
        if not meta_path.exists():
            raise FileNotFoundError(f"Cannot find experiment.xenium in {xenium_outs_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        microns_per_pixel = meta["pixel_size"]

        # 2. list morphology OME-TIFFs
        tiff_paths = sorted(morph_path.glob("*.ome.tif*"))
        if not tiff_paths:
            raise FileNotFoundError(f"No OME-TIFFs found in {morph_path}")

        selected_imgs = []
        selected_channel_names = []

        # 3. iterate over files, parse channel name, and load
        for path in tiff_paths:
            name = path.name

            # Strip ".ome.tif" or ".ome.tiff"
            base = name.split(".ome")[0]  # e.g. "ch0000_dapi" or "ch0001_atp1a1_cd45_e-cadherin"
            parts = base.split("_", 1)
            if len(parts) < 2:
                print(f"Warning: could not parse channel from filename '{name}', skipping.")
                continue

            channel_name = parts[1]  # everything after first "_"

            # Load the image
            arr = tifffile.imread(path)

            # Convert to 2D Y x X
            if arr.ndim == 2:
                img_yx = arr
            elif arr.ndim == 3:
                # Assumption: first axis is an extra dimension (e.g. scale); use first plane
                img_yx = arr[0]
            else:
                raise ValueError(
                    f"Unexpected number of dimensions ({arr.ndim}) for file {name}; "
                    "expected 2D or 3D."
                )

            # Check shape consistency across channels
            if selected_imgs:
                if img_yx.shape != selected_imgs[0].shape:
                    raise ValueError(
                        f"Image shape mismatch for file {name}: {img_yx.shape} "
                        f"vs {selected_imgs[0].shape} from previous file(s)."
                    )

            selected_imgs.append(img_yx)
            selected_channel_names.append(channel_name)

        if not selected_imgs:
            raise ValueError(
                "No images were loaded. All files failed channel parsing or there were none."
            )

        # 4. stack into Y x X x C
        img_yxc = np.stack(selected_imgs, axis=-1)

        print(f"\nFinal image shape (Y, X, C): {img_yxc.shape}")
        print("Channels included (in C order):")
        for i, ch in enumerate(selected_channel_names):
            print(f"  C[{i}] -> {ch}")

        return img_yxc, microns_per_pixel

    def xenium_sdata_to_anndata(sdata, xenium_outs_path, bin_size_um, z_plane=None, qv_threshold=20.0, only_assigned_to_cells=False):
        xenium_outs_path = Path(xenium_outs_path)

        # 1. Extract transcripts table
        pts = sdata.points["transcripts"]
        df = pts.compute() if hasattr(pts, "compute") else pts
        ignore_rows = df["feature_name"].str.startswith("NegControl") | df["feature_name"].str.startswith("Unassigned")
        df = df.loc[~ignore_rows]

        # 2. Rename required columns directly
        df = df.rename(columns={"x": "x_um", "y": "y_um", "z": "z_um", "feature_name": "gene"})

        # 3. Filters
        if qv_threshold is not None and "qv" in df.columns:
            df = df[df["qv"] >= qv_threshold]
        if z_plane is not None:
            df = df[np.isclose(df["z_um"], z_plane)]
        if only_assigned_to_cells and "cell_id" in df.columns:
            df = df[df["cell_id"].notna() & (df["cell_id"] != -1)]

        # 4. Bounding box only for relative coordinates
        x_min = df["x_um"].min()
        y_min = df["y_um"].min()

        df["x_rel_um"] = df["x_um"] - x_min
        df["y_rel_um"] = df["y_um"] - y_min

        # 5. Assign bins
        df["bin_x"] = (df["x_rel_um"] / bin_size_um).astype(int)
        df["bin_y"] = (df["y_rel_um"] / bin_size_um).astype(int)

        # 6. Aggregate counts
        grouped = df.groupby(["bin_x", "bin_y", "gene"]).size().reset_index(name="count")
        matrix = grouped.pivot_table(index=["bin_x", "bin_y"], columns="gene", values="count", fill_value=0)
        matrix = matrix.sort_index(axis=0).sort_index(axis=1)

        # 7. Compute bin centers
        bin_indices = np.array(matrix.index.tolist())
        bin_x = bin_indices[:, 0]
        bin_y = bin_indices[:, 1]

        x_center_rel_um = (bin_x + 0.5) * bin_size_um
        y_center_rel_um = (bin_y + 0.5) * bin_size_um

        um_x = x_center_rel_um + x_min
        um_y = y_center_rel_um + y_min

        # 8. Read microns-per-pixel
        with open(xenium_outs_path / "experiment.xenium", "r") as f:
            exp_meta = json.load(f)
        microns_per_pixel = exp_meta["pixel_size"]

        pxl_x = x_center_rel_um / microns_per_pixel
        pxl_y = y_center_rel_um / microns_per_pixel

        # 9. Build AnnData
        obs_index = pd.Index([f"bin_x{i}_y{j}" for i, j in zip(bin_x, bin_y)], name="bin_id")
        obs = pd.DataFrame({"um_x": um_x, "um_y": um_y, "pxl_row_in_fullres": pxl_y, "pxl_col_in_fullres": pxl_x}, index=obs_index)
        var = pd.DataFrame(index=matrix.columns)
        X = sparse.csr_matrix(matrix.values)

        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.uns["bin_size_um"] = bin_size_um
        adata.uns["microns_per_pixel"] = microns_per_pixel

        return adata
    # Load data
    sdata = xenium(path,cells_boundaries=False,cells_as_circles =False,
                                  nucleus_boundaries=False,nucleus_labels=False,cells_labels =False,aligned_images =False,
                                  cells_table =False)
    adata = xenium_sdata_to_anndata(sdata, path, bin_size_um=bin_size_um)
    mask = ~adata.var_names.str.startswith(("NegControl", "Unassigned")).astype(bool)
    adata = adata[:, mask].copy()

    # Load images    
    img, microns_per_pixel = load_xenium_image(path)

    high_res_scale = 0.25
    low_res_scale = 0.01

    downscaled_img, high_res_image, low_res_image, microns_per_pixel_down = HiVis_utils.rescale_img_and_adata(adata,
                                                                    microns_per_pixel, img,down_factor=downscale_factor,
                                                                    fluorescence=fluorescence,
                                                                    high_res_scale=high_res_scale, low_res_scale=low_res_scale)
    
    scalefactor_json = {"microns_per_pixel":microns_per_pixel_down,
                        "bin_size_um":bin_size_um,
                        "spot_diameter_fullres": bin_size_um/microns_per_pixel_down,
                        "tissue_hires_scalef": high_res_scale,
                        "tissue_lowres_scalef": low_res_scale}
    if properties is None:
        properties = {}
    mito_name_prefix = "MT-" if properties.get("organism") == "human" else "mt-"
    HiVis_utils._edit_adata(adata, scalefactor_json, mito_name_prefix)


    return HiVis(adata, downscaled_img, high_res_image, low_res_image, scalefactor_json, 
             name=name, path_output=path_output,properties=properties, agg=None, fluorescence=fluorescence)

def new_stereoseq(path_transcripts, path_image, bin_size_um, name, path_output, fluorescence, 
                  microns_per_pixel=0.5, properties=None, downscale_factor=1, flip_img=True, exp_col="MIDCounts"):
    """
    Loads Stereoseq data into HiVis.
    
    Parameters:
        path_transcripts (str): Path to the gene count table (csv/tsv).
        path_image (str): Path to the fullres TIFF image.
        bin_size_um (float): Desired bin size in microns.
        name (str): Sample name.
        path_output (str): Output directory.
        fluorescence (dict): Dictionary of channel names.
        microns_per_pixel (float): Resolution of the input image (default 0.5 for standard Stereo-seq).
        properties (dict): Optional metadata (e.g., organism).
        downscale_factor (int): Factor to downscale the image for visualization.
        flip_img (bool) - flip by 270 degrees and mirror the image.
        exp_col (str): Name of column in the data which stores the count
        
    Returns: HiVis object
    """

    def stereoseq_transcripts_to_anndata(transcripts_path, bin_size_um, microns_per_pixel):
        # 1. Load Data
        # Try reading as tab-separated first (common for stereoseq), fallback to comma
        try:
            df = pd.read_csv(transcripts_path, sep='\t')
            if "x" not in df.columns: # If loading failed or it's actually comma sep
                df = pd.read_csv(transcripts_path, sep=',')
        except:
            df = pd.read_csv(transcripts_path)

        # 2. Validate Columns
        required_cols = ["x", "y", "geneID", exp_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Expected column '{col}' in transcripts file. Found: {df.columns}")

        # 3. Calculate Bounding Box dynamically from data
        x_min = df["x"].min()
        y_min = df["y"].min()

        # 4. Convert to Microns
        df["x_um"] = df["x"] * microns_per_pixel
        df["y_um"] = df["y"] * microns_per_pixel
        
        # Calculate relative coordinates from the top-left of the bounding box
        min_x_um = x_min * microns_per_pixel
        min_y_um = y_min * microns_per_pixel
        
        df["x_rel_um"] = df["x_um"] - min_x_um
        df["y_rel_um"] = df["y_um"] - min_y_um

        # 5. Binning
        # Assign each transcript to a grid bin
        df["bin_x"] = (df["x_rel_um"] / bin_size_um).astype(int)
        df["bin_y"] = (df["y_rel_um"] / bin_size_um).astype(int)

        # 6. Aggregation
        grouped = df.groupby(["bin_x", "bin_y", "geneID"])[exp_col].sum().reset_index(name="count")

        # 7. Create Matrix
        matrix = grouped.pivot_table(index=["bin_x", "bin_y"], columns="geneID", values="count", fill_value=0)
        matrix = matrix.sort_index(axis=0)
        matrix = matrix.sort_index(axis=1)

        # 8. Create Spatial Coordinates for AnnData
        bin_indices = np.array(matrix.index.tolist())
        bin_x = bin_indices[:, 0]
        bin_y = bin_indices[:, 1]

        # Center of the bin in relative microns
        x_center_rel_um = (bin_x + 0.5) * bin_size_um
        y_center_rel_um = (bin_y + 0.5) * bin_size_um
        
        # Absolute microns (if needed later)
        um_x = x_center_rel_um + min_x_um
        um_y = y_center_rel_um + min_y_um

        # Map back to pixels in the original fullres image
        pxl_col_in_fullres = x_center_rel_um / microns_per_pixel + x_min
        pxl_row_in_fullres = y_center_rel_um / microns_per_pixel + y_min


        obs_index = pd.Index([f"bin_x{i}_y{j}" for i, j in zip(bin_x, bin_y)], name="bin_id")
        obs = pd.DataFrame({
            "um_x": um_x, 
            "um_y": um_y, 
            "pxl_row_in_fullres": pxl_row_in_fullres, 
            "pxl_col_in_fullres": pxl_col_in_fullres
        }, index=obs_index)

        var = pd.DataFrame(index=matrix.columns)
        X = sparse.csr_matrix(matrix.values)
        
        return ad.AnnData(X=X, obs=obs, var=var)

    def load_stereoseq_image(image_path, flip=False):
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")
            
        img = tifffile.imread(image_path)
        
        # Ensure we have the right dimensions (Rows, Cols, Channels)
        # If image is 2D (Gray), make it (R, C, 1) or leave as is depending on downstream needs
        # The reference code expects (Rows, Cols, Channels) for stacking, 
        # but typically rescaling functions handle 2D arrays fine or expect 3D.
        if img.ndim == 2:
            img = img[:, :, np.newaxis] 
        elif img.ndim == 3:
            # Check if channels are first or last (TiffFile usually does (C, H, W) or (H, W, C))
            # Heuristic: if dim 0 is small (e.g. 3 or 4) and dim 1/2 are large, likely C,H,W
            if img.shape[0] < 10 and img.shape[1] > 100:
                img = np.moveaxis(img, 0, -1)
        if flip:
            # Rotates -90 degrees and flips
            img = np.flipud(img)
            img = np.rot90(img, 3)
        return img
    
    high_res_scale = 0.25
    low_res_scale = 0.01

    # 1. Create AnnData from Table
    adata = stereoseq_transcripts_to_anndata(path_transcripts, bin_size_um=bin_size_um, microns_per_pixel=microns_per_pixel)

    # 2. Load Image
    img = load_stereoseq_image(path_image,flip=flip_img)

    # 3. Rescale image and create lower resolution images
    downscaled_img, high_res_image, low_res_image, microns_per_pixel_down = HiVis_utils.rescale_img_and_adata(
        adata,microns_per_pixel, img, down_factor=downscale_factor,fluorescence=fluorescence,
        high_res_scale=high_res_scale, low_res_scale=low_res_scale)
        
    scalefactor_json = {
        "microns_per_pixel": microns_per_pixel_down,
        "bin_size_um": bin_size_um,
        "spot_diameter_fullres": bin_size_um / microns_per_pixel_down,
        "tissue_hires_scalef": high_res_scale,
        "tissue_lowres_scalef": low_res_scale
    }

    if properties is None:
        properties = {}
    
    # Handle mitochondrial prefix
    mito_name_prefix = "MT-" if properties.get("organism") == "human" else "mt-"
    HiVis_utils._edit_adata(adata, scalefactor_json, mito_name_prefix)
    
    return HiVis(adata, downscaled_img, high_res_image, low_res_image, scalefactor_json, 
        name=name, path_output=path_output,properties=properties,agg=None, fluorescence=fluorescence)


class HiVis:
    '''
    Main class. Stores the data and images of the VisiumHD, enables plotting via HiVis.plot, \
    analysis via HiVis.analysis, and can store Aggregation instances in HiVis.agg.
    
    To make a new class, call the new_visiumHD() function.
    '''
    def __init__(self, adata, image_fullres, image_highres, image_lowres, scalefactor_json, 
                 name, path_output, properties=None, agg=None, fluorescence=False, plot_qc=True):
        self.agg = agg
        self.tree = None
        self.name, self.path_output = name, path_output 
        self.properties = properties if properties else {}
        self.organism = self.properties.get("organism")
        if isinstance(image_fullres, str): # paths of images, not the images themselves
            image_fullres, image_highres, image_lowres = HiVis_utils.load_images(image_fullres, image_highres, image_lowres)
        
        self.image_fullres, self.image_highres, self.image_lowres = image_fullres, image_highres, image_lowres
        
        self.fluorescence = fluorescence
        
        if isinstance(scalefactor_json, str):
            with open(scalefactor_json) as file:
                scalefactor_json = json.load(file)        
        self.json = scalefactor_json
        
        HiVis_utils.add_spatial_keys(self, adata, name) # add obsm["spatial"] and uns["spatial"]
            
        self.adata = adata
        if self.json is not None:
            adata.obs["pxl_col_in_lowres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
            adata.obs["pxl_row_in_lowres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_lowres_scalef"]
            adata.obs["pxl_col_in_highres"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
            adata.obs["pxl_row_in_highres"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["tissue_hires_scalef"]
            adata.obs["um_x"] = adata.obs["pxl_col_in_fullres"] * scalefactor_json["microns_per_pixel"]
            adata.obs["um_y"] = adata.obs["pxl_row_in_fullres"] * scalefactor_json["microns_per_pixel"]
        
        self.plot = HiVis_plot.PlotVisium(self)
        self.analysis = HiVis_analysis.AnalysisVisium(self)
        
        if image_fullres is not None:
            if fluorescence:
                self.image_fullres_orig = self.image_fullres.copy()
                self.recolor(fluorescence)
            self.plot._init_img()
        else: # disable methods that rely on image
            print("Without image_fullres, [plot.spatial,analysis.smooth, analysis.compute_distances] will be disabled")
            def _disabled_method(*args, **kwargs):
                raise RuntimeError("This method is disabled in combined HiVis")
            self.plot.spatial = _disabled_method
            self.analysis.smooth = _disabled_method
            self.analysis.compute_distances = _disabled_method
            disable = ["add_agg", "add_mask", "add_annotations","agg_cells", "agg_from_annotations","export_images", "remove_pixels", "recolor"]
            for method_name in disable:
                setattr(self, method_name, _disabled_method)
        if plot_qc and hasattr(self.plot, "spatial"):
            self.analysis.qc(save=True)
            plt.show()
            
    def add_agg(self, adata_agg, name):
        '''
        Creates and adds Aggregation to the HiVis instance. Can be accessed by self.agg[name].
        For example single-cells, tissue structures.
        
        Parameters:
            * adata_agg (ad.AnnData) - anndata containing aggregations
            * name (str) - name of the aggregation object
        '''
        
        if not isinstance(adata_agg, ad.AnnData):
            raise TypeError("adata_agg must be anndata")
        if self.agg:
            if name in self.agg:
                print(f"{name} allready in {self.name}. Renamed previous Agg to 'temp'.")
                self.agg["temp"] = self.agg[name]
                del self.agg[name]
        else:
            self.agg = {}
        agg_name = f"{self.name}_{name}"
        agg = Aggregation(self, adata_agg, name=agg_name)
        self.agg[name] = agg
    
    
    def add_annotations(self, path:str, name:str, measurements=True):
        '''
        Adds annotations made in Qupath (geojson)
        
        Parameters:
            * path (str) - path to geojson file
            * name (str) - name of the annotation (that will be called in the obs)
            * measurements (bool) - include measurements columns 
        '''
        HiVis_utils.validate_exists(path)
        annotations = gpd.read_file(path)
        if "classification" in annotations.columns:
            annotations["classification"] = annotations["classification"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            # annotations["classification"] = annotations["classification"].apply(json.loads)
            # annotations[name] = [x["name"] for x in annotations["classification"] if isinstance(x, dict) else x]
            annotations[name] = [
                x["name"] if isinstance(x, dict) else np.nan
                for x in annotations["classification"]
            ]
        else:
            annotations[name] = annotations.index
        annotations[f"{name}_id"] = annotations["id"]
        del annotations["id"]
        del annotations["objectType"]
        if "isLocked" in annotations.columns:
            del annotations["isLocked"]
        
        if "measurements" in annotations.columns and measurements:
            measurements_df = pd.json_normalize(annotations["measurements"])
            annotations = gpd.GeoDataFrame(pd.concat([annotations.drop(columns=["measurements"]), measurements_df], axis=1))
            perimeter = annotations.geometry.length
            area = annotations.geometry.area
            annotations["circularity"] = (4 * np.pi * area) / (perimeter ** 2)
            annotations.loc[perimeter == 0, "circularity"] = np.nan
            cols = list(measurements_df.columns) + ["circularity",name,f"{name}_id"]
        else:
            if measurements:
                print("No measurements found")
            cols = [name,f"{name}_id"]
        for col in cols:
            if col in self.adata.obs.columns:
                del self.adata.obs[col]
        obs = gpd.GeoDataFrame(self.adata.obs, 
              geometry=gpd.points_from_xy(self.adata.obs["pxl_col_in_fullres"],
                                          self.adata.obs["pxl_row_in_fullres"]))        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            merged_obs = gpd.sjoin(obs,annotations,how="left",predicate="within")
        merged_obs = merged_obs[~merged_obs.index.duplicated(keep="first")]
        
        self.adata.obs = self.adata.obs.join(pd.DataFrame(merged_obs[cols]),how="left")
        self.plot._init_img()
        
    
    def add_mask(self, mask_path:str, name:str, plot=True, cmap="Paired", downscale=10):
        '''
        assigns each spot a value based on mask (image).
        
        Parameters:
            * mask_path (str) - path to mask image
            * name (str) - name of the mask (that will be called in the metadata)
            * plot (bool) - plot the mask
            * cmap (str) - colormap for plotting
            * downscale (int) - sownscale the plot. only relevent if plot=True
            
        **Returns** the mask (np.array)
        '''
        HiVis_utils.validate_exists(mask_path)
        
        def _import_mask(mask_path):
            '''imports the mask'''
            print("[Importing mask]")
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            return mask_array
        
        def _plot_mask(mask_array, cmap, downscale):
            '''plots the mask'''
            if downscale > 1:
                mask_array = mask_array[::downscale, ::downscale]
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_array, cmap=cmap)
            num_colors = len(np.unique(mask_array[~np.isnan(mask_array)]))
            cmap = plt.get_cmap(cmap, num_colors)  # plt.cm.get_cmap was removed in matplotlib>=3.9; plt.get_cmap works on both old and new
            legend_elements = [Patch(facecolor=cmap(i), label=f'{i}') for i in range(num_colors)]
            plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.5))
            plt.axis('off') 
            plt.show()

        def _assign_spots(mask_array, name):
            '''assigns each spot a value from the mask'''
            def _get_mask_value(mask_array, x, y):
                if (0 <= x < mask_array.shape[1]) and (0 <= y < mask_array.shape[0]):
                    return mask_array[y, x]
                else:
                    return None
            def _get_spot_identity(row):
                x = round(row['pxl_col_in_fullres'])
                y = round(row['pxl_row_in_fullres'])
                return _get_mask_value(mask_array, x, y)
            
            tqdm.pandas(desc=f"Assigning spots identity [{name}]")
            
            self.adata.obs[name] = np.nan
            self.adata.obs[name] = self.adata.obs.progress_apply(
                _get_spot_identity, axis=1)

        mask_array = _import_mask(mask_path)
        if plot:
            _plot_mask(mask_array, cmap=cmap,downscale=downscale)
        _assign_spots(mask_array, name)
        self.plot._init_img()
        print(f"\nTo rename the values in the metadata, call the [update_meta] method with [{name}] and dictionary with current_name:new_name")
        return mask_array
    

    def agg_from_annotations(self, annotation_id_col, name="SC", obs2agg=None, geojson_path=None):
        '''
        Adds Aggregation object to self.agg[name], based on annotation column.
        
        Parameters:
            * annotation_id_col (str) - column name that the aggregation will be based on
            * name (str) - name to store the Aggregation in. Can be accessed via HiVis.agg[name]
            * obs2agg - what obs to aggregate from the HiVis. \
                        Can be a list of column names. numeric columns will be summed, categorical will be the mode. \
                        Can be a dictionary specifying the aggregation function. \
                        examples: {"value_along_axis":np.median} or {"value_along_axis":[np.median,np.mean]}
            * geojson_path (str) - path to geojson file that was used to create the annotations
        '''        
        aggregation_func = Aggregation_utils._aggregate_data_annotations
        
        annotation_col = annotation_id_col.replace("_id","")
        if annotation_col != annotation_id_col and annotation_col in self.adata.obs.columns: # Add annotation class to each object
            if obs2agg is not None:
                if annotation_col not in obs2agg:
                    if isinstance(obs2agg, list):
                        obs2agg += [annotation_col]
                    else:
                        obs2agg[annotation_col] = None
            else:
                obs2agg = [annotation_col]
        
        adata_agg, _ = Aggregation_utils.new_adata(self.adata, annotation_id_col, aggregation_func,obs2agg=obs2agg)
        
        # adata_agg = Aggregation_utils.add_spatial_keys(self, adata_agg, f"{self.name}_{name}")
        self.add_agg(adata_agg, name=name)
        self.agg[name].adata.obs[f"{annotation_col}_col"] = self.agg[name].adata.obs.index
        if geojson_path:
            self.agg[name].import_geometry(geojson_path, object_type="annotation")            
        
    
    def agg_cells(self, input_df, name="SC", obs2add=None, obs2agg=None, geojson_path=None):
        '''
        Adds Aggregation object to self.agg[name], based on CSV output of Qupath pipeline.
        
        Parameters:
            * input_df (pd.DataFrame) - output of Qupath pipeline 
            * name (str) - name to store the Aggregation in. Can be accessed via HiVis.agg[name]
            * obs2agg - what obs to aggregate from the HiVis. \
                        Can be a list of column names. numeric columns will be summed, categorical will be the mode. \
                        Can be a dictionary specifying the aggregation function. \
                        examples: {"value_along_axis":np.median} or {"value_along_axis":[np.median,np.mean]}
            * obs2add (list) - which columns from input_df should be copied to the Aggregation.adata.obs
            * nuc_only (bool) - aggregate only spots in nuclei
            * geojson_path (str) - path to geojson file that was exported from Qupath
        '''
        id_col = f"Cell_ID_{name}"
        input_df.rename(columns={"Object ID":"Cell_ID"}, inplace=True)
        spots_only, cells_only = Aggregation_utils.split_spots_cells(input_df)
        
        spots_only = spots_only.rename(columns={"Cell_ID": id_col})
        cells_only.index.name = id_col
        
        self.adata.obs = self.adata.obs.drop(columns=[id_col], errors='ignore')
        overlap = self.adata.obs.columns.intersection(spots_only.columns)
        overlap = overlap.difference([id_col])
        if len(overlap) > 0:
            self.adata.obs = self.adata.obs.drop(columns=list(overlap))
            print(f"Dropping overlapping columns from obs: {list(overlap)}")
        self.adata.obs = self.adata.obs.join(spots_only,how="left")
        
        aggregation_func = Aggregation_utils._aggregate_data_cells
        
        adata_agg, _ = Aggregation_utils.new_adata(self.adata, id_col, aggregation_func,obs2agg=obs2agg)
        
        # adata_agg.obs[id_col] = adata_agg.obs.index
        
        if obs2add:
            obs2add = [col for col in cells_only.columns if col in obs2add]
            Aggregation_utils.merge_cells(cells_only, adata_agg, obs2add, id_col=id_col)
        
        # adata_agg = Aggregation_utils.add_spatial_keys(self, adata_agg, f"{self.name}_{name}")
        
        self.add_agg(adata_agg, name=name)
        if geojson_path:
            self.agg[name].import_geometry(geojson_path)
     
        
    @property
    def columns(self):
        return self.adata.obs.columns.copy()
    
    
    def combine(self, other):
        '''
        Combines two HiVis objects into a single HiVis. Spatial plots and analysis will be disabled.
        '''
        return self + other 
            
    def copy(self, new_name=None, new_out_path=False, full=False):
        '''
        Creates a deep copy of the instance.
        if new_name is specified, renames the object and changes the path_output.
        If full is False, the name will be added to the current (previous) name.
        
        **Returns** new HiVis instance
        '''
        new = deepcopy(self)
        if new_name:
            new.rename(new_name, new_out_path=new_out_path, full=full)
        return new
    
    def crop(self,xlim=None,ylim=None):
        '''
        Creates a HiVis instance, cropped by given limits.
        
        Parameters:
            * xlim, ylim - [int,int]
            
        **Returns** new HiVis instance
        '''
        if xlim is None:
            xlim = self.plot.xlim_max
        if ylim is None:
            ylim = self.plot.ylim_max
        return self[(self["um_x"] > xlim[0]) & (self["um_x"] < xlim[1]) &
                    (self["um_y"] > ylim[0]) & (self["um_y"] < ylim[1]),:]
    
    
    def export_h5(self, path=None, force=False):
        '''
        Exports the adata as h5ad.
        
        Parameters:
            * path (str) - path to save the h5 file. If None, will save to path_output
            * force (bool) - save file even if it already exists
            
        **Returns** path where the file was saved (str)
        '''
        if path is None:
            path = self.path_output
        path = f"{path}/{self.name}_HiVis.h5ad"
        if not os.path.exists(path) or force:
            print("[Writing h5]")
            self.adata.write(path)
        return path
    
    def export_images(self, path=None, force=False):
        '''
        Exports full, high and low resolution images, and bins x,y coordinates, and the unit conversion json.
        
        Parameters:
            * path (str) - path to save the image files. If None, will save to path_output
            * force (bool)- save files even if they already exists
            
        **Returns** list of [3 images (np.array) and the spatial json]
        '''
        if path is None:
            path = self.path_output
        if not os.path.exists(path):
            os.makedirs(path)
        path_image_fullres = f"{path}/{self.name}_fullres.tif"
        image_fullres = self.image_fullres_orig if self.fluorescence else self.image_fullres
        path_image_highres = f"{path}/{self.name}_highres.tif"
        path_image_lowres = f"{path}/{self.name}_lowres.tif"
        images = HiVis_utils._export_images(path_image_fullres, path_image_highres, 
                                      path_image_lowres, image_fullres, 
                                      self.image_highres, self.image_lowres, force=force,
                                      um_per_pxl=self.json["microns_per_pixel"])
        
        path_json = f"{path}/{self.name}_scalefactors_json.json"
        with open(path_json, 'w') as file:
            json.dump(self.json, file, indent=4)
        
        cols = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres','pxl_col_in_fullres']
        path_obs = f"{path}/{self.name}_tissue_positions.csv"
        self.adata.obs[cols].to_csv(path_obs, index=True, index_label="barcode")
        
        images.append(self.json)
        
        return images
    
    
    def get(self, what, cropped=False, layer=None):
        '''
        Get a vector from data (a gene) or metadata (from obs or var). or subset the object.
        
        Parameters:
            * what - if string, will get data or metadata. else, will return a new HiVis object that is spliced. \
                     the splicing is passed to the self.adata
            * cropped - get the data from the adata_cropped after plotting spatial
            
        **Returns**: either np.array of data or, if subsetting, a new HiVis instance
        '''
        adata = self.adata_cropped if cropped else self.adata
        if isinstance(what, str):  # Easy access to data or metadata arrays
            if what in adata.obs.columns:  # Metadata from OBS
                column_data = adata.obs[what]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype("object").values
                return column_data.to_numpy()
            if what in adata.var.index:  # A gene
                gene_data = adata[:, what].X if layer is None else adata[:, what].layers[layer]
                return np.array(gene_data.todense().ravel()).flatten()
            if what in adata.var.columns:  # Gene metadata from VAR
                column_data = adata.var[what]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype("object").values
                return column_data.to_numpy()
            obs_cols_lower = adata.obs.columns.str.lower()
            if what.lower() in obs_cols_lower:
                col_name = adata.obs.columns[obs_cols_lower.get_loc(what.lower())]
                column_data = adata.obs[col_name]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype("object").values
                return column_data.to_numpy()
            if self.organism == "mouse" and (what.lower().capitalize() in adata.var.index):
                gene_name = what.lower().capitalize()
                gene_data = adata[:, gene_name].X if layer is None else adata[:, gene_name].layers[layer]
                return  np.array(gene_data.todense().ravel()).flatten()
            if self.organism == "human" and (what.upper() in adata.var.index):
                gene_name = what.lower().upper()
                gene_data = adata[:, gene_name].X if layer is None else adata[:, gene_name].layers[layer]
                return  np.array(gene_data.todense().ravel()).flatten()
            var_cols_lower = adata.var.columns.str.lower()
            if what.lower() in var_cols_lower:
                col_name = adata.var.columns[var_cols_lower.get_loc(what.lower())]
                column_data = adata.var[col_name]
                if column_data.dtype.name == 'category':  # Handle categorical dtype
                    return column_data.astype("object").values
                return column_data.to_numpy()
        else:
            # Create a new HiVis object based on adata subsetting
            return self.subset(what, remove_empty_pixels=False)
            
    
    def head(self, n=5):
        '''**Returns** HiVis.adata.obs.head(n), where n is number of rows'''
        return self.adata.obs.head(n)
    
    def merge(self, adata, obs=None, var=None, layer=None ,umap=True, pca=True, hvg=True, obsm=None, uns=None):
        '''
        Merge info from an anndata to self.adata, in case genes have been filtered.
        
        Parameters:
            * adata (ad.AnnData) - anndata where to get the values from
            * obs - single string or list of obs to merge
            * var - single string or list of var to merge
            * layer - single string or list of layers to merge
            * umap (bool) - add umap to OBSM, and UMAP coordinates to obs
            * pca (bool) - add PCA to OBSM
            * hvg (bool) - add highly variable genes to vars
        '''
        
        if not obs:
            obs = []
        elif isinstance(obs, str):
            obs = [obs]
        if not var:
            var = []
        elif isinstance(var, str):
            var = [var]
        if isinstance(obsm, str):
            obsm = [obsm]
        if isinstance(uns, str):
            uns = [uns]   
            
        if umap and "X_umap" in adata.obsm:
            if self.adata.shape[0] == adata.shape[0]:
                self.adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
            else:
                print("Cant add UMAP to obsm, size of adatas don't match")
            umap_coords = adata.obsm['X_umap']
            adata.obs['UMAP_1'] = umap_coords[:, 0]
            adata.obs['UMAP_2'] = umap_coords[:, 1]
            
            obs += ['UMAP_1','UMAP_2']
        if pca and "X_pca" in adata.obsm:
            if self.adata.shape[0] == adata.shape[0]:
                self.adata.obsm['X_pca'] = adata.obsm['X_pca'].copy()
        if hvg and 'highly_variable' in adata.var.columns:
            if not var:
                var = ['highly_variable']
            else:
                if 'highly_variable' not in var:
                    var += ['highly_variable']
        if layer is not None:
            if layer in adata.layers:
                if self.adata.shape[0] == adata.shape[0]:
                    self.adata.layers[layer] = adata.layers[layer].copy()
                else:
                    print(f"Can't add layer {layer} to self.adata.layers, size of adatas don't match")
            else:
                print(f"Layer {layer} not found in the provided adata.")
        if obs:
            existing_columns = [col for col in obs if col in self.adata.obs.columns]
            if existing_columns:
                self.adata.obs.drop(columns=existing_columns, inplace=True)
            self.adata.obs = self.adata.obs.join(adata.obs[obs], how="left")
            
            HiVis_utils._convert_bool_columns_to_float(self.adata.obs)
        if var:
            existing_columns = [col for col in var if col in self.adata.var.columns]
            if existing_columns:
                self.adata.var.drop(columns=existing_columns, inplace=True)
            self.adata.var = self.adata.var.join(adata.var[var], how="left")
            
            HiVis_utils._convert_bool_columns_to_float(self.adata.var)
        
        if uns:
            for u in uns:
                self.adata.uns[u] = adata.uns[u].copy()
        if obsm:
            for o in obsm:
                self.adata.obsm[o] = adata.obsm[o].copy()
                
    def recolor(self, fluorescence=None, normalization_method="percentile"):
        '''
        Recolors a flurescence image
        
        Parameters:
            * fluorescence is either list of colors or dict {channel: color...}. color can be None.
            * normalization_method - {"percentile", "histogram","clahe","sqrt" or None for minmax}
        '''
        if not self.fluorescence:
            raise ValueError("recolor() works for fluorescence visium only")
        if not fluorescence:
            fluorescence = self.fluorescence
        channels = list(self.fluorescence.keys())    
        if isinstance(fluorescence, list):
            if len(fluorescence) != len(channels):
                raise ValueError(f"Flurescence should include all channels: {channels}")
            self.fluorescence = {channels[i]:fluorescence[i] for i in range(len(channels))}
        elif isinstance(fluorescence, dict):
            if list(fluorescence.keys()) != channels:
                raise ValueError(f"Flurescence should include all channels: {channels}")
            self.fluorescence = fluorescence
        self.image_fullres = HiVis_utils.fluorescence_to_RGB(self.image_fullres_orig, 
                                                                self.fluorescence.values(), 
                                                                normalization_method)
        self.plot._init_img()
    
    
    def remove_pixels(self, column: str, values: list, marging=1):
        '''
        Removes pixels in images, based on adata.obs[column].isin(values).
        
        Parameters:
            * marging (int) - how many pixels to extend the removed pixels.
            
        **Returns** new HiVis instance.
        '''
        
        # Identify which pixels to remove based on the given condition
        obs_values = self.adata.obs[column]
        remove_mask = obs_values.isin([v for v in values if not pd.isna(v)])
        if any(pd.isna(v) for v in values): # Handle NaNs
            remove_mask |= obs_values.isna()
    
        # Determine the background color: Black (0) if fluorescence images, white (255) otherwise
        if self.fluorescence:
            img_fullres_new = self.image_fullres_orig.copy() if self.image_fullres_orig is not None else None
            background_value = 0
        else:
            img_fullres_new = self.image_fullres.copy() if self.image_fullres is not None else None
            background_value = 255
        img_highres_new = self.image_highres.copy() if self.image_highres is not None else None
        img_lowres_new = self.image_lowres.copy() if self.image_lowres is not None else None
    
        # Extract spot diameter and compute corresponding sizes for each resolution
        spot_diameter_fullres = self.json['spot_diameter_fullres']
        # For indexing, we need integer sizes
        
        from math import ceil
        spot_size_fullres = int(ceil(spot_diameter_fullres))
        spot_size_hires = int(ceil(spot_diameter_fullres * self.json['tissue_hires_scalef']))
        spot_size_lowres = int(ceil(spot_diameter_fullres * self.json['tissue_lowres_scalef']))
        # Ensure sizes are at least 1
        spot_size_fullres = max(spot_size_fullres, 1)
        spot_size_hires = max(spot_size_hires, 1)
        spot_size_lowres = max(spot_size_lowres, 1)
        
    
        # The image info tuples as before
        img_info = [
            (img_fullres_new, "pxl_col_in_fullres", "pxl_row_in_fullres", spot_size_fullres),
            (img_highres_new, "pxl_col_in_highres", "pxl_row_in_highres", spot_size_hires),
            (img_lowres_new, "pxl_col_in_lowres", "pxl_row_in_lowres", spot_size_lowres)
        ]
    
        images = []
        for i, (img_new, col_name, row_name, spot_size) in enumerate(img_info):
            if img_new is not None:
                pxl_cols = self.adata.obs[col_name].values.astype(int)
                pxl_rows = self.adata.obs[row_name].values.astype(int)
                half_spot = spot_size // 2 + marging # +1 is marging
    
                # Instead of removing one pixel, remove a square region
                for idx, to_remove in enumerate(remove_mask):
                    if to_remove:
                        r = pxl_rows[idx]
                        c = pxl_cols[idx]
    
                        # Compute the boundaries of the square, clamped within image bounds
                        top = max(r - half_spot, 0)
                        bottom = min(r + half_spot + 1, img_new.shape[0])  # +1 to include the boundary
                        left = max(c - half_spot, 0)
                        right = min(c + half_spot + 1, img_new.shape[1])
    
                        # Set the entire block to background_value
                        img_new[top:bottom, left:right, :] = background_value
            images.append(img_new)
        
        # Create a new object with the modified images
        name = self.name + "_edited" if not self.name.endswith("_edited") else self.name
        new_obj = HiVis(self.adata.copy(),images[0],images[1],
                           images[2],self.json,name,self.path_output,
                           properties=self.properties.copy(),
                           fluorescence=self.fluorescence.copy() if self.fluorescence else None)        
        return new_obj
    
    def rename(self, new_name: str, new_out_path=False, full=False):
        '''
        Renames the object and changes the path_output.
        If full is False, the name will be added to the current (previous) name
        '''
        if full:
            self.name = new_name
        else:
            self.name = self.name.replace("_subset","")
            self.name = f"{self.name}_{new_name}"
        if new_out_path:
            self.path_output = self.path_output + f"/{new_name}"
    
    
    def save(self, path=None):
        '''
        Saves the instance in pickle format.
        If no path specified, will save in the path_output as the name of the instance.
        
        **Returns** the path of the file (str)
        '''
        print(f"SAVING [{self.name}]")
        if not path:
            path = f"{self.path_output}/{self.name}.pkl"
        else:
            if not path.endswith(".pkl"):
                path += ".pkl"
        self.plot.current_ax = None
        if self.agg:
            for agg in self.agg:
                self.agg[agg].plot.current_ax = None

        with open(path, "wb") as f:
            dill.dump(self, f)            
        return path
    
    @property
    def shape(self):
        return self.adata.shape
    
    
    def subset(self, what=(slice(None), slice(None)), remove_empty_pixels=False, crop_agg=True):
        '''
        Create a new HiVis objects based on adata subsetting.
        
        Parameters:
            - what (tuple) - tuple of two elements. slicing instruction for adata. examples:
                - (slice(None), slice(None)): Select all spots and all genes.
                - ([0, 1, 2], slice(None)): Select the first three spots and all genes.
                - (slice(None), ['GeneA', 'GeneB']): Select all spots and specific genes.
                - (adata.obs['obs1'] == 'value', slice(None)): Select spots where 
                  the 'obs1' column in adata.obs is 'value', and all genes.
            - remove_empty_pixels (bool) - if True, the images will only contain pixels under bins
            - crop_agg (bool) - crop Agg objects. If False, plotting of aggregations might break.
            
        **Returns** new HiVis instance
        '''
        adata = self.adata[what].copy()
        name = self.name + "_subset" if not self.name.endswith("_subset") else ""
        if self.image_fullres is not None:
            image_fullres_crop, image_highres_crop, image_lowres_crop, xlim_pixels_fullres, ylim_pixels_fullres = self.__crop_images(adata, remove_empty_pixels)
            adata_shifted = self.__shift_adata(adata, xlim_pixels_fullres, ylim_pixels_fullres)
        else:
            image_fullres_crop, image_highres_crop, image_lowres_crop = None, None, None
            adata_shifted = adata
        # remove columns from previous analyses
        adata_shifted.var = adata_shifted.var.loc[:,~adata_shifted.var.columns.str.startswith(("cor_","exp_"))]
        adata_shifted.var = adata_shifted.var.drop(columns=[col for col in ["residual","cv","expression_mean","cv_log10","mean_log"] if col in adata_shifted.var.columns])
        
        new_obj = HiVis(adata_shifted, image_fullres_crop, image_highres_crop, 
                           image_lowres_crop, self.json, name, self.path_output,agg=None,plot_qc=False,
                           properties=self.properties.copy(),fluorescence=self.fluorescence.copy() if self.fluorescence else None)    
        # HiVis.HiVis_utils.add_spatial_keys(new_obj, new_obj)
        # update the link in all aggregations to the new HiVis instance
        if self.agg: 
            for agg in self.agg:
                if crop_agg:
                    adata_agg = self.agg[agg].adata.copy()
                    idx_col = adata_agg.obs.index.name
                    obs_mask = adata_agg.obs.index.isin(adata_shifted.obs[idx_col])
                    common_genes = adata_agg.var_names.intersection(adata_shifted.var_names)
                    adata_agg_shifted = adata_agg[obs_mask, common_genes]
                    # remove columns from previous analyses
                    adata_agg_shifted.var = adata_agg_shifted.var.loc[:,~adata_agg_shifted.var.columns.str.startswith(("cor_","exp_"))]
                    adata_agg_shifted.var = adata_agg_shifted.var.drop(columns=[col for col in ["residual","cv","expression_mean","cv_log10","mean_log"] if col in adata_agg_shifted.var.columns])

                    adata_agg_shifted = self.__shift_adata(adata_agg_shifted, xlim_pixels_fullres, ylim_pixels_fullres)
                else:
                    adata_agg_shifted = self.agg[agg].adata
                if not adata_agg_shifted.shape[0] == 0:
                    new_obj.add_agg(adata_agg_shifted.copy(),agg)
                else:
                    print(f"Aggregation [{agg}] is empty")
        return new_obj
    
    def to_spatialdata(self):
        import pandas as pd, geopandas as gpd
        from shapely.geometry import Point
        from spatialdata.models import ShapesModel
        from spatialdata.models import Image2DModel
        import spatialdata as sd
        import re
        
        def clean_name(s):
            s = str(s).replace("µ", "u").replace("μ", "u")
            return re.sub(r"[^0-9A-Za-z_.-]", "_", s)
 
        
        adata = self.adata.copy()
        adata.obs.columns = [clean_name(c) for c in adata.obs.columns]
        adata.obs = adata.obs.assign(**{
            c: adata.obs[c].astype("string").fillna("")
            for c in adata.obs.select_dtypes(include=["object", "string", "category"])
        })
        img = self.image_fullres_orig if self.fluorescence else self.image_fullres
        
        img = img.transpose(2, 0, 1)  # now shape is (3, y, x)

        img_model = Image2DModel.parse(data=img, scale_factors=(2, 2, 2))
        centers = list(zip(adata.obs["pxl_col_in_fullres"],  # x coordinates
                           adata.obs["pxl_row_in_fullres"])) # y coordinates
        radius = (self.json["spot_diameter_fullres"] / 2)  
        df = pd.DataFrame([radius] * len(centers), columns=["radius"])
        gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in centers])
        shapes = ShapesModel.parse(gdf)

        if "spatial" in adata.uns.keys():
            del adata.uns["spatial"]
        if "spatial" in adata.obsm.keys():
            del adata.obsm["spatial"]
        from spatialdata.models import TableModel
        adata_table = TableModel.parse(adata)
        adata_table.uns["spatialdata_attrs"] = {"region": "spots","region_key": "region","instance_key": "spot_id"}
        adata_table.obs["region"] = pd.Categorical(["spots"] * adata_table.n_obs)
        adata_table.obs["spot_id"] = shapes.index  # align each obs with the shapes GeoDataFrame index
        sdata = sd.SpatialData(images={self.name: img_model}, shapes={"spots": shapes},tables={"adata": adata_table},)
        return sdata
    
    
    def update(self, agg=False):
        '''Updates the methods in the instance. Should be used after modifying the source code in the class'''
        HiVis_utils.update_instance_methods(self)
        HiVis_utils.update_instance_methods(self.plot)
        HiVis_utils.update_instance_methods(self.analysis)
        self.plot._init_img()
        if agg and self.agg:
            for agg in self.agg:
                self.agg[agg].update()
        else:
            _ = gc.collect()
    
    
    def update_meta(self, name:str, values:dict, type_="obs"):
        r'''
        Updates values in metadata (obs or var)
        
        Parameters:
            * name (str) - name of metadata
            * values (dict) -{old_value:new_value}
            * type\_ - either "obs" or "var"
        '''
        if type_ == "obs":
            if name not in self.adata.obs.columns:
                raise ValueError(f"No metadata called [{name}] in obs")
            original_dtype = self.adata.obs[name].dtype
            self.adata.obs[name] = self.adata.obs[name].apply(lambda x: values.get(x, x) if pd.notna(x) else x)
            
            # Convert back to original dtype if it was categorical
            if pd.api.types.is_categorical_dtype(original_dtype):
                self.adata.obs[name] = self.adata.obs[name].astype('category') 
        elif type_ == "var":
            if name not in self.adata.var.columns:
                raise ValueError(f"No metadata called [{name}] in var")
            original_dtype = self.adata.var[name].dtype
            self.adata.var[name] = self.adata.var[name].apply(lambda x: values.get(x, x) if pd.notna(x) else x)
            
            # Convert back to original dtype if it was categorical
            if pd.api.types.is_categorical_dtype(original_dtype):
                self.adata.var[name] = self.adata.var[name].astype('category')    
        else:
            raise ValueError("type_ must be either 'obs' or 'var'")
        self.plot._init_img()
        

    def __add__(self, other):
        '''Combines two HiVis objects into a single HiVis object. Some methods will be disabled.'''
        if not str(type(other)) == str(type(self)):
            raise ValueError("Addition supported only for HiVis class")
        self.adata.obs["source_"] = self.name
        other.adata.obs["source_"] = other.name if other.name != self.name else f"{self.name}_1"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            adata = ad.concat([self.adata, other.adata], join='outer')
        del self.adata.obs["source_"]
        adata.obs_names_make_unique()
        
        name = "combined"
        properties = {**self.properties, **other.properties}
        new_obj = HiVis(adata, image_fullres=None, image_highres=None, image_lowres=None,
                        scalefactor_json=None, name=name, 
                        path_output=self.path_output,agg=None,plot_qc=False,
                        properties=properties,fluorescence=None)    
        return new_obj    


    def __contains__(self, what):
        if (what in self.adata.obs) or (what in self.adata.var) or (what in self.adata.var_names):
            return True
        if self.agg:
            if what in self.agg:
                return True
        return False


    def __crop_images(self, adata, remove_empty_pixels=False):
        '''
        Helper function for get().
        Crops the images based on the spatial coordinates in a subsetted `adata` 
        and adjusts the adata accordingly (shifts x, y)
        remove_empty_pixels - whether to remove pixels that dont have spots on them.
        '''
        # Crop images
        def _crop_img(adata, img, col, row):
            '''crops one image by the x,y values in adata.obs, as specified by col, row'''
            pxl_col = adata.obs[col].values
            pxl_row = adata.obs[row].values
            xlim_pixels = [int(np.floor(pxl_col.min())), int(np.ceil(pxl_col.max()))]
            ylim_pixels = [int(np.floor(pxl_row.min())), int(np.ceil(pxl_row.max()))]
            # Ensure the limits are within the image boundaries
            xlim_pixels = [max(0, xlim_pixels[0]), min(img.shape[1], xlim_pixels[1])]
            ylim_pixels = [max(0, ylim_pixels[0]), min(img.shape[0], ylim_pixels[1])]
            if xlim_pixels[1] <= xlim_pixels[0] or ylim_pixels[1] <= ylim_pixels[0]:
                raise ValueError("Invalid crop dimensions.")
            img_crop = img[ylim_pixels[0]:ylim_pixels[1],xlim_pixels[0]:xlim_pixels[1],:].copy()
            
            if remove_empty_pixels:
                # remove pixels in images that don't have spots
                pxl_cols_shifted = pxl_col - xlim_pixels[0]
                pxl_rows_shifted = pxl_row - ylim_pixels[0]
                mask = np.zeros((img_crop.shape[0], img_crop.shape[1]), dtype=bool)
                for cx, cy in zip(pxl_cols_shifted, pxl_rows_shifted):
                    # Ensure we only mark valid coordinates within the cropped image
                    cx = int(cx)  
                    cy = int(cy)
                    if 0 <= cx < img_crop.shape[1] and 0 <= cy < img_crop.shape[0]:
                        mask[cy, cx] = True
                # Set non-adata remove pixels that are not covered by spots
                background_value = 0 if self.fluorescence else 255 # black for fluorescence, white for RGB
                img_crop[~mask] = background_value
            return img_crop, xlim_pixels, ylim_pixels
        
        image_fullres = self.image_fullres_orig if self.fluorescence else self.image_fullres
        image_fullres_crop, xlim_pixels_fullres, ylim_pixels_fullres = _crop_img(adata, image_fullres, "pxl_col_in_fullres", "pxl_row_in_fullres")
        image_highres_crop , _ , _ = _crop_img(adata, self.image_highres, "pxl_col_in_highres", "pxl_row_in_highres")
        image_lowres_crop , _ , _ = _crop_img(adata, self.image_lowres, "pxl_col_in_lowres", "pxl_row_in_lowres")

        return image_fullres_crop, image_highres_crop, image_lowres_crop,xlim_pixels_fullres, ylim_pixels_fullres
    
    
    def __delitem__(self, key):
        '''deletes metadata'''
        if isinstance(key, str):
            if key in self.adata.obs:
                del self.adata.obs[key]
            elif key in self.adata.var.columns:
                del self.adata.var[key]
            else:
                raise KeyError(f"'{key}' not found in adata.obs")
            self.plot._init_img()
        else:
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
            

    def __getitem__(self, what):
        '''Get a vector from data (a gene) or metadata (from obs or var). or subset the object.'''
        item = self.get(what, cropped=False)
        if item is None:
            raise KeyError(f"[{what}] isn't in data or metadatas")
        return item
    

    def __repr__(self):
        # s = f"HiVis[{self.name}]"
        s = self.__str__()
        return s
    
    
    def __setitem__(self, key, value):
        if not hasattr(value, '__len__'):
            raise ValueError("Assigned value must be iterable or array-like")
        if len(value) == self.adata.shape[0]:
            self.adata.obs[key] = value
        elif len(value) == self.adata.shape[1]:
            self.adata.var[key] = value
        else:
            raise ValueError("Values must be in the length of OBS or VAR")
        self.plot._init_img()
        
        
    def __shift_adata(self, adata, xlim_pixels_fullres, ylim_pixels_fullres):
        """
        Shifts the coordinates in an adata, based on xlim, ylim (in pixel space). \
        Also shifts the geometry WKT in micron space.
        """
        adata_shifted = adata.copy()
        drop_columns = ["pxl_col_in_lowres","pxl_row_in_lowres",
                        "pxl_col_in_highres","pxl_row_in_highres",
                        "um_x","um_y"]
        adata_shifted.obs.drop(columns=drop_columns, inplace=True, errors="ignore")
    
        # Shift the coordinates
        adata_shifted.obs["pxl_col_in_fullres"] -= xlim_pixels_fullres[0]
        adata_shifted.obs["pxl_row_in_fullres"] -= ylim_pixels_fullres[0]
    
        # Shift the geometry in micron space
        if "geometry" in adata_shifted.obs.columns:
            x_offset_microns = xlim_pixels_fullres[0] * self.json["microns_per_pixel"]
            y_offset_microns = ylim_pixels_fullres[0] * self.json["microns_per_pixel"]
    
            def _shift_wkt_geometry(geom_wkt):
                if isinstance(geom_wkt, str) and geom_wkt.strip():
                    geom = wkt.loads(geom_wkt)
                    geom = affinity.translate(geom, xoff=-x_offset_microns, yoff=-y_offset_microns)
                    return geom.wkt  # Store back as WKT
                return np.nan
    
            adata_shifted.obs["geometry"] = (
                adata_shifted.obs["geometry"]
                .fillna("")
                .apply(_shift_wkt_geometry)
            )
    
        return adata_shifted   

        
    def __str__(self):
        s = f"# {self.name} #\n"
        if hasattr(self, "organism"): s += f"\tOrganism: {self.organism}\n"
        if hasattr(self, "organ"): s += f"\tOrgan: {self.organ}\n"
        if hasattr(self, "sample_id"): s += f"\tID: {self.sample_id}\n"
        s += f"\tSize: {self.adata.shape[0]} x {self.adata.shape[1]}\n"
        s += '\nobs: '
        s += ', '.join(list(self.adata.obs.columns))
        s += '\n\nvar: '
        s += ', '.join(list(self.adata.var.columns))
        if self.agg:
            s += '\n\nAggregations:\n'
            for agg in self.agg:
                s += f"[{agg}]\tshape: {self.agg[agg].adata.shape[0]} x {self.agg[agg].adata.shape[1]}\n"
        return s
    