# QuPath Workflows for HiVis

For general Qupath tutorial see the [documentation](https://qupath.readthedocs.io/en/stable/index.html).

## Overview

Good segmentation is important for achieving high quality aggregation of transcript bins into cells or anatomical regions.  
Here we provide scripts that implement segmentation workflows for different datasets.  
Start with baseline workflow that suits your data type and tune it according to the specific tissue and image sample quality.

Some preliminary tasks are needed prior to running the workflows for selecting regions of interest either manually or by training pixels classifiers for anatomical regions.  
Some of the workflows use object classifiers to discard artifacts which are falsely recognized as cells.

---
This tutorial includes:
- [Cell segmentation - based on nuclei only (H&E or fluorescence, using Stardist)](#cell-segmentation---based-on-nuclei-only-he-or-fluorescence-using-stardist)
- [Cell segmentation - based on multiple channels (fluorescence, using Cellpose or InstaSeg)](#cell-segmentation---based-on-multiple-channels-fluorescence-using-cellpose-or-instaseg)
- [Dedicated cell segmentation for mouse liver - based on multiple channels and different segmentation for hepatocytes and epithelial cells](#Dedicated-cell-segmentation-for-mouse-liver---based-on-multiple-channels-and-different-segmentation-for-hepatocytes-and-epithelial-cells) 
  - [Input Data](#input-data)
  - [Selecting regions of interest using Manual Annotations](#selecting-regions-of-interest-using-manual-annotations)
  - [Training Pixel Classifier for Automatic segmentation of anatomical regions](#training-pixel-classifier-for-automatic-segmentation-of-anatomical-regions)
  - [Training Object Classifiers for discarding artifacts](#training-object-classifiers-for-discarding-artifacts)
  - [Training your own cell segmentation model](#training-your-own-cell-segmentation-model)
- [Technical details of implementation](#technical-details-of-implementation)
  - [Cell Segmentation](#cell-segmentation)
  - [Associating Bins to Cells](#associating-bins-to-cells)
  - [QuPath implementation](#qupath-implementation)
- [References](#references)

---

## Cell segmentation - based on nuclei only (H&E or fluorescence, using Stardist)

Download and install [Stardist extension for QuPath](https://github.com/qupath/qupath-extension-stardist).

For H&E data use the [StarDist script](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/VisiumHDAnalysis_StarDist_AnatomicalRegions_FilterNuc.groovy).

For fluorescent data use the [fluorescence script](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/VisiumHDAnalysis_Flourescent.groovy).

Import the segmentation into Python and aggregate gene expression with `HiVis.agg_cells()`.

### Baseline Workflow
- Detect the Whole tissue or Anatomical regions automatically or manually select regions of interest
- Segment Cells within each anatomical region, using the nuclei stain followed by expansion
- Load the VisiumHD bins position and associate bins to cells
- Save results in HiVis compatible formats:
  - Export the cell borders as `.geoJson` file
  - Export Anatomical regions borders as `.geoJson` file
  - Save `.csv` file with both cells and bins information

### Advanced options supported by the baseline workflow
- Restrict expansion within anatomical regions
- Requires a trained pixel classifier or manual annotation
- Region-specific expansion size
- Region-specific segmentation parameters (e.g., detection probability threshold)
- Nuclei classification–based expansion
- Use a trained object classifier to apply different parameters to different cell types
- Artifact filtering
- Train an object classifier to distinguish nuclei vs artifacts (helpful for H&E)

### Scripts Parameters

**Control parameters (0 or 1):**
- `segmentTissue` — Segment the tissue, and not allow the detection of nuclei outside it. Also limits the expansion of cells to the tissue area.
- `segmentAnatomicalRegions` — Segment anatomical regions to allow varying expansion of nuclei in each region.
- `segmentCells` — Segment cells (`1`) or only nuclei (`0`).
- `filterNucBeforeCellExpansion` — Filter artefact nuclei prior to expansion.
- `AddMeasurementsToCells` — Add measurements and channels intensities to cells.
- `loadSpots` — Load VisiumHD bins.
- `associateSpotsToCells` — Associate spots to cells.
- `runPixelClassifierForSpot` — Give each bin an identity based on the pixel classifier.
- `runPixelClassifierForCell` — Give each cell an identity based on the pixel classifier.
- `exportCellsAsGeoJson` — Export geometry of cells.
- `exportAnnotationsAsGeoJson` — Export geometry of annotations.
- `saveResultTable` — Save the results as a CSV file.

**General parameters:**
- `scalefactors_json` — Path of `scalefactors_json.json` file containing scale factors (exported by `HiVis.export_images()`).
- `csvfile` — Path of `tissue_positions.csv` file containing bins positions (exported by `HiVis.export_images()`).
- `WholeTissueClassifier` — Name of WholeTissue pixel classifier.
- `wholeTissueClass` — Name of whole-tissue class in the WholeTissue pixel classifier.
- `WholeTissue_MinSize` — Minimal WholeTissue connected-component size.
- `WholeTissue_MinHoleSize` — Minimal hole size to keep when creating WholeTissue regions; smaller holes are filled.
- `PixelClassifier` — Name of pixel classifier, if `runPixelClassifier*` were set to `1`.

**Stardist parameters:**
- `StarDistPathModel` — Path for Stardist model, such as `he_heavy_augment.pb`.
- `param_threshold` — Threshold for detection. All cells segmented by StarDist will have a detection probability associated with it, where higher values indicate more certain detections. Floating point, range is `0` to `1`. Default `0.5`.
- `normalize_low_pct` — Lower limit for normalization. Set to `0` to disable.
- `normalize_high_pct` — Upper limit for normalization. Set to `100` to disable.
- `param_tilesize` — Size of tile in pixels for processing. Must be a multiple of 16. Lower values may solve memory-related errors, but can take longer to process. Default is `1024`.
- `PositiveNegativeNucClassifier` — **OPTIONAL** name of object classifier to filter nuclei prior to expansion.

**Expansion parameters:**
- `AnatomicalRegionsClassNames` — List of class names of the different anatomical regions.
- `AnatomicalRegionsExpansionMicrons` — List of expansion values (in µm) for each of the anatomical regions.


## Cell segmentation - based on multiple channels (fluorescence, using Cellpose or InstaSeg)

If both cell and nuclei markers are available, you can use segmentation methods that accurately detect cell borders, such as:
- [InstaSeg](https://github.com/instanseg/instanseg)
- [Cellpose](https://github.com/MouseLand/cellpose)  

Both come with built-in models that perform well in many cases.  
If needed, train your own model for improved accuracy on specific tissue types.

The **Baseline workflow** and **advanced options** are similar to the nuclei-only workflow described above.

### Additional Parameters

#### **Segmentation and Channels**
- **cellSegmentationMethod** — Defines how cell boundaries are determined relative to nuclei.  
  *Possible values:* `"cellBorders"`, `"expandNuc"`, `"onlyNuc"`.
- **segmentationAlg** — Algorithm used for segmentation.  
  *Possible values:* `"cellpose"`, `"stardist"`, `"instaseg"`.
- **nucChannel** — Image channel used for nuclei detection (e.g., `"Channel 4"` for Tonsil).
- **membraneChannels** — Channels used to identify cell membranes (e.g., `["Channel 1", "Channel 3"]` for Tonsil).

#### **Cellpose Parameters**
- **useCellposeSAM** — Use SAM (Segment Anything Model) integration in Cellpose (`1` to enable, `0` to disable).  
- **CellposeNucModel** — Name of model used for nuclei segmentation (e.g., `"nuc"`, `"cpsam"`).  
- **CellposeCellModel** — Name of model used for cell segmentation (e.g., `"cyto3"`, `"cpsam"`) or path to a custom `.cpm` model file.  
- **CellposeCellDiameter** — Approximate cell diameter in pixels (e.g., `15`, `50`).  
- **CellposeNucDiameter** — Approximate nucleus diameter in pixels (e.g., `15`, `21`).  

#### **StarDist Parameters**
- **StarDistPathModel** — Full path to `.pb` Stardist model file.  
- **param_threshold** — Detection probability threshold (`0.0–1.0`); higher values yield more confident cell detections.  
- **normalize_low_pct** — Lower percentile for normalization (e.g., `0–5`); set `0` to disable.  
- **normalize_high_pct** — Upper percentile for normalization (e.g., `95–100`); set `100` to disable.  
- **param_tilesize** — Tile size (in pixels) for parallel processing; must be a multiple of 16 (default `1024`). Smaller values reduce memory use but increase runtime.  

#### **InstaSeg Parameters**
- **InstaSegModel** — Full path to InstaSeg model used for nuclei and cell segmentation.  
- **InstaSeg_tileDims** — Tile size (in pixels) for processing (e.g., `1024`).  
- **InstaSeg_interTilePadding** — Overlap between tiles to avoid edge artifacts (e.g., `32`).  
- **InstaSeg_nThreads** — Number of processing threads.  
- **InstaSeg_device** — Compute device to use for segmentation.  
  *Possible values:* `"gpu"`, `"cpu"`.  


---
## Dedicated cell segmentation for mouse liver - based on multiple channels and different segmentation for hepatocytes and epithelial cells

The [mouse liver script](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/VisiumHDAnalysis_DoubleNucleatedCells_v1.groovy).
implements a more complex workflow designed specifically to achieve accurate cell segmentation of mouse liver tissue, in which hepatocytes are very big, and can be mono-nucleated or double-nucleated, and in some cases the nuclei is not seen at the imaged slice.  
 
### Workflow
- Segment Tissue region using pixel classifier - blood vessels / empty / tissue 
  Filter out objects by size/shape
- Segment hepatocytes Cells and Nucs within Tissue using Cellpose 
  Use a cellpose model trained on double-nucleated cells for cells, and cyto3 for Nuc
- Associate Nucs to cells (0/1/2)
  keep the Nucs inside cells as separate objects, with parent id encoded in their name
- Segment cells in blood-vesseles regions using Stardist based on DAPI+expansion 
- Load VisiumHD spots (bins) as detections
- Associate spots to Cells and Nucs, set the inCell/inNuc flag
- For each spot and cell calculate pixel classifier probability 
- Measure distance of cells/nucs/spots from anatomical regions (created by the pixel classifier)
- Measure distance of spots to the parent cell border and to its nuc
- Export cell and nuc ROIs as geojson
- Export features for all detections : cell/nuc/spots into csv table


## Get Prepared

### Input Data

- **High-resolution image taken by a microscope**
- **Parquet tissue position matrix**: 'tissue_positions.parquet'  
  This file is located in the '/outs/binned_outputs/square_002um/spatial' directory of a completed Space Ranger run.
  It contains a table of the location of each 2x2 µm barcode square where each column is a barcode. 

Convert it to '.csv' for QuPath compatibility: 
```python
import pandas as pd 
path = “tissue_positions.parquet”   
metadata = pd.read_parquet(path) 
metadata.to_csv(path.replace(".parquet",".csv"),index=False)
```
- **Scaling information file**: 'scalefactors_json.json'
- **2x2 µm filtered barcode matrix**: 'filtered_feature_bc_matrix.h5'  
  This file is NOT used by the QuPath scripts, but only by the HiVis python library.
  Located in '/outs/binned_outputs/square_002um' directory of a completed Space Ranger run.
  This 'h5' file stores the number of UMIs associated with a feature or gene (rows), for each barcode (columns). 

### Selecting regions of interest using Manual Annotations

In the annotation tab, select the class you want to annotate and click "set selected".
Then annotate with any of the [annotation tools](https://qupath.readthedocs.io/en/stable/docs/starting/annotating.html#annotation-tools).

### Training Pixel Classifier for Automatic segmentation of anatomical regions

Follow the [official Qupath pixel classifier tutorial](https://qupath.readthedocs.io/en/stable/docs/tutorials/pixel_classification.html).

When selecting parametes, we suggest to choose:
* Type: Random Trees
* Resolution: moderate (but can vary)
* Features: all available features 
* Scales: [1,2,4,8] (but can vary)
* Channels:
	- For H&E: Hematoxylin, Eosin
	- For fluorescence: choose only the relevant fluorescence channels
* No normalization
* Output: Classification

### Export
To export the classifier result as a label image (mask), run the script 
[ExportPixelClassifierAsLabelImage.groovy](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/ExportPixelClassifierAsLabelImage.groovy) 
with downsample=1.

### Import into HiVis
Classifier can be imported into Python with [HiVis.add_mask()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.add_mask).
This assigns annotation for each bin.

### Training Object Classifiers for discarding artifacts

...

### Training your own cell segmentation model

Follow the instructions in [QuPath Cellpose extension](https://github.com/BIOP/qupath-extension-cellpose)
Here are few comments we find important : 
- Use duplicated images for training or even different project
- Select multiple ROIs that represent well the variability in the appearance of the cells in the data . 
	Select regions that contain multiple cells (eg 50-100 regions of 10-20 cells each).
	Use "Create region annotations…" with specified size and class (Training / Validation)
- For each such region run cell segmentation with your favourite model, use the flag "createAnnotations" to enable editing the segmentation
- Delete artifacts, and cell that are completely wrong. Correct segmentation when needed, using the different annotation tools. We recommend using Brush or Polygon. You can also use SAM extension 
- Make sure to be consistent when annotating (eg: is the cell membrane included or not )
- You must annotate all the cells within each of the selected ROIs

## Technical details of implementation

### Cell Segmentation

**If both cell and nuclei Borders are available**  
Use segmentation methods that accurately detect the cell border, such as:
- [InstaSeg](https://github.com/instanseg/instanseg)
- [Cellpose](https://github.com/MouseLand/cellpose)  
Both come with built-in models that perform well in many cases. If needed, train your own model.

**If only nuclei borders are available**  
Examples: H&E image or fluorescent DAPI-only staining.
- Cell borders are **approximated** by expanding the detected nuclei.
- Expansion is limited by:
  - A **maximum distance** from the nucleus
  - Until it touches a **neighboring expanding cell**

Nuclei segmentation options:
- [StarDist](https://github.com/stardist/stardist)
- [Cellpose](https://github.com/MouseLand/cellpose)
- [InstaSeg](https://github.com/instanseg/instanseg),

### Associating Bins to Cells

Association of Visium-HD bins to segmented cells is implemented **within QuPath**.

Bins are uploaded from the converted 'tissue_positions.csv' file 

**Bin-to-Cell Association Rules**
- A bin (spot) is associated to a cell (or nucleus) if its center is inside that region.
- For each **bin**:
  - Set Name to: 'BinBarcode__ParentCellObjectID'
  - Set:
    - inCell: 0 or 1
    - inNuc: 0 or 1
- For each **cell**, Set:
  - nSpots
  - nNucSpots
  
### QuPath implementation

QuPath has two object types:
- **Annotations**
  - Flexible and can contain child objects
- **Detections**
  - More efficient and suited for large number of objects
  - Cannot have children
  - `Cells` and `Tiles` are both subtypes of `Detections`

For millions of bins and hundreds of thousands of cells:
- Bins should be detections of type: `Tile`
- Cells should be detections (not annotations)

As a result, **QuPath's built-in parent-child hierarchy cannot be used**.
Instead, **parent cell ID is encoded in the bin’s** `Name` **field** as:
> BinBarcode__ParentCellObjectID

## References

**QuPath**: Bankhead P, Loughrey MB, Fernández JA, Dombrowski Y, McArt DG, Dunne PD, et al. QuPath: Open source software for digital pathology image analysis. Sci Rep. 2017 Dec 4;7(1):16878. 

**Stardist**: Weigert M, Schmidt U. Nuclei instance segmentation and classification in histopathology images with StarDist. 2022. Available from: http://arxiv.org/abs/2203.02284

**Cellpose**: Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods. 2021 Jan;18(1):100–6. 

**InstanSeg**: Goldsborough T, Philps B, O’Callaghan A, Inglis F, Leplat L, Filby A, et al. InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation. arXiv; 2024. Available from: http://arxiv.org/abs/2408.15954

