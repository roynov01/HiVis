# Guide for working with Qupath for HiVis

For general Qupath tutorial see the [documentation](https://qupath.readthedocs.io/en/stable/index.html).

## Overview
Good segmentation is important for achieving high quality aggregation of transcript bins into cells or anatomical regions. Start with baseline workflow that suits your data type and tune it according to the specific tissue and image sample quality. 

## Baseline workflow 

1. **Detect the Whole tissue or Anatomical regions** automatically (or manually)
2. **Segment Cells** within each anatomical region 
3. **Load the VisiumHD bins position and associate bins to cells**  
4. **Save results in HiVis compatible formats:**  
   - Export the **cell borders** as '.geoJson' file
   - Export **Anatomical regions** borders as '.geoJson' file
   - Save '.csv' file with both **cells** and **bins** information

## Input Data 

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

## Cell Segmentation  
**If both cell and nuclei Borders are available**  
Use segmentation methods that accurately detect the cell border, such as:
- [InstaSeg](https://github.com/instanseg/instanseg)
- [Cellpose](https://github.com/MouseLand/cellpose)  
Both come with built-in models that perform well in many cases. If needed, train your own model.

**Use this script:**
>VisiumHDAnalysis_Flourescent.groovy


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

**Advanced options supported by the baseline workflow**:
- **Restrict expansion within anatomical regions**
  - Requires a trained **pixel classifier** or **manual annotation**
- **Region-specific expansion size**
- **Region-specific segmentation parameters** (e.g., detection probability threshold)
- **Nuclei classification–based expansion**
  - Use a trained *object classifier* to apply different parameters to different cell types
- **Artifact filtering**
  - Train an object classifier to distinguish *nuclei vs artifacts* (helpful for H&E)

**Use this script for H&E data**:
> VisiumHDAnalysis_StarDist_AnatomicalRegions_FilterNuc.groovy

**Use this script for fluorescent data**:
> VisiumHDAnalysis_Flourescent.groovy

## Associating Bins to Cells  
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

## QuPath Implementation Details  
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

## Optional: Run Pixel Classifier Per Bin  
- Each spot receives measurements of the *percentage of its area* covered by each class
- Uses **hard decision classification**

## Manual annoatations
In the annotation tab, select the class you want to annotate and click "set selected".
Then annotate with any of the [annotation tools](https://qupath.readthedocs.io/en/stable/docs/starting/annotating.html#annotation-tools).
To export the annotations, select the annotations you want to export (ctrl+A for all), and click on File => Export objects as GeoJSON, leave the default options. 

## Train pixel classifier for Anatomical regions 
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
["ExportPixelClassifierAsLabelImage.groovy"](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/ExportPixelClassifierAsLabelImage.groovy)
 with downsample=1.

## Train your own model 
...

## Scripts Overview
  
| Script Filename                                      | Use Case                              |
|------------------------------------------------------|----------------------------------------|
| `VisiumHDAnalysis_Flourescent.groovy`                | Fluorescent images (cell + nuclei)     |
| `VisiumHDAnalysis_StarDist_AnatomicalRegions_FilterNuc.groovy` | H&E images or DAPI only + artifact filtering |
| `VisiumHDAnalysis_DoubleNucleatedCells_v1.groovy'    | Custom script for mouse liver Fluorescent images (segment heptocytes and their nuclei with cellpose, segment non-parenchymal cells with Stardist + expansion)     |

## Stardist
Download and install 
[Stardist extension for QuPath](https://github.com/qupath/qupath-extension-stardist).

## Cellpose
Download and install 
[Cellpose extension for QuPath](https://github.com/BIOP/qupath-extension-cellpose).

## References
QuPath Documentation
Cellpose
StarDist
InstaSeg

## License
MIT



  

