# Guide for working with Qupath for HiVis
For general Qupath tutorial see the [documentation](https://qupath.readthedocs.io/stable/index.html).

This tutorial includes:
1. Manual annotations
2. Implementing pixel classifier
3. Segmentation scripts - overview and parameters
4. Cells segmentation tools and options
5. Segmentation scripts workflow

<span style="color:red">**Important!**</span><br>
The image that should be used in Qupath is the cropped image that is created when creating a new HiVis object in Python.<br>
Alternetaviely, and highly advisible - call [HiVis.export_images()](https://hivis.readthedocs.io/latest/items.html#HiVis.HiVis.HiVis.export_images) - and use the exported fullres_image.tif.<br>
Similarly, for Qupath scripts, use the tissue_positions.csv file that is exported by [HiVis.export_images()](https://hivis.readthedocs.io/latest/items.html#HiVis.HiVis.HiVis.export_images).


## 1. Manual annotations
In the annotation tab, select the class you want to annotate and click "set selected".
Then annotate with any of the [annotation tools](https://qupath.readthedocs.io/stable/docs/starting/annotating.html#annotation-tools).

<u>Export</u><br>
To export the annotations, select the annotations you want to export (ctrl+A for all), and click on File => Export objects as GeoJSON, leave the default options. 

<u>Import into HiVis</u><br>
Annotations can be imported into Python with [HiVis.add_annotations()](https://hivis.readthedocs.io/latest/items.html#HiVis.HiVis.HiVis.add_annotations).
This assigns annotation for each bin.<br>
You can create an "Aggregation" object from the annotations by calling [HiVis.agg_from_annotation()](https://hivis.readthedocs.io/latest/items.html#HiVis.HiVis.HiVis.agg_from_annotations).

## 2. Implementing pixel classifiers
Follow the [official Qupath pixel classifier tutorial](https://qupath.readthedocs.io/stable/docs/tutorials/pixel_classification.html).

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


<u>Export</u><br>
To export the classifier result as a label image (mask), run the script 
[ExportPixelClassifierAsLabelImage.groovy](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/ExportPixelClassifierAsLabelImage.groovy) 
with downsample=1.

<u>Import into HiVis</u><br>
Classifier can be imported into Python with [HiVis.add_mask()](https://hivis.readthedocs.io/latest/items.html#HiVis.HiVis.HiVis.add_mask).
This assigns annotation for each bin. 

Note; if using a very high resolution classifier (not recommended), apply the classifier with the groovy scripts in the next sections, this will output the exact coverege of classes for each bin, rather than categorical class.

## 3. Segmentation scripts - overview and parameters

1. **Detect the Whole tissue or Anatomical regions** automatically (or manually)
2. **Segment Cells** within each anatomical region 
3. **Load the VisiumHD bins position and associate bins to cells**  
4. **Save results in HiVis compatible formats:**  
   - Export the **cell borders** as '.geoJson' file
   - Export **Anatomical regions** borders as '.geoJson' file
   - Save '.csv' file with both **cells** and **bins** information

  
| Script Filename                                      | Use Case                              |
|------------------------------------------------------|----------------------------------------|
| `VisiumHDAnalysis_Flourescent.groovy`                | Fluorescent images (cell + nuclei)     |
| `VisiumHDAnalysis_StarDist.groovy` | H&E images or DAPI only. Includes option for artifact filtering |
| `VisiumHDAnalysis_DoubleNucleatedCells_v1.groovy`    | Custom script for mouse liver Fluorescent images:<br>segment heptocytes and their nuclei with cellpose,<br>segment non-parenchymal cells with Stardist & expansion     |

### <u>Parameters</u><br>
| Parameter                    | Possible Values | Description                                                                                  |
| ---------------------------- | --------------- | -------------------------------------------------------------------------------------------- |
**General Workflow Steps**
| `segmentTissue`              | `0`, `1`        | If `1`, segment the entire tissue region.                                                    |
| `segmentAnatomicalRegions`   | `0`, `1`        | If `1`, perform segmentation of anatomical subregions.                                       |
| `segmentCells`               | `0`, `1`        | If `1`, segment individual cells.                                                            |
| `loadSpots`                  | `0`, `1`        | If `1`, load Visium HD spot coordinates and metadata.                                        |
| `associateSpotsToCells`      | `0`, `1`        | If `1`, associate spatial transcriptomics spots to segmented cells.                          |
| `runPixelClassifierForSpot`  | `0`, `1`        | Run a pixel classifier for spot-level classification.                                        |
| `runPixelClassifierForCell`  | `0`, `1`        | Run a pixel classifier for cell-level classification. Requires `AddMeasurementsToCells = 1`. |
| `exportCellsAsGeoJson`       | `0`, `1`        | Export segmented cells as GeoJSON for visualization or spatial analysis.                     |
| `exportAnnotationsAsGeoJson` | `0`, `1`        | Export QuPath annotations as GeoJSON.                                                        |
| `saveResultTable`            | `0`, `1`        | If `1`, export results table as tab-separated `.txt` file.                                   |
**Segmentation and Channels**
| `cellSegmentationMethod` | `"cellBorders"`, `"expandNuc"`, `"onlyNuc"`                          | Defines how cell boundaries are determined relative to nuclei. |
| `segmentationAlg`        | `"cellpose"`, `"stardist"`, `"instaseg"`                             | Algorithm used for segmentation.                               |
| `nucChannel`             | `"Channel 4"` (Tonsil), `"Channel 4"` (mouse_liver)                  | Image channel used for nuclei detection.                       |
| `membraneChannels`       | `["Channel 1", "Channel 3"]` (Tonsil), `["Channel 1"]` (mouse_liver) | Channels used to identify cell membranes.                      |
**Cellpose Parameters**
| `useCellposeSAM`       | `0`, `1`                                         | Use SAM (Segment Anything Model) integration in Cellpose. |
| `CellposeNucModel`     | `"nuc"`, `"cpsam"`                               | Model used for nuclei segmentation.                       |
| `CellposeCellModel`    | `"cyto3"`, `"cpsam"`, path to custom `.cpm` file | Model used for cell segmentation.                         |
| `CellposeCellDiameter` | integer (e.g., `15`, `50`)                       | Approximate cell diameter in pixels.                      |
| `CellposeNucDiameter`  | integer (e.g., `15`, `21`)                       | Approximate nucleus diameter in pixels.                   |
**StarDist Parameters**
| Parameter                   | Possible Values                 | Description                                                       |
| `StarDistPathModel`         | path to `.pb` model file        | Path to trained StarDist model.                                   |
| `clear_existing_detections` | `true`, `false`                 | Whether to clear previous detections before segmentation.         |
| `param_threshold`           | `0.0–1.0`                       | Detection probability threshold; higher = more confident cells.   |
| `normalize_low_pct`         | integer (e.g., `0–5`)           | Lower percentile for normalization (set `0` to disable).          |
| `normalize_high_pct`        | integer (e.g., `95–100`)        | Upper percentile for normalization (set `100` to disable).        |
| `param_tilesize`            | multiple of 16 (default `1024`) | Tile size (in pixels) for processing; smaller = lower memory use. |
**InstaSeg Parameters**
| `InstaSegModel`             | path to InstaSeg model directory | Model used for nuclei and cell segmentation.   |
| `InstaSeg_tileDims`         | integer (e.g., `1024`)           | Tile size (in pixels) for processing.          |
| `InstaSeg_interTilePadding` | integer (e.g., `32`)             | Overlap between tiles to avoid edge artifacts. |
| `InstaSeg_nThreads`         | integer                          | Number of processing threads.                  |
| `InstaSeg_device`           | `"gpu"`, `"cpu"`                 | Compute device to use for segmentation.        |
**Classes and Output**
| `cellClassName`    | `"Cell"`                                    | Name of the QuPath class for cells.           |
| `nucClassName`     | `"Nuc"`                                     | Name of the QuPath class for nuclei.          |
| `spotClassName`    | `"Spot"`                                    | Name of the QuPath class for Visium HD spots. |
| `resultsSubFolder` | `"results_tonsil"`, `"results_mouse_liver"` | Subfolder for saving exported results.        |
**Whole Tissue Parameters**
| `wholeTissueClass`        | `"WholeTissue"`, `"Epithelium"`, `"Crypt"`, etc.                | Class label for the full tissue region.                    |
| `WholeTissueClassifier`   | `"WholeTissue_Tonsil_Moderate_v1"`, `"WholeTissue_mouse_liver"` | Pixel classifier name used for whole tissue segmentation.  |
| `WholeTissue_MinSize`     | integer (e.g., `10000`)                                         | Minimum connected component size (in pixels).              |
| `WholeTissue_MinHoleSize` | integer (e.g., `3000–10000`)                                    | Minimum size of holes to retain; smaller holes are filled. |
**Anatomical Region Parameters**
| `ClassNameForAnatomicalRegions`     | `"WholeTissue"`, `"tissue"`                                        | Name of parent class for anatomical regions.            |
| `AnatomicalRegionsPixelClassifier`  | classifier name (e.g., `"epithel_non_epithel_ignore_moderate_v1"`) | Pixel classifier used to define anatomical subregions.  |
| `AnatomicalRegions_MinSize`         | integer (e.g., `500`)                                              | Minimum region size to keep.                            |
| `AnatomicalRegions_MinHoleSize`     | integer (e.g., `40`, `500`)                                        | Minimum hole size to preserve.                          |
| `AnatomicalRegionsClassNames`       | e.g., `["WholeTissue"]`                                            | List of region class names.                             |
| `AnatomicalRegionsExpansionMicrons` | list of integers (e.g., `[5]`)                                     | Expansion distance (in µm) per anatomical region class. |
**Pixel Classifier Parameters**
| `PixelClassifier` | classifier name (e.g., `"epithelial_celiac_classifier"`) | Pixel classifier used for cell classification. |
**Visium HD Spot Parameters**
| `scalefactors_json`     | path to `scalefactors_json.json` file | Defines scaling between image and Visium coordinate system.      |
| `csvfile`               | path to `tissue_positions.csv` file   | Contains spot positions and barcodes.                            |



## 4. Cells segmentation tools and options
**If both cell and nuclei Borders are available**  
Use segmentation methods that accurately detect the cell border, such as:
- [InstaSeg](https://github.com/instanseg/instanseg)
- [Cellpose](https://github.com/BIOP/qupath-extension-cellpose)
 
Both come with built-in models that perform well in many cases. If needed, train your own model.

**If only nuclei borders are available**  
Examples: H&E image or fluorescent DAPI-only staining.
- Cell borders are **approximated** by expanding the detected nuclei.
- Expansion is limited by:
  - A **maximum distance** from the nucleus
  - Until it touches a **neighboring expanding cell**

Nuclei segmentation options:
- [StarDist](https://github.com/qupath/qupath-extension-stardist))
- [Cellpose](https://github.com/BIOP/qupath-extension-cellpose)
- [InstaSeg](https://github.com/instanseg/instanseg)

<br>

**Advanced options supported by the baseline workflow**:
- **Restrict expansion within anatomical regions**
  - Requires a trained **pixel classifier** or **manual annotation**
- **Region-specific expansion size**
- **Region-specific segmentation parameters** (e.g., detection probability threshold)
- **Nuclei classification–based expansion**
  - Use a trained *object classifier* to apply different parameters to different cell types
- **Artifact filtering**
  - Train an object classifier to distinguish *nuclei vs artifacts* (helpful for H&E)
  

### Train your own Cellpose model 
Train your own cellpose model within QuPath: 
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



## 5. Segmentation scripts workflow

### Associating Bins to Cells  
Association of Visium-HD bins to segmented cells is implemented **within QuPath**.<br>
Bins are uploaded from the 'tissue_positions.csv' file 


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
  
### Optional: Run Pixel Classifier Per Bin  
- Each spot receives measurements of the *percentage of its area* covered by each class
- Uses **hard decision classification**





