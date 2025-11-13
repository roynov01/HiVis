# Guide for working with Qupath for HiVis
For general Qupath tutorial see the [documentation](https://qupath.readthedocs.io/en/stable/index.html).

This tutorial includes:
1. Manual annotations
2. Implementing pixel classifier
3. Cell segmentation - based on nuclei only (H&E or fluorescence, using Stardist)
4. Cell segmentation - based on multiple channels (fluorescence, using Cellpose or InstanSeg)  
5. Technical details of QuPath Implementation

**Important!**

The image that should be used in Qupath is the cropped image that is created when creating a new HiVis object in Python.

Alternetaviely, and highly advisible - call [HiVis.export_images()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.export_images) - and use the exported fullres_image.tif


## 1. Manual annotations
In the annotation tab, select the class you want to annotate and click "set selected".
Then annotate with any of the [annotation tools](https://qupath.readthedocs.io/en/stable/docs/starting/annotating.html#annotation-tools).

### Export
To export the annotations, select the annotations you want to export (ctrl+A for all), and click on File => Export objects as GeoJSON, leave the default options. 

### Import into HiVis

Annotations can be imported into Python with [HiVis.add_annotations()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.add_annotations).
This assigns annotation for each bin.

You can create an "Aggregation" object from the annotations by calling [HiVis.agg_from_annotation()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.agg_from_annotations).

## 2. Pixel classifiers
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


## 3. Cell segmentation - based on nuclei only (H&E or fluorescence, using Stardist)
Download and install 
[Stardist extension for QuPath](https://github.com/qupath/qupath-extension-stardist).

Use the [StarDist script](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/VisiumHDAnalysis_StarDist_AnatomicalRegions_FilterNuc.groovy).

Import the segmentation into Python and aggregate gene expression with [HiVis.agg_cells()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.agg_cells).

### Control parameters (0 or 1):
* **segmentTissue** - Segment the tissue, and not allow the detection of nuclei outside it. Also limits the expansion of cells to the tissue area.
* **segmentAnatomicalRegions** - Segment anatomical regions to allow varying expansion  of nuclei in each region.
* **segmentCells** - Segment cells (1) or only nuclei (0).
* **filterNucBeforeCellExpansion** - Filter artefact nuclei prior to expansion.
* **AddMeasurementsToCells** - Add measurements and channels intensities to cells.
* **loadSpots** - Load VisiumHD bins.
* **associateSpotsToCells** - Associate spots to cells.
* **runPixelClassifierForSpot** - Give each bin an identity based on the pixel classifier.
* **runPixelClassifierForCell** - Give each cell an identity based on the pixel classifier.
* **exportCellsAsGeoJson** - Export geometry of cells.
* **exportAnnotationsAsGeoJson** - 
# <span style="color:red">**ADD**</span>
* **saveResultTable** - Save the results as a CSV file.

### General parameters:
* **scalefactors_json** - Path of scalefactors_json.json file containing scale factors positions. exported by [HiVis.export_images()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.export_images)
* **csvfile** - Path of tissue_positions.csv file containing bins positions. exported by [HiVis.export_images()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.export_images)
* **WholeTissueClassifier** - Name of WholeTissue pixel classifier
* **wholeTissueClass** - Name of whole-tissue class in the WholeTissue pixel classifier
* **WholeTissue_MinSize** - Minimal WholeTissue connected-component size 
* **WholeTissue_MinHoleSize** - Minmal hole size to keep when creating WholeTissue regions, samller holes are filled 
* **PixelClassifier** - Name of pixel classifier, if runPixelClassifier were set to 1


### Stardist parameters:
* **StarDistPathModel** - Path for Stardist model, such as he_heavy_augment.pb.
* **param_threshold** - Threshold for detection. All cells segmented by StarDist will have a detection probability associated with it, where higher values indicate more certain detections. Floating point, range is 0 to 1. Default 0.5
* **normalize_low_pct** - Lower limit for normalization. Set to 0 to disable.
* **normalize_high_pct** - Upper limit for normalization. Set to 100 to disable.
* **param_tilesize** - Size of tile in pixels for processing. Must be a multiple of 16. Lower values may solve any memory-related errors, but can take longer to process. Default is 1024.
* **PositiveNegativeNucClassifier** - OPTIONAL name of object classifier to filter nuclei prior to expansion.

Expension parameters

* **AnatomicalRegionsClassNames** - List of names of identities.
* **AnatomicalRegionsExpansionMicrons** - List of expansion amount for each of the identities.


## 4. Cell segmentation - based on multiple channels (fluorescence, using Cellpose or InstanSeg)  
Download and install 
[Cellpose extension for QuPath](https://github.com/BIOP/qupath-extension-cellpose).

Use the [fluorescence script](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/VisiumHDAnalysis_Flourescent.groovy).

Import the segmentation into Python and aggregate gene expression with [HiVis.agg_cells()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.agg_cells).

### Control parameters (0 or 1):

* **segmentTissue** - Segment the tissue, and not allow the detection of nuclei outside it. Also limits the expansion of cells to the tissue area.
* **segmentAnatomicalRegions** - Segment anatomical regions to allow varying expansion  of nuclei in each region.
* **segmentCells** - Segment cells (1) or only nuclei (0).
* **filterNucBeforeCellExpansion** - Filter artefact nuclei prior to expansion.
* **AddMeasurementsToCells** - Add measurements and channels intensities to cells.
* **loadSpots** - Load VisiumHD bins.
* **associateSpotsToCells** - Associate spots to cells.
* **runPixelClassifierForSpot** - Give each bin an identity based on the pixel classifier.
* **runPixelClassifierForCell** - Give each cell an identity based on the pixel classifier.
* **exportCellsAsGeoJson** - Export geometry of cells.
* **exportAnnotationsAsGeoJson** - 
# <span style="color:red">**ADD**</span>
* **saveResultTable** - Save the results as a CSV file.

### General parameters:
* **scalefactors_json** - Path of scalefactors_json.json file containing scale factors positions. exported by [HiVis.export_images()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.export_images)
* **csvfile** - Path of tissue_positions.csv file containing bins positions. exported by [HiVis.export_images()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.export_images)
* **WholeTissueClassifier** - Name of WholeTissue pixel classifier
* **wholeTissueClass** - Name of whole-tissue class in the WholeTissue pixel classifier
* **WholeTissue_MinSize** - Minimal WholeTissue connected-component size 
* **WholeTissue_MinHoleSize** - Minmal hole size to keep when creating WholeTissue regions, samller holes are filled 
* **PixelClassifier** - Name of pixel classifier, if runPixelClassifier were set to 1

### Stardist parameters:
* **StarDistPathModel** - Path for Stardist model, such as he_heavy_augment.pb.
* **param_threshold** - Threshold for detection. All cells segmented by StarDist will have a detection probability associated with it, where higher values indicate more certain detections. Floating point, range is 0 to 1. Default 0.5
* **normalize_low_pct** - Lower limit for normalization. Set to 0 to disable.
* **normalize_high_pct** - Upper limit for normalization. Set to 100 to disable.
* **param_tilesize** - Size of tile in pixels for processing. Must be a multiple of 16. Lower values may solve any memory-related errors, but can take longer to process. Default is 1024.
* **PositiveNegativeNucClassifier** - OPTIONAL name of object classifier to filter nuclei prior to expansion.

Expension parameters

* **AnatomicalRegionsClassNames** - List of names of identities.
* **AnatomicalRegionsExpansionMicrons** - List of expansion amount for each of the identities.


### Cellpose parameters:
* **CellposeNucModel** - Path for model file, or name of buyilt-in model
* **CellposeCellModel** - Path for model file, or name of buyilt-in model
* **CellposeCellDiameter** - Diameter of cells (µm)
* **CellposeNucDiameter** - Diameter of nuclei (µm)


### InstaSeg parameters:
* **InstaSegModel** - Path for model file
* **InstaSeg_tileDims** - 
* **InstaSeg_interTilePadding** - 

# <span style="color:red">**ADD**</span>

* **InstaSeg_nThread** - Number of threads
* **InstaSeg_device** - "gpu" or "cpu"


### Train a custom Cellpose model

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

## 5. Technical details of QuPath Implementation
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
