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
Alternetaviely, call HiVis.export_images() - and use the exported fullres_image.tif


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



### Import into HiVis
Cells can be imported into Python with [HiVis.agg_cells()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.agg_cells).

## 4. Cell segmentation - based on multiple channels (fluorescence, using Cellpose or InstanSeg)  



### Import into HiVis
Cells can be imported into Python with [HiVis.agg_cells()](https://hivis.readthedocs.io/en/latest/items.html#HiVis.HiVis.HiVis.agg_cells).


## 5. Technical details of QuPath Implementation
QuPath has two object types:
- **Annotations**
  - Flexible and can contain child objects
- **Detections**
  - More efficient and suited for large datasets
  - Cannot have children
  - `Cells` and `Tiles` are both subtypes of `Detections`

For millions of bins and hundreds of thousands of cells:
- Bins should be detections of type: `Tile`
- Cells should be detections (not annotations)

As a result, **QuPath's built-in parent-child hierarchy cannot be used**.
Instead, **parent cell ID is encoded in the bin’s** `Name` **field** as:
> BinBarcode__ParentCellObjectID

## Referances



