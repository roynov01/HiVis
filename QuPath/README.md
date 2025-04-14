# Guide for working with Qupath for HiVis

For general Qupath tutorial see the [documentation](https://qupath.readthedocs.io/en/stable/index.html).

## Manual annoatations
In the annotation tab, select the class you want to annotate and click "set selected".
Then annotate with any of the [annotation tools](https://qupath.readthedocs.io/en/stable/docs/starting/annotating.html#annotation-tools).
To export the annotations, select the annotations you want to export (ctrl+A for all), and click on File => Export objects as GeoJSON, leave the default options. 

## Pixel classification
Follow the [official Qupath pixel classifier tutorial](https://qupath.readthedocs.io/en/stable/docs/tutorials/pixel_classification.html).

When selecting parametes, we suggest to choose:
* Type: Random Trees
* Resolution: moderate (but can vary)
* Features: all available features 
* Scales: [1,2,4,8] (but can vary)
* Channels:
	- For H&E: Hematoxylin, Eosin, Residual 
	- For fluorescence: choose only the fluorescence channels
* No normalization
* Output: Classification

### Export
To export the classifier as a label image (mask), run the script 
["ExportPixelClassifierAsLabelImage.groovy"](https://github.com/roynov01/HiVis/blob/main/QuPath/scripts/ExportPixelClassifierAsLabelImage.groovy)
 with downsample=1.


## Stardist
Download and install 
[Stardist extension for QuPath](https://github.com/qupath/qupath-extension-stardist).

Run the script...


## Cellpose
Download and install 
[Cellpose extension for QuPath](https://github.com/BIOP/qupath-extension-cellpose).

Run the script...
