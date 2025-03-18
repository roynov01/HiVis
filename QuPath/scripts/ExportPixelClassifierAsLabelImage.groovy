// This scripts outputs a label image (mask) of a pixel classifier, as tif file.
//Code originally from https://forum.image.sc/t/qupath-script-with-pixel-classifier/45597/10

def pixelClassifier = "my_classifier"  // name of the pixel classifier
def outSubFolder = "image_export"  // subfolder where the image will be saved
def downsample = 1  // can be changed for snapshots, but the HiVis can use only non downsampled image.

def imageData = getCurrentImageData()
def classifier = loadPixelClassifier(pixelClassifier)
name = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())
mkdirs(buildFilePath(PROJECT_BASE_DIR, outSubFolder))
def predictionServer = PixelClassifierTools.createPixelClassificationServer(imageData, classifier)
def path = buildFilePath(PROJECT_BASE_DIR, outSubFolder, name+'_prediction.tif')
writeImageRegion(predictionServer, RegionRequest.createInstance(predictionServer, downsample), path)
