// VisiumHDAnalysis_StarDist_AnatomicalRegions_FilterNuc.groovy - Quantify VisiumHD H&E Slide 
// 
// By: Ofra Golani, with Roy Novoselsky, Shalev Itzkovitz
//
// Workflow
// ============================================================
// 
// - Segment Tissue region using pixel classifier
// - Segment anatomical regions within the WholeTissue using pixel classifier
// - Sement Cells within each anatomical regions separately, using StarDist with per-class Expansion parameters 
//   Optionally filter Nuc before expansion based on object classifier 
//      in this case, the script make sure to mask out expanded regions that are outside of the parent annotation
// - Load VisiumHD spots as detections
// - Associate spots to cells - encode the parent cell within the spot name as barcode_cellID
//   For each spot set inCell and inNuc flags
//   For each cell count the number of inCell and inNuc spots
// - Run pixel classifier on the spots 
// - Export cell boundaries as GeoJson file 
// - Export anatomical regions boundaries as GeoJson file 
// - Save detection and annotation measurements into csv files
//   The detection table contain both cells and spots information


// ===================  Workflow control Parameters  ====================================

def segmentTissue                 = 0 // 
def segmentAnatomicalRegions      = 0
def segmentCells                  = 1 // 
def filterNucBeforeCellExpansion  = 1
def AddMeasurementsToCells        = 0
def loadSpots                     = 0
def associateSpotsToCells         = 0
def runPixelClassifierForSpot     = 0
def runPixelClassifierForCell     = 0 // MAKE SURE THAT AddMeasurementsToCells IS ALSO 1
def exportCellsAsGeoJson          = 0
def exportAnnotationsAsGeoJson    = 0
def saveResultTable               = 0 // export result table to Tab-separated txt file

var cellClassName = "Cell"
var spotClassName = "Spot"
def cellClass = getPathClass(cellClassName)

def resultsSubFolder = 'results_qp6' // subfolder for table 

def wholeTissueClass = "WholeTissue" //"Epithelium" //"non_epithelium" // "Crypt" //"WholeTissue" 

def WholeTissueClassifier = "WholeTissue_High_v1"    // name of WholeTissue pixel classifier
def WholeTissue_MinSize =  10000          // Minimal WholeTissue connected-component size        
def WholeTissue_MinHoleSize = 3000        // minmal hole size to keep when creating WholeTissue regions, samller holes are filled 


//setImageType('BRIGHTFIELD_H_E');

// ===================  Parameters for Creation of TissueWithoutGoblet annotations ======================================================================
// ===================  Parameters for Creation of Anatomical Regions annotations using the TissueWithoutGoblet annotations =============================
def ClassNameForAnatomicalRegions = "WholeTissue" //"tissue" //"WholeTissue"
def AnatomicalRegionsPixelClassifier = "epithel_non_epithel_ignore_moderate_v1" //"epithelial_celiac_classifier" //"WholeTissue_High_v1" //"epithelial_celiac_classifier"
def AnatomicalRegions_MinSize = 500 //500
def AnatomicalRegions_MinHoleSize = 40 //500 //80

// ===================  Cell Segmentation parameters  - Segmentation is based on StarDist + expansion ====================================
AnatomicalRegionsClassNames = ["epithel", "non_epithel"]
AnatomicalRegionsExpansionMicrons  = [7, 2] //[7, 2]
//AnatomicalRegionsClassNames = ["WholeTissue"]
//AnatomicalRegionsExpansionMicrons  = [2]
//AnatomicalRegionsExpansionMicrons  = [7]
// NOTE - see below for AnatomicalRegionsStarDist

var StarDistPathModel = 'A:/shared/QuPathScriptsAndProtocols/QuPath_StarDistModels/he_heavy_augment.pb'
// Stardist parameters 
var clear_existing_detections = false
var param_threshold    = 0.1 //0.1 //0.5 //threshold for detection. All cells segmented by StarDist will have a detection probability associated with it, where higher values indicate more certain detections. Floating point, range is 0 to 1. Default 0.5
var normalize_low_pct  = 1   //lower limit for normalization. Set to 0 to disable
var normalize_high_pct = 99  // upper limit for normalization. Set to 100 to disable.
//var param_expansion    = 20 //3 //20 //5   //size of cell expansion in pixels. Default is 10.
var param_tilesize     = 1024 //size of tile in pixels for processing. Must be a multiple of 16. Lower values may solve any memory-related errors, but can take longer to process. Default is 1024.

// Get Pixel size from the image
def cal = getCurrentServer().getPixelCalibration()
def pixelWidth = cal.pixelWidth
def pixelHeight = cal.pixelHeight
def AnatomicalRegionsExpansionPixels = AnatomicalRegionsExpansionMicrons.collect { it / pixelWidth }


//print("============ pixelWidth="+ pixelWidth+ ", pixelHeight="+pixelHeight+" =====================")

// ===================  Nuc Classifier Parameters   ====================================
def PositiveNegativeNucClassifier = "PositiveNegative_Nuc_v3"


 // ===================  Pixel Classifier Parameters   ====================================
def PixelClassifier = "epithelial_celiac_classifier"

// ===================  Visium HD Spots parameters  ====================================
//double spot_diameter_fullres = 7.30563538369773 //8.048464667916532; //29.22254153479092; // take this value from the file scalefactors_json.json
def scalefactors_json = 'A:/royno/HiVis_proj_v2/datasets/mouse_intestine_scalefactors_json.json' // TO CHANGE
def csvfile = 'A:/royno/HiVis_proj_v2/datasets/mouse_intestine_tissue_positions.csv' // TO CHANGE

// =======================================================================================================
// ===================  Code Begins - Dont Change from here downward  ====================================

// =======================================================================================================
// ===================  Segment Whole Tissue =============================================================
var imageData = getCurrentImageData()
if (segmentTissue) {
    //createAnnotationsFromPixelClassifier(WholeTissueClassifier, WholeTissue_MinSize, WholeTissue_MinHoleSize, "SELECT_NEW")
    createAnnotationsFromPixelClassifier(WholeTissueClassifier, WholeTissue_MinSize, WholeTissue_MinHoleSize)
    
    println '======================== Whole Tissue segmentation Done =================== '
}

// =======================================================================================================
// ===================  Segment Anatomical regions  ======================================================
if (segmentAnatomicalRegions)
{
    selectObjectsByClassification(ClassNameForAnatomicalRegions);  
    createAnnotationsFromPixelClassifier(AnatomicalRegionsPixelClassifier, AnatomicalRegions_MinSize, AnatomicalRegions_MinHoleSize)
    
    println '======================== Anatomical regions segmentation Done =================== '
}

// =======================================================================================================
// ===================  Segment Cells within each anatomical region independently  =======================
if (segmentCells) {
    
    AnatomicalRegionsClassNames.indices.each { k ->
        regionClassName = AnatomicalRegionsClassNames[k]
        println '======================== Running Cell segmentation on '+regionClassName+' Expansion='+AnatomicalRegionsExpansionPixels[k]+'  ... =========================== '
        resetSelection()
        pathObjects =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass(regionClassName) }
        if (pathObjects.isEmpty()) {
            Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
            return
        }

        var stardistNuc = StarDist2D.builder(StarDistPathModel)
              .threshold(param_threshold)              // Prediction threshold
              .normalizePercentiles(normalize_low_pct,normalize_high_pct) // Percentile normalization
              .pixelSize(pixelWidth)              // Resolution for detection
              //.includeProbability(true)
              .measureIntensity()
              //.tileSize(param_tilesize)
              .measureShape()
              //.cellExpansion(param_expansion) //Cell expansion in microns
              .build()
        
        var stardist_cells = StarDist2D.builder(StarDistPathModel)
              .threshold(param_threshold)              // Prediction threshold
              .normalizePercentiles(normalize_low_pct,normalize_high_pct) // Percentile normalization
              .pixelSize(pixelWidth)              // Resolution for detection
              //.includeProbability(true)
              //.measureIntensity()
              //.tileSize(param_tilesize)
              .measureShape()
              //.cellExpansion(param_expansion) //Cell expansion in microns
              .cellExpansion(AnatomicalRegionsExpansionMicrons[k]) //Cell expansion in microns
              .build()

        if (filterNucBeforeCellExpansion) {
            // workaround ..
            existingCells = getCellObjects()
            selectObjects(existingCells)
            clearSelectedObjects(true);
            // end of workaround ..
            
            stardistNuc.detectObjects(imageData, pathObjects)
            runObjectClassifier(PositiveNegativeNucClassifier); 
            selectObjectsByClassification("negative");
            var negative = getSelectedObjects()
            println 'In '+regionClassName+' #Negative='+negative.size()            
            clearSelectedObjects(true);
            
            selectObjectsByClassification("positive");
            def nuc = getSelectedObjects()
            println 'In '+regionClassName+' #Positive='+nuc.size()            
            def cellObjects = CellTools.detectionsToCells(nuc, AnatomicalRegionsExpansionPixels[k], -1) //Cell expansion in pixels
            removeObjects(nuc,true)
            addObjects(cellObjects)
            selectObjects(cellObjects)
            //for (cell in cellObjects) {cell.setPathClass(cellClass) }            
            for (cell in cellObjects) {cell.setPathClass(getPathClass(regionClassName)) }            
// *    

            detectionToAnnotationDistancesSigned(false)
            // constrain the cells by the parent Annotation
            var hierarchy = getCurrentHierarchy()
            
            double maxDistanceToCheck = - AnatomicalRegionsExpansionMicrons[k]*3 
            println 'Checking all cells closer to annotation border than '+ maxDistanceToCheck
            // Loop through each annotation
            for (annotation in pathObjects) {
                // Get the ROI of the parent annotation
                var roi = annotation.getROI()  
                // Get all cells within this annotation, but to make it run faster, check for intersection only cells close to the border
                var measurementsName = "Signed distance to annotation "+regionClassName+" µm"
                //var cells = hierarchy.getObjectsForROI(PathDetectionObject.class, roi).findAll{it.getMeasurementList().getMeasurementValue(measurementsName) > maxDistanceToCheck}
                //var cells = hierarchy.getObjectsForROI(PathDetectionObject.class, roi).findAll{it.getMeasurementList().get(measurementsName) > maxDistanceToCheck}
                var cells = hierarchy.getObjectsForROI(PathDetectionObject.class, roi)
                // Clip each cell to the parent ROI
                for (cell in cells) {
                    println 'clipping cell #'+ cell.getID().toString() + ', Distance='+cell.getMeasurementList().get(measurementsName)
                    //cell.setROI(cell.getROI().clipToROI(roi))
                    var roiCell = cell.getROI()
                    roiCell = RoiTools.combineROIs(roi, roiCell, CombineOp.INTERSECT);
                    cell.setROI(roiCell)        
                }
            }       
//  */               
            if (!existingCells.isEmpty()) 
                addObjects(existingCells)
        } // filterNucBeforeCellExpansion
        else {
             stardist_cells.detectObjects(imageData, pathObjects)
             //for (cell in getCellObjects()) {cell.setPathClass(cellClass) }                     
             //for (cell in getCellObjects()) {cell.setPathClass(getPathClass(regionClassName)) }                     
        }
        
        // OG  keep region class ?? 
        //for (cell in getCellObjects()) {cell.setPathClass(cellClass) }
        
        println '======================== Cell segmentation on '+AnatomicalRegionsClassNames[k]+' Done =========================== '
    }
}

resolveHierarchy()

// =======================================================================================================
// ===================  Load Visium HD Spots and associate them to Cells =================================
if (loadSpots) {
    // Extract scale factors
    // Read the file content
    def jsonText = new String(Files.readAllBytes(Paths.get(scalefactors_json)))
    
    // Use regex to extract the value of "spot_diameter_fullres"
    def pattern = ~/\"spot_diameter_fullres\"\s*:\s*([0-9.]+)/
    def matcher = pattern.matcher(jsonText)
    
    double spot_diameter_fullres 
    if (matcher.find()) {
        //def spot_diameter_fullres = matcher.group(1).toDouble()
        spot_diameter_fullres = matcher.group(1).toDouble()
        println "Spot diameter full resolution: ${spot_diameter_fullres}"
    } else {
        println "Could not find 'spot_diameter_fullres' in the JSON"
    }
        
    println '======================== Importing Spots ... ==================================='
    // Create BufferedReader
    def csvReader = new BufferedReader(new FileReader(csvfile));
    def plane = ImagePlane.getDefaultPlane();

    listOfObjects = [];    
    int first_row = 1;
    // Loop through all the rows of the CSV file.
    while ((row = csvReader.readLine()) != null) {
    
        if (first_row)
        {
            first_row = 0;
        }
        else 
        {
            def rowContent = row.split(",")
            String barcode = rowContent[0] as String;
            int  in_tissue = rowContent[1] as int;
            int  array_row = rowContent[2] as int;
            int  array_col = rowContent[3] as int;
            double cx = rowContent[4] as double;
            double cy = rowContent[5] as double;
            /*String barcode = rowContent[1] as String;
            int  in_tissue = rowContent[2] as int;
            int  array_row = rowContent[3] as int;
            int  array_col = rowContent[4] as int;
            double cx = rowContent[5] as double;
            double cy = rowContent[6] as double;*/
            
            int first_time = 1;
            // Create annotation
            if (in_tissue) {
                double px = cx - spot_diameter_fullres/2;
                double py = cy - spot_diameter_fullres/2;
                def roi = new RectangleROI(py, px, spot_diameter_fullres, spot_diameter_fullres, plane)
                
                // Spots are imported as detections
                //def detection = new PathDetectionObject(roi, PathClass.fromString("Spot"));                
                def detection = new PathTileObject(roi, PathClass.fromString("Spot"),null);                
                if (first_time) {
                    first_time = 0;
                }
                //detection.getMeasurementList().putMeasurement("array_row", array_row);
                detection.getMeasurementList().put("array_row", array_row);
                detection.getMeasurementList().put("array_col", array_col);
                detection.getMeasurementList().put("cx", cx);
                detection.getMeasurementList().put("cy", cy);
                detection.getMeasurementList().put("InCell", 0);
                detection.getMeasurementList().put("InNuc", 0);
                //detection.setName(barcode);
                detection.setName(barcode.replace('"',''));
                listOfObjects << detection;        
            }
        }
    }
    //imageData.getHierarchy().addObjects(listOfAnnotation, true);
    addObjects(listOfObjects);
    println '======================== Spots imported ==================================='
}

if (associateSpotsToCells) {
    println '======================== Associating Spots To Cells ... ==================================='
    def cellObj = getCellObjects() 
    for (def cell in cellObj) {
        def subCellObj = getCurrentHierarchy().getObjectsForROI(null, cell.getROI()) // get all objects within each cellObj RO        מSפםא = 0
        nSpots = 0
        for (s in subCellObj) {
            if (s.isTile()) {
                nSpots++
                parentName = cell.getID().toString()
                sName = s.getName()
                //newName = sName + "__" + parentName;
                //newName = String.format("%s__%s", sName, parentName)
                newName = String.join("",sName, "__", parentName)
                s.setName(newName)
                s.measurements.put("InCell", 1)
                s.measurements.put("InNuc", 0)
            }        
        }
        nNucSpots = 0
        if (cell.getNucleusROI()) {
            def subNucObj = getCurrentHierarchy().getObjectsForROI(null, cell.getNucleusROI()) // get all objects within each cellObj Nucleus ROI
            for (s in subNucObj) {
                if (s.isTile()) {
                    nNucSpots++
                    s.measurements.put("InNuc", 1)
                }
            }
        }
        //cell.measurements.put("nSpots",subCellObj.size())
        //cell.measurements.put("nNucSpots",subNucObj.size())
        cell.measurements.put("nSpots",nSpots)
        cell.measurements.put("nNucSpots",nNucSpots)
    }
        
    println '======================== Associate Spots To Cells Done ==================== '
}


// ===================  Add pixel classifier measurements to spots ====================================
if (runPixelClassifierForSpot) {
    selectTiles()
    addPixelClassifierMeasurements(PixelClassifier, PixelClassifier)
    resetSelection()
}

// ===================  Add pixel classifier measurements to cells ====================================
if (runPixelClassifierForCell) {
    selectObjectsByClassification("Cell");
    addPixelClassifierMeasurements(PixelClassifier, PixelClassifier)
    resetSelection()
}

// ===================  Add Measurements to Cells  ====================================
if (AddMeasurementsToCells) {
    selectObjectsByClassification("Cell");
    runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons":2.0,"region":"ROI","tileSizeMicrons":25.0,"colorOD":false,"colorStain1":true,"colorStain2":true,"colorStain3":false,"colorRed":false,"colorGreen":false,"colorBlue":false,"colorHue":false,"colorSaturation":false,"colorBrightness":false,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":false,"haralickDistance":1,"haralickBins":32}')
    addShapeMeasurements("AREA", "LENGTH", "CIRCULARITY", "SOLIDITY", "MAX_DIAMETER", "MIN_DIAMETER", "NUCLEUS_CELL_RATIO")
    selectAnnotations();
    runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin', '{"fwhmMicrons":25.0,"smoothWithinClasses":false}')
}

// ===================  Export Cells As GeoJson ====================================
if (exportCellsAsGeoJson) {    
    println '======================== save Cells as GeoJson ... ==================================='
    
    File directory = new File(buildFilePath(PROJECT_BASE_DIR,resultsSubFolder));
    directory.mkdirs();
    imageName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())

    cells = getCellObjects()
    //cells = getDetectionObjects().findAll{ !it.isTile() };
    selectObjects(cells)  
    //exportSelectedObjectsToGeoJson("A:\\royno\\Visium_HD_liver\\experiment1\\qupath_project\\mouse_liver_98_WT_fullres.tif - 2d copy for alternative analysis (1).geojson", "PRETTY_JSON", "FEATURE_COLLECTION")
    exportSelectedObjectsToGeoJson(buildFilePath(directory.toString(),imageName+'_cells.geojson'), "EXCLUDE_MEASUREMENTS", "PRETTY_JSON", "FEATURE_COLLECTION")        
    println '======================== save Cells as GeoJson Done ==================================='
}

// ===================  Export Annotations (usually anatomical Regions) As GeoJson ====================================
if (exportAnnotationsAsGeoJson) {    
    println '======================== save Annotations as GeoJson ... ==================================='
    
    File directory = new File(buildFilePath(PROJECT_BASE_DIR,resultsSubFolder));
    directory.mkdirs();
    imageName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())

    anns = getAnnotationObjects()
    selectObjects(anns)  
    exportSelectedObjectsToGeoJson(buildFilePath(directory.toString(),imageName+'_annotations.geojson'), "EXCLUDE_MEASUREMENTS", "PRETTY_JSON", "FEATURE_COLLECTION")        
    println '======================== save Annotations as GeoJson Done ==================================='
}

// ===================  Save Results ====================================
if (saveResultTable)
{        
    println '====================== Save Results Table ... ======================='
    File directory = new File(buildFilePath(PROJECT_BASE_DIR,resultsSubFolder));
    directory.mkdirs();
    imageName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())
    saveAnnotationMeasurements(buildFilePath(directory.toString(),imageName+'_annotations.csv'));
    saveDetectionMeasurements(buildFilePath(directory.toString(),imageName+'_detections.csv'));
    println '====================== Save Results Table Done ======================='
}

println '======================== Workflow Done! ==================================='


// ===================  Library import  ====================================

import qupath.ext.stardist.StarDist2D
import qupath.ext.biop.cellpose.Cellpose2D
import qupath.lib.images.servers.TransformedServerBuilder
import qupath.lib.images.ImageData
import groovy.time.*
import java.io.BufferedReader;
import java.io.FileReader;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathTileObject;
//import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.roi.ROIs
import qupath.lib.roi.RectangleROI
import qupath.lib.roi.RoiTools.CombineOp
import java.nio.file.Files
import java.nio.file.Paths
import java.util.regex.Matcher
import java.util.regex.Pattern
