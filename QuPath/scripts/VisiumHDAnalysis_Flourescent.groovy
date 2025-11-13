// VisiumHDAnalysis_StarDist_AnatomicalRegions_FilterNuc.groovy - Quantify VisiumHD Flourescent Slide 
// 
// By: Ofra Golani, with Roy Novoselsky, Shalev Itzkovitz
//
// Workflow
// ============================================================
// 
// - Segment Tissue region using pixel classifier
// - (Segment anatomical regions within the WholeTissue using pixel classifier)
// - Segment Cells within each anatomical regions separately, using either:
//        Exact Cell Segmentation using nuclei and membrane information. Two methods are supported
//         - cell segmentation with Cellpose using 2 inputs: 
//           * average of available membrane channels
//           * nuclei channel 
//         - cell segmentation with InstaSeg using the available membrane channels + the nuclei channel 
//
//         OR
//
//        Nuclei segmentation with per-class Expansion parameters with either 
//            - StarDist   OR 
//            - Cellpose 
//
// - Load VisiumHD spots as detections
// - Associate spots to cells - encode the parent cell within the spot name as barcode_cellID
//   For each spot set inCell and inNuc flags
//   For each cell count the number of inCell and inNuc spots
// - (Run pixel classifier on the spots )
// - Export cell boundaries as GeoJson file 
// - Export anatomical regions boundaries as GeoJson file 
// - Save detection and annotation measurements into csv files 
//   The detection table contain both cells and spots information
//
// QuPath 0.6.0 compatible script
//
// sample data: tonsil_fullres.tif 
//
// ===================  Workflow control Parameters  =========================================================

def segmentTissue                 = 1 // 
def segmentAnatomicalRegions      = 1
def segmentCells                  = 1 // 
def loadSpots                     = 1
def associateSpotsToCells         = 1
def runPixelClassifierForSpot     = 0
def runPixelClassifierForCell     = 0 
def exportCellsAsGeoJson          = 1
def exportAnnotationsAsGeoJson    = 1
def saveResultTable               = 1 // export result table to Tab-separated txt file

// ===================  File paths  ==========================================================================
def resultsSubFolder = 'results_tonsil' // subfolder for table 

def scalefactors_json = 'A:/royno/HiVis_proj_v2/datasets/tonsil_scalefactors_json.json' // Use Full path
def csvfile = 'A:/royno/HiVis_proj_v2/datasets/tonsil_tissue_positions.csv'             // Use Full path

// ===================  Segmentation parameters  =============================================================
var cellClassName = "Cell"
var nucClassName = "Nuc"
var spotClassName = "Spot"

def cellSegmentationMethod     = "cellBorders" // "cellBorders" // "expandNuc" "onlyNuc"
def segmentationAlg            = "cellpose" //"cellpose" // "stardist" "instaseg"
def nucChannel                 = "Channel 4"                 // Tonsil
def membraneChannels           = ["Channel 1", "Channel 3"]  // Tonsil

// Cellpose parameters 
var useCellposeSAM         = 1 
var CellposeNucModel       = "nuc" 
var CellposeCellModel      = "cyto3" 
//var CellposeCellModel    = 'A:/royno/Visium_HD_liver/experiment1/qupath_project/models/Custom_model_2025-01-08_17_02.cpm'
var CellposeCellDiameter   = 15 // Tonsil
var CellposeNucDiameter    = 15 // Tonsil

// Stardist parameters 
//var StarDistPathModel = 'A:/shared/QuPathScriptsAndProtocols/QuPath_StarDistModels/dsb2018_heavy_augment.pb'
var StarDistPathModel = 'A:/royno/HiVis_proj_v2/Qupath6/Models/StarDistModels/dsb2018_heavy_augment.pb' // use Full path
var param_threshold    = 0.5 // threshold for detection. All cells segmented by StarDist will have a detection probability associated with it, where higher values indicate more certain detections. Floating point, range is 0 to 1. Default 0.5
var normalize_low_pct  = 1   //lower limit for normalization. Set to 0 to disable
var normalize_high_pct = 99  // upper limit for normalization. Set to 100 to disable.
var param_tilesize     = 1024 //size of tile in pixels for processing. Must be a multiple of 16. Lower values may solve any memory-related errors, but can take longer to process. Default is 1024.

// InstaSeg parameters
//var InstaSegModel           = "A:/ofrag/QuPath-InstaSeg-Models/downloaded/fluorescence_nuclei_and_cells-0.1.0"
var InstaSegModel             = "A:/royno/HiVis_proj_v2/Qupath6/Models/InstaSegModels/fluorescence_nuclei_and_cells-0.1.0" // use full path 
def InstaSeg_tileDims         = 1024
def InstaSeg_interTilePadding = 32
def InstaSeg_nThreads         = 4
def InstaSeg_device           = "gpu" 

// ===================  Pixel classifier and anatomical region expansion  ====================================
def PixelClassifier         = "epithel_region_moderate" // "epithelial_celiac_classifier"

// Whole tissue classifier
def wholeTissueClass        = "WholeTissue" 
def WholeTissueClassifier   = "WholeTissue_Tonsil_Moderate_v1"    // name of WholeTissue pixel classifier
def WholeTissue_MinSize     = 10000         // Minimal WholeTissue connected-component size        
def WholeTissue_MinHoleSize = 10000         // Minmal hole size to keep when creating WholeTissue regions, samller holes are filled 

// Anatomical region classifier
def ClassNameForAnatomicalRegions    = "WholeTissue" // Annotations which will be further divided into anatomical regions
def AnatomicalRegionsPixelClassifier = "epithel_non_epithel_ignore_moderate_v1" // "epithel_non_epithel_ignore_moderate_v1" //"epithelial_celiac_classifier" //"WholeTissue_High_v1" //"epithelial_celiac_classifier"
def AnatomicalRegions_MinSize        = 500 
def AnatomicalRegions_MinHoleSize    = 40 
AnatomicalRegionsClassNames        = ["WholeTissue"] //["epithel", "non_epithel"]
AnatomicalRegionsExpansionMicrons  = [5] //[7, 2] // Nuclei expansion parameter for different anatomical regions



// =======================================================================================================
// ===================  Code Begins - Dont Change from here downward  ====================================
// =======================================================================================================

def cal = getCurrentServer().getPixelCalibration()
def pixelWidth = cal.pixelWidth
def pixelHeight = cal.pixelHeight
def AnatomicalRegionsExpansionPixels = AnatomicalRegionsExpansionMicrons.collect { it / pixelWidth }

def cellClass = getPathClass(cellClassName)
var imageData = getCurrentImageData()

// imageData for cellpose 
// Note here that it is the imageData we created above and not the result of getCurrentImageData()
// Some server magic. Extract channels of interest and project them.
//def avgServer = new TransformedServerBuilder( getCurrentServer() ).extractChannels(membraneChannels[0], membraneChannels[1]).averageChannelProject().build()

def serverBuilder = new TransformedServerBuilder(getCurrentServer())
if (membraneChannels.size() == 0) {
    print "No channels specified in membraneChannels"
    return
}
// Extract all specified channels
serverBuilder = serverBuilder.extractChannels(*membraneChannels)
def memChannelName = membraneChannels[0]
// If more than 1 channel, apply average projection
if (membraneChannels.size() > 1) {
    serverBuilder = serverBuilder.averageChannelProject()
    memChannelName = "Average channels"
} 
// Finalize the server
def avgServer = serverBuilder.build()
// Extract the one other channel we want
def singleChannel = new TransformedServerBuilder( getCurrentServer() ).extractChannels(nucChannel).build()
// Make a combined server. Notice the order here is DAPI first, then the average
def combined = new ConcatChannelsImageServer( getCurrentServer(), [singleChannel, avgServer] )
// Need to create a new in-place ImageData for cellpose later
def imageData_combined = new ImageData(combined)
               
// Channels for InstaSeg
def channels = [nucChannel] + membraneChannels  // concatenate
def transformedChannelsForInstaSeg = channels.collect { ch ->
    ColorTransforms.createChannelExtractor(ch)
} 

// ===================  Segment Whole Tissue =============================================================
if (segmentTissue) {
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
            Dialogs.showErrorMessage("Cell Segmentation", "Please select a parent object!")
            return
        }

        var stardist_nuc = StarDist2D.builder(StarDistPathModel)
              .channels(nucChannel)            
              .threshold(param_threshold)              // Prediction threshold
              .normalizePercentiles(normalize_low_pct,normalize_high_pct) // Percentile normalization
              .pixelSize(pixelWidth)              // Resolution for detection
              //.includeProbability(true)
              .classify( nucClassName )       // PathClass to give newly created objects
              .measureIntensity()
              .tileSize(param_tilesize)
              .measureShape()
              //.cellExpansion(param_expansion) //Cell expansion in microns
              .build()
        
        var stardist_cells = StarDist2D.builder(StarDistPathModel)
              .channels(nucChannel)            
              .threshold(param_threshold)              // Prediction threshold
              .normalizePercentiles(normalize_low_pct,normalize_high_pct) // Percentile normalization
              .pixelSize(pixelWidth)              // Resolution for detection
              //.includeProbability(true)
              .classify( cellClassName )       // PathClass to give newly created objects
              .measureIntensity()
              .tileSize(param_tilesize)
              .measureShape()
              //.cellExpansion(param_expansion) //Cell expansion in microns
              .cellExpansion(AnatomicalRegionsExpansionMicrons[k]) //Cell expansion in microns
              .build()

        def cellpose_cells = Cellpose2D.builder( CellposeCellModel )
                .pixelSize(pixelWidth)             // Resolution for detection in um
                //.channels( "Average channels", nucChannel )	      // Select detection channel(s)
                .channels( memChannelName, nucChannel )	      // Select detection channel(s)
                .normalizePercentilesGlobal(0.1, 99.8, 10)
                .tileSize(param_tilesize)                  // If your GPU can take it, make larger tiles to process fewer of them. 
                //.cellposeChannels(1,2)         // Need these, otherwise it just sends the one channel. These will be sent directly to --chan and --chan2
                                                // OG - CHECK IF WE NEED THE ABOVE
                .diameter(CellposeCellDiameter)                    // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
                .classify( cellClassName )       // PathClass to give newly created objects
                .measureShape()                // Add shape measurements
                .measureIntensity()             // Add cell measurements (in all compartments)  
                .simplify(0)                   // Simplification 1.6 by default, set to 0 to get the cellpose masks as precisely as possible
                .build()
        
        def cellpose_nuc = Cellpose2D.builder( CellposeNucModel )
                .pixelSize(pixelWidth)             // Resolution for detection in um
                .channels( nucChannel )	      // Select detection channel(s)
                .normalizePercentilesGlobal(0.1, 99.8, 10)
                .tileSize(param_tilesize)                  // If your GPU can take it, make larger tiles to process fewer of them. 
                //.cellposeChannels(1,2)         // Need these, otherwise it just sends the one channel. These will be sent directly to --chan and --chan2
                .diameter(CellposeNucDiameter)                    // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
                //.cellExpansion(5.0)              // Approximate cells based upon nucleus expansion
                .classify( nucClassName )       // PathClass to give newly created objects
                .measureShape()                // Add shape measurements
                .measureIntensity()             // Add cell measurements (in all compartments)  
                .simplify(0)                   // Simplification 1.6 by default, set to 0 to get the cellpose masks as precisely as possible
                .build()

        def cellpose_expand_nuc = Cellpose2D.builder( CellposeNucModel )
                .pixelSize(pixelWidth)             // Resolution for detection in um
                .channels( nucChannel )	      // Select detection channel(s)
                .normalizePercentilesGlobal(0.1, 99.8, 10)
                .tileSize(param_tilesize)                  // If your GPU can take it, make larger tiles to process fewer of them. 
                //.cellposeChannels(1,2)         // Need these, otherwise it just sends the one channel. These will be sent directly to --chan and --chan2
                .diameter(CellposeNucDiameter)                    // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
                .cellExpansion(AnatomicalRegionsExpansionMicrons[k])    // Approximate cells based upon nucleus expansion
                .classify( cellClassName )       // PathClass to give newly created objects
                .measureShape()                // Add shape measurements
                .measureIntensity()             // Add cell measurements (in all compartments)  
                .simplify(0)                   // Simplification 1.6 by default, set to 0 to get the cellpose masks as precisely as possible
                .build()

        def cellpose_sam_cells = Cellpose2D.builder( CellposeCellModel )
                .pixelSize(pixelWidth)             // Resolution for detection in um
                //.channels( "Average channels", nucChannel )	      // Select detection channel(s)
                .channels( memChannelName, nucChannel )	      // Select detection channel(s)
                .normalizePercentilesGlobal(0.1, 99.8, 10)
                .tileSize(param_tilesize)                  // If your GPU can take it, make larger tiles to process fewer of them. 
                //.cellposeChannels(1,2)         // Need these, otherwise it just sends the one channel. These will be sent directly to --chan and --chan2
                                                // OG - CHECK IF WE NEED THE ABOVE
                .diameter(CellposeCellDiameter)                    // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
                .useCellposeSAM()  
                .classify( cellClassName )       // PathClass to give newly created objects
                .measureShape()                // Add shape measurements
                .measureIntensity()             // Add cell measurements (in all compartments)  
                .simplify(0)                   // Simplification 1.6 by default, set to 0 to get the cellpose masks as precisely as possible
                .build()
        
        def cellpose_sam_nuc = Cellpose2D.builder( CellposeNucModel )
                .pixelSize(pixelWidth)             // Resolution for detection in um
                .channels( nucChannel )	      // Select detection channel(s)
                .normalizePercentilesGlobal(0.1, 99.8, 10)
                .tileSize(param_tilesize)                  // If your GPU can take it, make larger tiles to process fewer of them. 
                //.cellposeChannels(1,2)         // Need these, otherwise it just sends the one channel. These will be sent directly to --chan and --chan2
                .diameter(CellposeNucDiameter)                    // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
                .useCellposeSAM()  
                //.cellExpansion(5.0)              // Approximate cells based upon nucleus expansion
                .classify( nucClassName )       // PathClass to give newly created objects
                .measureShape()                // Add shape measurements
                .measureIntensity()             // Add cell measurements (in all compartments)  
                .simplify(0)                   // Simplification 1.6 by default, set to 0 to get the cellpose masks as precisely as possible
                .build()

        def cellpose_sam_expand_nuc = Cellpose2D.builder( CellposeNucModel )
                .pixelSize(pixelWidth)             // Resolution for detection in um
                .channels( nucChannel )	      // Select detection channel(s)
                .normalizePercentilesGlobal(0.1, 99.8, 10)
                .tileSize(param_tilesize)                  // If your GPU can take it, make larger tiles to process fewer of them. 
                //.cellposeChannels(1,2)         // Need these, otherwise it just sends the one channel. These will be sent directly to --chan and --chan2
                .diameter(CellposeNucDiameter)                    // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
                .cellExpansion(AnatomicalRegionsExpansionMicrons[k])    // Approximate cells based upon nucleus expansion
                .useCellposeSAM()  
                .classify( cellClassName )       // PathClass to give newly created objects
                .measureShape()                // Add shape measurements
                .measureIntensity()             // Add cell measurements (in all compartments)  
                .simplify(0)                   // Simplification 1.6 by default, set to 0 to get the cellpose masks as precisely as possible
                .build()

         def instaseg_cells = qupath.ext.instanseg.core.InstanSeg.builder()
                .modelPath(InstaSegModel)
                .device(InstaSeg_device)
                .inputChannels(transformedChannelsForInstaSeg)
                .outputChannels()
                .tileDims(InstaSeg_tileDims)
                .interTilePadding(InstaSeg_interTilePadding)
                .nThreads(InstaSeg_nThreads)
                .makeMeasurements(true)
                .randomColors(false)
                .outputType("Default")
                .build()
                //.detectObjects()

        // Run detection for the selected objects
        switch (cellSegmentationMethod) 
        {
           case "cellBorders":
               if (segmentationAlg == "cellpose") {
                   println "cellSegmentationMethod is cellBorders, segmentationAlg is cellpose"
                   if (useCellposeSAM)
                       cellpose_sam_cells.detectObjects(imageData_combined, pathObjects)               
                   else
                       cellpose_cells.detectObjects(imageData_combined, pathObjects)               
               }
               if (segmentationAlg == "instaseg") {
                   println "cellSegmentationMethod is cellBorders, segmentationAlg is instaseg"
                   instaseg_cells.detectObjects(imageData, pathObjects)
               }
               if (segmentationAlg == "stardist") {
                   println "stardist does not suite   cellBorders  segmentation mode. switch to   expandNuc"  or  onlyNuc
               }
               break
           case "expandNuc":
               println "cellSegmentationMethod is expandNuc"
               if (segmentationAlg == "stardist") {
                   println "cellSegmentationMethod is expandNuc, segmentationAlg is stardist"
                   stardist_cells.detectObjects(imageData, pathObjects)                  
               }
               if (segmentationAlg == "cellpose") {
                   println "cellSegmentationMethod is expandNuc, segmentationAlg is cellpose"
                   if (useCellposeSAM)
                       cellpose_sam_expand_nuc.detectObjects(imageData, pathObjects) 
                   else
                       cellpose_expand_nuc.detectObjects(imageData, pathObjects) 
               }
               break
           case "onlyNuc":
               println "cellSegmentationMethod is onlyNuc"
               if (segmentationAlg == "stardist") {
                   println "cellSegmentationMethod is onlyNuc, segmentationAlg is stardist"
                   stardist_nuc.detectObjects(imageData, pathObjects)                  
               }
               if (segmentationAlg == "cellpose") {
                   println "cellSegmentationMethod is onlyNuc, segmentationAlg is cellpose"
                   if (useCellposeSAM)
                       cellpose_sam_nuc.detectObjects(imageData, pathObjects)                  
                   else
                       cellpose_nuc.detectObjects(imageData, pathObjects)                  
               }
               break
           default:
               println "Unknown method"           
        }
        // OG  keep region class ?? 
        //for (cell in getCellObjects()) {cell.setPathClass(cellClass) }
        
        println '======================== Cell segmentation on '+AnatomicalRegionsClassNames[k]+' Done =========================== '
    }
}

resolveHierarchy()

// =======================================================================================================
// ===================  Load Visium HD Spots and associate them to Cells =================================
def spot_diameter_fullres = 1 // dummy value
if (loadSpots) {
    // Extract scale factors
    // Read the file content
    def jsonText = new String(Files.readAllBytes(Paths.get(scalefactors_json)))
    
    // Use regex to extract the value of "spot_diameter_fullres"
    def pattern = ~/\"spot_diameter_fullres\"\s*:\s*([0-9.]+)/
    def matcher = pattern.matcher(jsonText)
    
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
    addObjects(listOfObjects);
    println '======================== Spots imported ==================================='
}

if (associateSpotsToCells) {
    println '======================== Associating Spots To Cells ... ==================================='
    def cellObj = getDetectionObjects().findAll{!it.isTile()} 
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
        if (cell.isCell())
        {
            if (cell.getNucleusROI()) {
                def subNucObj = getCurrentHierarchy().getObjectsForROI(null, cell.getNucleusROI()) // get all objects within each cellObj Nucleus ROI
                for (s in subNucObj) {
                    if (s.isTile()) {
                        nNucSpots++
                        s.measurements.put("InNuc", 1)
                    }
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

// ===================  Export Cells As GeoJson ====================================
if (exportCellsAsGeoJson) {    
    println '======================== save Cells as GeoJson ... ==================================='
    
    File directory = new File(buildFilePath(PROJECT_BASE_DIR,resultsSubFolder));
    directory.mkdirs();
    imageName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())

    //cells = getCellObjects()
    cells = getDetectionObjects().findAll{ !it.isTile() };
    selectObjects(cells)  
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
import qupath.lib.scripting.QP
import qupath.ext.biop.cellpose.Cellpose2D
import qupath.lib.images.servers.TransformedServerBuilder
import qupath.lib.images.ImageData
import groovy.time.*
import java.io.BufferedReader;
import java.io.FileReader;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathTileObject;
import qupath.lib.roi.ROIs
import qupath.lib.roi.RectangleROI
import qupath.lib.roi.RoiTools.CombineOp
import qupath.ext.biop.cellpose.Cellpose2D
import qupath.lib.images.ImageData
import qupath.lib.images.servers.ConcatChannelsImageServer
import qupath.lib.images.servers.TransformedServerBuilder
import java.nio.file.Files
import java.nio.file.Paths
import java.util.regex.Matcher
import java.util.regex.Pattern
import qupath.lib.analysis.features.ObjectMeasurements
import java.util.stream.Collectors;
import org.locationtech.jts.algorithm.locate.IndexedPointInAreaLocator;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.PrecisionModel;
import qupath.lib.analysis.DistanceTools

