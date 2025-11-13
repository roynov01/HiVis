// VisiumHDAnalysis - Quantify VisiumHD Mouse Liver Flourescent Slide 
// 
// By: Ofra Golani, for Roy Novoselsky, Shalev Itzkovitz
//
// Workflow
// ============================================================
// 
// - Segment Tissue region using pixel classifier - blood vessels / empty / tissue 
//   Filter out objects by size/shape
// - Segment hepatocytes Cells and Nucs within Tissue using Cellpose - use model trained on double-nucleated cells for cells, cyto3 for Nuc
//   Associate Nucs to cells (0/1/2)
//   keep the Nucs inside cells as separate objects, with parent id encoded in their name
// - Segment cells in blood-vesseles regions using Stardist based on DAPI+expansion 
// - Load VisiumHD spots (bins) as detections
// - Associate spots to Cells and Nucs, set the inCell/inNuc flag
// - for each spot and cell calculate pixel classifier probability 
// - Measure distance of cells/nucs/spots from anatomical regions (created by the pixel classifier)
// - Measure distance of spots to the parent cell border and to its nuc
// - Export cell and nuc ROIs as geojson
// - Export features for all detections : cell/nuc/spots
//
// QuPath 0.6.0 compatible script
// 
// sample_data: mouse_liver
//
// ===================  Workflow control Parameters  =========================================================

def segmentTissue               = 1 
def createAnatomicalRegionsFromPixelClassifier = 1
def segmentCells                = 1 
def loadSpots                   = 1
def associateSpotsToCells       = 1
def runPixelClassifierForSpot   = 0
def runPixelClassifierForCell   = 0 
def AddDistanceMeasurements     = 1
def measureSpotZonationInCell   = 1
def exportCellsAndNucsAsGeoJson = 1
def saveResultTable             = 1 // export result table to Tab-separated txt file

// ===================  File paths  ==========================================================================
def resultsSubFolder = 'results_mouse_liver'
def scalefactors_json = 'A:/royno/HiVis_proj_v2/datasets/mouse_liver_98_WT_scalefactors_json.json' // Use Full path
def csvfile = 'A:/royno/HiVis_proj_v2/datasets/mouse_liver_98_WT_tissue_positions.csv' // mouse_liver - TO CHANGE

// ===================  Pixel classifier and anatomical region expansion  ====================================
def wholeTissueClass = "WholeTissue" 
def BVClass          = "Blood_vessel" // 
def hepatoClass      = "WholeTissue"  //  
            
def WholeTissueClassifier   = "Whole Tissue" // name of WholeTissue pixel classifier
def WholeTissue_MinSize     =  25            // Minimal WholeTissue connected-component size        
def WholeTissue_MinHoleSize = 50             // minmal hole size to keep when creating WholeTissue regions, samller holes are filled 

// ===================  Pixel Classifier for Anatomical regions   ====================================
def PixelClassifier = "blood_vessels_fullres"
def minObjectSize   = 10
def minHoleSize     = 100
def minEmptyArea    = 400
def maxEmptyArea    = 500000

// ===================  Segmentation parameters  =============================================================
// Cell detection control 
def runCellDetection   = 1
def runNucDetection    = 1
def combineNucAndCells = 1
def keepNucsInCells    = 1
def segmentBVCells     = 1

var cellClassName = "Cell"
var spotClassName = "Spot"

// ===================  Cell Segmentation parameters  ====================================
// hepatocytes parameters 
//def pathModel_cyto = 'cyto3' // Specify the model name (cyto, nuclei, cyto2, ... or a path to your custom model as a string)
// Path to costum model 
def pathModel_cyto = 'A:/royno/Visium_HD_liver/experiment1/qupath_project/models/Custom_model_2025-01-08_17_02.cpm'
def pathModel_nuc = 'cyto3' // Other models for Cellpose https://cellpose.readthedocs.io/en/latest/models.html
def nucChannel       = 3
def membChannel      = 0
def nucDiameter      = 21
def membDiameter     = 50

// bv cells parameters 
def use_cellpose_for_bv_cells = 0 // 1=use cellpose, 0=use stardist
def pathModel_bv_nuc = 'nuc'
def bvNucDiameter    = 15
double bvCellExpansionMicrons = 2 //size of cell expansion in microns. 
//def bv_stardist_modelPath = "A:/shared/QuPathScriptsAndProtocols/QuPath_StarDistModels/dsb2018_heavy_augment.pb"
def bv_stardist_modelPath = "A:/shared/QuPathScriptsAndProtocols/QuPath_StarDistModels/stardist_for_vishnu_v5.pb"
def bv_stardist_threshold = 0.3 //0.3
def bv_MaxNucArea         = 80 
def bv_MinNucArea         = 10   
def bv_MinNucIntensity    = 18000 //17000 //17000 //remove any detections with an intensity less than or equal to this value          


// =======================================================================================================
// ===================  Code Begins - Dont Change from here downward  ====================================
// =======================================================================================================

// Get Pixel size from the image
def server = getCurrentServer()
def cal = server.getPixelCalibration()
def pixelWidth = cal.pixelWidth
def pixelHeight = cal.pixelHeight
double bvCellExpansionPixels = bvCellExpansionMicrons/pixelWidth

def cellClass = getPathClass(cellClassName)
var imageData = getCurrentImageData()
def downsample = 1.0


// ===================  Segment Whole Tissue =============================================================
if (segmentTissue) {
    createAnnotationsFromPixelClassifier(WholeTissueClassifier, WholeTissue_MinSize, WholeTissue_MinHoleSize)
    
    println '======================== Whole Tissue segmentation Done =================== '
}

// ===================  Segment Anatomical regions  ======================================================
if (createAnatomicalRegionsFromPixelClassifier) {

    println "[DEBUG] createAnatomicalRegionsFromPixelClassifier start, Number of annotation objects: ${getAnnotationObjects().size()}"
    
    selectObjectsByClassification(wholeTissueClass);
    createAnnotationsFromPixelClassifier(PixelClassifier, minObjectSize, minHoleSize)

    // filter objects by size
    selectObjectsByClassification("empty");
    runPlugin('qupath.lib.plugins.objects.SplitAnnotationsPlugin', '{}')
        
    //AreaMeasurement='Area µm^2' //Name of the measurement you want to perform filtering on
    toDelete =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("empty") && (it.getROI().getScaledArea(pixelWidth, pixelHeight) > maxEmptyArea) }
    removeObjects(toDelete, true)
    toDelete1 =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("empty") && (it.getROI().getScaledArea(pixelWidth, pixelHeight) < minEmptyArea) }
    removeObjects(toDelete1, true)

    //selectObjectsByClassification("empty");
    toMerge =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("empty") }
    if (toMerge.size() > 0) {
        selectObjectsByClassification("empty");
        mergeSelectedAnnotations()
    }
        
    println "[DEBUG] createAnatomicalRegionsFromPixelClassifier end, Number of annotation objects: ${getAnnotationObjects().size()}"
}


// ===================  Segment Cells within whole Tissue: detect hepato-Cells and Nuc, associate Nuc to cells, expand Nuc not in hapto-cells within blood vessel rehions ====================================
if (segmentCells) {

    println "[DEBUG] Segment Cells start, Number of annotation objects: ${getAnnotationObjects().size()}"
    
    // Run hepato-cells segmentation
    resetSelection()
    selectObjectsByClassification(hepatoClass);

    var pathObjects = getSelectedObjects()
    if (pathObjects.isEmpty()) {
        Dialogs.showErrorMessage("hepato-Cells Segmentation", "Please select a parent object!")
        return
    }

    // Cell Detection 
    // =========================================================================================
    def cellpose_cyto = Cellpose2D.builder( pathModel_cyto )
            .pixelSize( pixelWidth )                  // Resolution for detection in um
            .channels(membChannel,nucChannel)	      // Select detection channel(s)
            .diameter( membDiameter )                // Median object diameter. 
            //.measureShape()                           // Add shape measurements
            //.measureIntensity()                       // Add cell measurements (in all compartments)
            .build()
    
    println "[DEBUG] Segment Cells before hepato cellpose, Number of annotation objects: ${getAnnotationObjects().size()}"
    // work-around to keep anatomical regions annotations
    empty_regions =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("empty") }
    hepato_regions =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("hepato") }
    bv_regions =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("Blood_vessel") }
    
    if (runCellDetection) {
            
        println "[INFO] Running Cellpose detection for cytoplasm..."
        cellpose_cyto.detectObjects( imageData, pathObjects )
        println "[DEBUG] Segment Cells after hepato cellpose, Number of annotation objects: ${getAnnotationObjects().size()}"        
        println "[DEBUG] Detection complete. Checking detected objects..."
        cytos = getDetectionObjects()
        println "[DEBUG] Number of detected cytoplasm objects: ${cytos.size()}"
        cytos.each{it.setPathClass(getPathClass("Cyto"))}
        println '[DEBUG] - finished Cellpose for cytoplasm'
                
    } 
        
    // hepatocyte Nuc Detection 
    // =========================================================================================
    def cellpose_nuc = Cellpose2D.builder( pathModel_nuc )
            .pixelSize( pixelWidth )                  // Resolution for detection in um
            .channels(nucChannel)	              // Select detection channel(s)
            .diameter( nucDiameter )                 // Median object diameter. 
            .measureShape()                           // Add shape measurements
            .measureIntensity()                       // Add cell measurements (in all compartments)
            .build()
    
    if (runNucDetection)
    {   
        println "[INFO] Running Cellpose detection for nuclei..."
        cellpose_nuc.detectObjects(imageData, pathObjects)
        nucs = getDetectionObjects()
        nucs.each{ it.setPathClass(getPathClass("Nuc"))}
        if (cytos) addObjects(cytos) // needed because cellpose detectors remove existing detections
    }

    // work-around to keep anatomical regions annotations
    empty_regions1 =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("empty") }
    hepato_regions1 =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("hepato") }
    bv_regions1 =  getAnnotationObjects().findAll { it.getPathClass() == getPathClass("Blood_vessel") }
    if (empty_regions.size() != empty_regions1.size()) {
        if (empty_regions1.size() > 0) {
            selectObjectsByClassification("empty");
            clearSelectedObjects() 
        }
        addObjects(empty_regions)
    }
    if (hepato_regions.size() != hepato_regions1.size()) {
        if (hepato_regions1.size() > 0) {
            selectObjectsByClassification("hepato");
            clearSelectedObjects() 
        }
        addObjects(hepato_regions)
    }
    if (bv_regions.size() != bv_regions1.size()) {
        if (bv_regions1.size() > 0) {
            selectObjectsByClassification("Blood_vessel");
            clearSelectedObjects() 
        }
        addObjects(bv_regions)
    }
    println "[DEBUG] Segment Cells after work-around, Number of annotation objects: ${getAnnotationObjects().size()}"        

    // Nuc Cell Association
    // =========================================================================================
    if (combineNucAndCells) 
    {    
        println "[INFO] Running Nuc-Cell Association ..."
        cytos = getDetectionObjects().findAll{ it.getPathClass() == getPathClass("Cyto") } 
        print("nCyto = " + cytos.size())
        nucs  = getDetectionObjects().findAll{ it.getPathClass() == getPathClass("Nuc") } 
        print("nNuc = " + nucs.size())
        // make sure to clear everything 
        bv_cells = getDetectionObjects().findAll{ it.getPathClass() == getPathClass("BV-Cell") } 
        clearDetections()    
      
        // Combine cytos and nuclei detections to create cell objects
        // (we simply check that the nuclei center is inside the cell center) 
        nucs.each{ nuc ->      
           nuc.measurements.put("inHepato", 0)
        }
        
        cells = []
        cIdx = 0;
        cells_nucs = []
        cytos.each{ cyto ->
            cyto_nucs = []
            nucs.each{ nuc ->      
                if ( cyto.getROI().contains( nuc.getROI().getCentroidX() , nuc.getROI().getCentroidY())) {
                   cyto_nucs.add(nuc)  
                   nuc.measurements.put("inHepato", 1)
                }
            }
            nNuc = cyto_nucs.size()
            if (nNuc == 0) {
                //print("cIdx"+cIdx+", nNuc=0")
                cell = PathObjects.createCellObject(cyto.getROI(), null, getPathClass("Cell-noNuc"), null );
                cell.measurements.put("nNuc", nNuc)
                cells.add(cell);
            }
            else if (nNuc == 1) {
                //print("cIdx"+cIdx+", nNuc=1")
                cell = PathObjects.createCellObject(cyto.getROI(), cyto_nucs[0].getROI(), getPathClass("Cell-oneNuc"), null );
                cell.measurements.put("nNuc", nNuc)
                cells.add(cell);
    
                if (keepNucsInCells) {
                    cnuc = cyto_nucs[0]
                    parentName = cell.getID().toString()
                    nName = cnuc.getName()
                    newName = String.join("",nName, "__", parentName)
                    cnuc.setName(newName)
                    cnuc.setPathClass(getPathClass("NucInCell"))
                    cells_nucs.add(cnuc)
                }    
            } else if (nNuc == 2) {
                //print("cIdx"+cIdx+", nNuc=2")
                def rois = cyto_nucs.collect { it.getROI() }
                def combine_roi  = RoiTools.union(rois)
                def cell = PathObjects.createCellObject(cyto.getROI(), combine_roi, getPathClass("Cell-twoNuc"), null );
                cell.measurements.put("nNuc", nNuc)
                cells.add(cell);
    
                if (keepNucsInCells) {
                    cyto_nucs.each { cnuc ->
                        parentName = cell.getID().toString()
                        nName = cnuc.getName()
                        newName = String.join("",nName, "__", parentName)
                        cnuc.setName(newName)
                        cnuc.setPathClass(getPathClass("NucInCell"))
                        cells_nucs.add(cnuc)
                    }    
                }
            }
            cIdx++
        } // loop on cyto
        print("nCells = " + cells.size())
        addObjects(cells)
        if (keepNucsInCells) {
            print("nCells_Nucs = " + cells_nucs.size())
            addObjects(cells_nucs)
        }
                
        // Intensity & Shape Measurements
        // adapted from : https://forum.image.sc/t/transferring-segmentation-predictions-from-custom-masks-to-qupath/43408/12
        def measurements = ObjectMeasurements.Measurements.values() as List
        def compartments = ObjectMeasurements.Compartments.values() as List // Won't mean much if they aren't cells...
        def shape = ObjectMeasurements.ShapeFeatures.values() as List
        def cells = getCellObjects()
        for ( cell in cells) {
            ObjectMeasurements.addIntensityMeasurements( server, cell, downsample, measurements, compartments )
            ObjectMeasurements.addCellShapeMeasurements( cell, cal,  shape )
        }        
        if (!bv_cells.isEmpty()) 
            addObjects(bv_cells)
    }    
    // Segment BV cells  
    if (segmentBVCells) {
        selectObjectsByClassification(BVClass);
    
        var bvObjects = getSelectedObjects()
        if (bvObjects.isEmpty()) {
            Dialogs.showErrorMessage("BV-Cells segmentation", "Please select a parent object!")
            return
        }

        def hapatoCellObj = getCellObjects().findAll{ it.getPathClass() == getPathClass("Cell-noNuc") || it.getPathClass() == getPathClass("Cell-oneNuc") || it.getPathClass() == getPathClass("Cell-twoNuc") }  
        def nucInCellObj = getDetectionObjects().findAll{ it.getPathClass() == getPathClass("NucInCell") } 

        // BV Cell Detection 
        // =========================================================================================
        if (use_cellpose_for_bv_cells) {
            println "[INFO] Running Cellpose detection for BV-Cells..."

            def cellpose_bv_cells = Cellpose2D.builder( pathModel_bv_nuc )
                    .pixelSize( pixelWidth )                  // Resolution for detection in um
                    .channels( nucChannel)          // Select detection channel(s)
                    //.normalizePercentilesGlobal( 0.1, 99.8, 10 ) // Convenience global percentile normalization. arguments are percentileMin, percentileMax, dowsample.
                    .diameter( bvNucDiameter )                 // Median object diameter. 
                    //.cellprobThreshold( 0.0 )
                    //.flowThreshold( 0.4 )
                    .cellExpansion( bvCellExpansionMicrons )  // Approximate cells based upon nucleus expansion
                    .measureShape()                           // Add shape measurements
                    .measureIntensity()                       // Add cell measurements (in all compartments)
                    .classify( "BV-Cell" )
                    .build()
            
            println "[INFO] Running Cellpose detection for BV..."
            cellpose_bv_cells.detectObjects( imageData, bvObjects )
        } else {
            println "[INFO] Running StarDist detection for BV-Cells..."
            // Customize how the StarDist detection should be applied
            def stardist = StarDist2D
                .builder(bv_stardist_modelPath)
                //.channels(nucChannel)            // Extract channel called 'DAPI'
                .channels("Channel 4")            // Extract channel called 'DAPI'
                .normalizePercentiles(1, 99) // Percentile normalization
                //.normalizePercentiles(2, 98) // Percentile normalization
                .threshold(bv_stardist_threshold)              // Probability (detection) threshold
                .pixelSize(pixelWidth)              // Resolution for detection
                .cellExpansion(bvCellExpansionMicrons)            // Expand nuclei to approximate cell boundaries
                .measureShape()              // Add shape measurements
                .measureIntensity()          // Add cell measurements (in all compartments)
                .includeProbability(true)
                .classify( "BV-Cell" )
                .build()
                
            stardist.detectObjects(imageData, bvObjects)
            stardist.close() // This can help clean up & regain memory
            
            // Filter bv-cells by size and Intensity
            def NucAreaMeasurement='Nucleus: Area µm^2' //Name of the measurement you want to perform filtering on
            
            def cell_toDelete =  getDetectionObjects().findAll {measurement(it, NucAreaMeasurement) > bv_MaxNucArea}
            removeObjects(cell_toDelete, true)
            def cell_toDelete1 =  getDetectionObjects().findAll {measurement(it, NucAreaMeasurement) < bv_MinNucArea}
            removeObjects(cell_toDelete1, true)
            
            //def NucIntensityMeasurement='Channel 4: Nucleus: Mean' //Name of the measurement you want to perform filtering on   // for QP5
            def NucIntensityMeasurement='Nucleus: Channel 4: Mean' //Name of the measurement you want to perform filtering on     // for QP6
            def cell_toDelete2 = getDetectionObjects().findAll {measurement(it, NucIntensityMeasurement) <= bv_MinNucIntensity}
            removeObjects(cell_toDelete2, true)                                                           
        }
        println "[DEBUG] Detection complete. Checking detected objects..."
        bv_cells = getDetectionObjects()
        println "[DEBUG] Number of detected bv objects: ${bv_cells.size()}"
        
        
        // work-around to keep anatomical regions annotations
        hapatoCellObj1 = getCellObjects().findAll{ it.getPathClass() == getPathClass("Cell-noNuc") || it.getPathClass() == getPathClass("Cell-oneNuc") || it.getPathClass() == getPathClass("Cell-twoNuc") }  
        nucInCellObj1 = getDetectionObjects().findAll{ it.getPathClass() == getPathClass("NucInCell") } 
        if (hapatoCellObj.size() != hapatoCellObj1.size()) {
            if (hapatoCellObj1.size() > 0) {
                selectObjects(hapatoCellObj)
                clearSelectedObjects() 
            }
            addObjects(hapatoCellObj)
        }
        if (nucInCellObj.size() != nucInCellObj1.size()) {
            if (nucInCellObj1.size() > 0) {
                selectObjects(nucInCellObj1)
                clearSelectedObjects() 
            }
            addObjects(nucInCellObj)
        }    
    } // ==================== End of  segmentBVCells ======================================================

    fireHierarchyUpdate()

    println "[DEBUG] Segment Cells end, Number of annotation objects: ${getAnnotationObjects().size()}"
    println '======================== Cell segmentation Done =========================== '
}


// ===================  Load Visium HD Spots and associate them to Cells ====================================
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
    
    println "[INFO] Loading Spots ..."
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

            /*String barcode = rowContent[0] as String;
            int  in_tissue = 1 //rowContent[2] as int;
            //int  array_row = rowContent[3] as int;
            //int  array_col = rowContent[4] as int;
            double cx = rowContent[1] as double;
            double cy = rowContent[2] as double;*/

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
                //detection.getMeasurementList().putMeasurement("array_col", array_col);
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
    // ========= Associate spots to hepato-Cells ==========================
    idx=0
    def cellObj = getCellObjects().findAll{ it.getPathClass() == getPathClass("Cell-noNuc") || it.getPathClass() == getPathClass("Cell-oneNuc") || it.getPathClass() == getPathClass("Cell-twoNuc") }  
    for (def cell in cellObj) {
        def subCellObj = getCurrentHierarchy().getObjectsForROI(null, cell.getROI()) // get all objects within each cellObj ROI
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
        if (cell.getNucleusROI() ) {
            //print(idx + ": "+ parentName)
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
        idx++
    }

    // ========= Associate spots also to Nuc ==========================
    def nucs_in_cells = getDetectionObjects().findAll{ it.getPathClass() == getPathClass("NucInCell") } 

    for (def cnuc in nucs_in_cells) {
        def subCellObj = getCurrentHierarchy().getObjectsForROI(null, cnuc.getROI()) // get all objects within each cellObj RO        מSפםא = 0
        nSpots = 0
        for (s in subCellObj) {
            if (s.isTile()) {
                nSpots++
                parentName = cnuc.getID().toString()
                sName = s.getName()
                newName = String.join("",sName, "++", parentName)
                s.setName(newName)
            }        
        }
        cnuc.measurements.put("nSpots",nSpots)
        cnuc.measurements.put("nNucSpots",nSpots)
    }

    // ========= Associate spots also to bv-Cells ==========================
    def bv_cellObj = getCellObjects().findAll{ it.getPathClass() == getPathClass("BV-Cell") }  
    for (def bv_cell in bv_cellObj) {
        def bv_subCellObj = getCurrentHierarchy().getObjectsForROI(null, bv_cell.getROI()) // get all objects within each cellObj ROI
        nSpots = 0
        for (s in bv_subCellObj) {
            if (s.isTile()) {
                if (s.measurements.get("InCell") == 0) {
                    nSpots++
                    parentName = bv_cell.getID().toString()
                    sName = s.getName()
                    //newName = sName + "__" + parentName;
                    //newName = String.format("%s__%s", sName, parentName)
                    newName = String.join("",sName, "**", parentName)
                    s.setName(newName)
                    s.measurements.put("InCell", 1)
                    //s.measurements.put("InNuc", 0) // change inNuc property only if not in hepato cell - name contains __
                }        
            }
        }
        nNucSpots = 0
        if (bv_cell.getNucleusROI() ) {
            //print(idx + ": "+ parentName)
            def bv_subNucObj = getCurrentHierarchy().getObjectsForROI(null, bv_cell.getNucleusROI()) // get all objects within each cellObj Nucleus ROI
            for (s in bv_subNucObj) {
                sName = s.getName()
                if (s.isTile()) {
                    if (!sName.contains("__")) {
                        nNucSpots++
                        s.measurements.put("InNuc", 1) // change inNuc property only if not in hepato cell - name contains __
                    }
                }
            }
        }
        bv_cell.measurements.put("nSpots",nSpots)
        bv_cell.measurements.put("nNucSpots",nNucSpots)
        idx++
    }

    
    println '======================== Associate Spots To Cells Done ==================== '
}


// ===================  Measure distance from spot to Cel border and Cell Nuclei - for heapto cells only  ====================================
if (measureSpotZonationInCell) {

    //var cal1 = server.getPixelCalibration();
    //double pixelWidth1 = cal1.getPixelWidth().doubleValue();
    //double pixelHeight1 = cal1.getPixelHeight().doubleValue();
    
    var transform = pixelWidth == 1 && pixelHeight == 1 ? null : AffineTransformation.scaleInstance(pixelWidth, pixelHeight);
    PrecisionModel precisionC = null;
    //PrecisionModel precisionN = null;
    
    def cellObj = getCellObjects().findAll{ it.getPathClass() == getPathClass("Cell-noNuc") || it.getPathClass() == getPathClass("Cell-oneNuc") || it.getPathClass() == getPathClass("Cell-twoNuc") }  
    
    for (def cell in cellObj) 
    {
        cellGeometry = cell.getROI().getGeometry()    
        if (transform != null) {
	    cellGeometry = transform.transform(cellGeometry);
	    if (precisionC == null)
		precisionC = cellGeometry.getPrecisionModel();
	}        
        var precisionModelC = precisionC == null ? GeometryTools.getDefaultFactory().getPrecisionModel() : precisionC;
        //var locatorC = null
        var locatorC = cellGeometry == null ? null : new IndexedPointInAreaLocator(cellGeometry);
				// See https://github.com/locationtech/jts/issues/571
	if (locatorC != null)
	    locatorC.locate(new Coordinate(0, 0));    

        var nucGeometry = null
        var locatorN  = null
        if (cell.measurements.get("nNuc") > 0) 
        {                    
            nucGeometry = cell.getNucleusROI().getGeometry()         
            if (transform != null) {
    	        nucGeometry = transform.transform(nucGeometry);
    	        //if (precisionN == null)
    		//    precisionN = nucGeometry.getPrecisionModel();
    	    }
    	    //var precisionModelN = precisionN == null ? GeometryTools.getDefaultFactory().getPrecisionModel() : precisionN;
            locatorN = nucGeometry == null ? null : new IndexedPointInAreaLocator(nucGeometry);
    				// See https://github.com/locationtech/jts/issues/571
    	    if (locatorN != null)
    	        locatorN.locate(new Coordinate(0, 0));    
    	} 

        def subCellObj = getCurrentHierarchy().getObjectsForROI(null, cell.getROI()).findAll{ it.isTile() } // get all spot objects within each cellObj ROI        
        for (s in subCellObj) 
        {
            Coordinate coord = new Coordinate(s.getROI().getCentroidX() * pixelWidth, s.getROI().getCentroidY() * pixelHeight);            
            //Coordinate coord = new Coordinate(s.getROI().getCentroidX() , s.getROI().getCentroidY() );            
            precisionModelC.makePrecise(coord);
            distToCell = DistanceTools.computeDistance(coord, cellGeometry, locatorC, true)
            s.measurements.put("DistToCell", distToCell)
            if (cell.measurements.get("nNuc") > 0) {                
                distToNuc = DistanceTools.computeDistance(coord, nucGeometry, locatorN, true)
                s.measurements.put("DistToNuc", distToNuc)
            }
        }
    }   
    println '======================== Measure Spot Zonation In Cell Done ==================== '
}

// ===================  Add pixel classifier measurements to spots ====================================
if (runPixelClassifierForSpot) {
    selectTiles()
    addPixelClassifierMeasurements(PixelClassifier, PixelClassifier)
    resetSelection()
}

// ===================  Add pixel classifier measurements to cells ====================================
if (runPixelClassifierForCell) {
    //selectObjectsByClassification("Cell");
    selectCells();
    addPixelClassifierMeasurements(PixelClassifier, PixelClassifier)
    resetSelection()

    selectObjectsByClassification("NucInCell");
    addPixelClassifierMeasurements(PixelClassifier, PixelClassifier)
    resetSelection()
}

// ===================  Add Measurements to Cells  ====================================
if (AddDistanceMeasurements) {

    myDetectionToAnnotationDistances(imageData, false, false)
    
    println '======================== Add Distance Measurements Done ==================== '
}

if (exportCellsAndNucsAsGeoJson) {    
    
    File directory = new File(buildFilePath(PROJECT_BASE_DIR,resultsSubFolder));
    directory.mkdirs();
    imageName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())

    cells = getCellObjects()
    //cells = getDetectionObjects().findAll{ !it.isTile() };
    selectObjects(cells)  
    //exportSelectedObjectsToGeoJson("A:\\royno\\Visium_HD_liver\\experiment1\\qupath_project\\mouse_liver_98_WT_fullres.tif - 2d copy for alternative analysis (1).geojson", "PRETTY_JSON", "FEATURE_COLLECTION")
    exportSelectedObjectsToGeoJson(buildFilePath(directory.toString(),imageName+'_cells.geojson'), "EXCLUDE_MEASUREMENTS", "PRETTY_JSON", "FEATURE_COLLECTION")        

    nucInCell = getDetectionObjects().findAll{ it.getPathClass() == getPathClass("NucInCell") } 
    selectObjects(nucInCell)  
    //exportSelectedObjectsToGeoJson("A:\\royno\\Visium_HD_liver\\experiment1\\qupath_project\\mouse_liver_98_WT_fullres.tif - 2d copy for alternative analysis (1).geojson", "PRETTY_JSON", "FEATURE_COLLECTION")
    exportSelectedObjectsToGeoJson(buildFilePath(directory.toString(),imageName+'_nucInHepatoCells.geojson'), "EXCLUDE_MEASUREMENTS", "PRETTY_JSON", "FEATURE_COLLECTION")        

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
}

println "[DEBUG] End of script, Number of annotation objects: ${getAnnotationObjects().size()}"
println '======================== Workflow Done! ==================================='



//================================== Helper Functions ====================================
/**
 * Compute the distance for all detection object centroids to the closest annotation with each valid, not-ignored classification and add 
 * the result to the detection measurement list.
 * @param imageData
 * @param splitClassNames if true, split the classification name. For example, if an image contains classifications for both "CD3: CD4" and "CD3: CD8",
 *                        distances will be calculated for all components (e.g. "CD3", "CD4" and "CD8").
 * @param signedDistances optionally calculate signed distances, i.e. negative values for source centroids that occur inside target objects representing the distance to the target object boundary
 * @since v0.4.0
 */
//public static void myDetectionToAnnotationDistances(ImageData<?> imageData, boolean splitClassNames, boolean signedDistances) {
def myDetectionToAnnotationDistances(ImageData<?> imageData, boolean splitClassNames, boolean signedDistances) 
{
    var server = imageData.getServer();
    var hierarchy = imageData.getHierarchy();
    var annotations = hierarchy.getAnnotationObjects();
    //var detections = hierarchy.getCellObjects();
    //if (detections.isEmpty())
    var detections = hierarchy.getDetectionObjects();
    
    // TODO: Support TMA cores
    if (hierarchy.getTMAGrid() != null)
        logger.warn("Detection to annotation distances command currently ignores TMA grid information!");
    
    var pathClasses = annotations.stream()
            .map(p -> p.getPathClass())
            .filter(p -> p != null && p.isValid() && !PathClassTools.isIgnoredClass(p))
            .collect(Collectors.toSet());
    
    var cal = server.getPixelCalibration();
    String distanceString = signedDistances ? "Signed distance" : "Distance";
    String xUnit = cal.getPixelWidthUnit();
    String yUnit = cal.getPixelHeightUnit();
    double pixelWidth = cal.getPixelWidth().doubleValue();
    double pixelHeight = cal.getPixelHeight().doubleValue();
    if (!xUnit.equals(yUnit))
        throw new IllegalArgumentException("Pixel width & height units do not match! Width " + xUnit + ", height " + yUnit);
    String unit = xUnit;
    
    for (PathClass pathClass : pathClasses) {
        if (splitClassNames) {
            var names = PathClassTools.splitNames(pathClass);
            for (var name : names) {
                logger.debug("Computing distances for {}", pathClass);
                var filteredAnnotations = annotations.stream().filter(a -> PathClassTools.containsName(a.getPathClass(), name)).toList();
                if (!filteredAnnotations.isEmpty()) {
                    String measurementName = distanceString + " to annotation with " + name + " " + unit;
                    DistanceTools.centroidToBoundsDistance2D(detections, filteredAnnotations, pixelWidth, pixelHeight, measurementName, signedDistances);
                }
            }
        } else {
            logger.debug("Computing distances for {}", pathClass);
            var filteredAnnotations = annotations.stream().filter(a -> a.getPathClass() == pathClass).toList();
            if (!filteredAnnotations.isEmpty()) {
                String name = distanceString + " to annotation " + pathClass + " " + unit;
                DistanceTools.centroidToBoundsDistance2D(detections, filteredAnnotations, pixelWidth, pixelHeight, name, signedDistances);
            }
        }
    }
    hierarchy.fireObjectMeasurementsChangedEvent(DistanceTools.class, detections);
}

// ******************* imports **********************************
import qupath.ext.stardist.StarDist2D
import qupath.lib.scripting.QP
import qupath.ext.biop.cellpose.Cellpose2D
import qupath.lib.analysis.features.ObjectMeasurements
import groovy.time.*
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.stream.Collectors;
import org.locationtech.jts.algorithm.locate.IndexedPointInAreaLocator;
import org.locationtech.jts.geom.Coordinate;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathTileObject;
import org.locationtech.jts.geom.PrecisionModel;
import org.locationtech.jts.geom.util.AffineTransformation;
import qupath.lib.roi.ROIs
import qupath.lib.roi.RectangleROI
import qupath.lib.roi.RoiTools.CombineOp
import qupath.lib.analysis.DistanceTools
