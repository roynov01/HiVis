
.. autofunction:: HiVis.HiVis.new

.. autofunction:: HiVis.HiVis.load

HiVis
====================================================

Main class. Stores the data and images of the VisiumHD, enables plotting via HiVis.plot, and can store Aggregation instances in HiVis.agg.
To make a new class, call the new() function.

General methods
****************
.. automethod:: HiVis.HiVis.HiVis.copy
.. automethod:: HiVis.HiVis.HiVis.recolor
.. automethod:: HiVis.HiVis.HiVis.rename
.. automethod:: HiVis.HiVis.HiVis.update_meta
.. automethod:: HiVis.HiVis.HiVis.head
.. automethod:: HiVis.HiVis.HiVis.update

Segmentations, annotations, masks
**********************************
.. automethod:: HiVis.HiVis.HiVis.add_mask
.. automethod:: HiVis.HiVis.HiVis.add_annotations
.. automethod:: HiVis.HiVis.HiVis.agg_stardist
.. automethod:: HiVis.HiVis.HiVis.agg_from_annotations

Analyses
**********************************
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.qc
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.dge
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.pseudobulk
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.noise_mean_curve
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.cor
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.score_genes
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.smooth
.. automethod:: HiVis.HiVis_analysis.AnalysisVisium.compute_distances

Plots
*************************************

Methods are called by HiVis.plot.

.. automethod:: HiVis.HiVis_plot.PlotVisium.spatial
.. automethod:: HiVis.HiVis_plot.PlotVisium.hist
.. automethod:: HiVis.HiVis_plot.PlotVisium.cor
.. automethod:: HiVis.HiVis_plot.PlotVisium.noise_mean_curve
.. automethod:: HiVis.HiVis_plot.PlotVisium.save

Export
**********************************
.. automethod:: HiVis.HiVis.HiVis.save
.. automethod:: HiVis.HiVis.HiVis.export_h5
.. automethod:: HiVis.HiVis.HiVis.export_images


Attributes
************************************
.. autoattribute:: HiVis.HiVis.HiVis.adata
.. rubric:: AnnData that stores the data and metadata of the bins
.. autoattribute:: HiVis.HiVis.HiVis.agg
.. rubric:: Dictionary that stores all Aggregation objects
.. autoattribute:: HiVis.HiVis.HiVis.properties
.. rubric:: Dictionary, can store any information
.. autoattribute:: HiVis.HiVis.HiVis.shape
.. rubric:: Returns adata.shape
.. autoattribute:: HiVis.HiVis.HiVis.columns
.. rubric:: Returns adata.columns
.. autoattribute:: HiVis.HiVis.HiVis.name
.. rubric:: Name of the object
.. autoattribute:: HiVis.HiVis.HiVis.path_output
.. rubric:: Path where all plots and files will be saved
.. autoattribute:: HiVis.HiVis.HiVis.image_fullres
.. rubric:: Fullres image. Also available highres and lowres.
.. autoattribute:: HiVis.HiVis.HiVis.fluorescence
.. rubric:: Dictionary, stores fluorescence information (channels and display colors)
.. autoattribute:: HiVis.HiVis.HiVis.json
.. rubric:: Dictionary, stores spatial scale factors


Aggregation
=====================================================

Stores the data and images of the aggregated VisiumHD bins, enables plotting via Aggregation.plot.

General methods
****************
.. automethod:: HiVis.Aggregation.Aggregation.copy
.. automethod:: HiVis.Aggregation.Aggregation.rename
.. automethod:: HiVis.Aggregation.Aggregation.merge
.. automethod:: HiVis.Aggregation.Aggregation.sync
.. automethod:: HiVis.Aggregation.Aggregation.combine
.. automethod:: HiVis.Aggregation.Aggregation.import_geometry
.. automethod:: HiVis.Aggregation.Aggregation.head
.. automethod:: HiVis.Aggregation.Aggregation.update

Analyses
**********************************
.. automethod:: HiVis.HiVis_analysis.AnalysisAgg.dge
.. automethod:: HiVis.HiVis_analysis.AnalysisAgg.pseudobulk
.. automethod:: HiVis.HiVis_analysis.AnalysisAgg.noise_mean_curve
.. automethod:: HiVis.HiVis_analysis.AnalysisAgg.cor
.. automethod:: HiVis.HiVis_analysis.AnalysisAgg.score_genes
.. automethod:: HiVis.HiVis_analysis.AnalysisAgg.smooth
.. automethod:: HiVis.HiVis_analysis.AnalysisAgg.compute_distances

Plots
************************************

Methods are called by Aggregation.plot.

.. automethod:: HiVis.HiVis_plot.PlotAgg.spatial
.. automethod:: HiVis.HiVis_plot.PlotAgg.cells
.. automethod:: HiVis.HiVis_plot.PlotAgg.umap
.. automethod:: HiVis.HiVis_plot.PlotAgg.hist
.. automethod:: HiVis.HiVis_plot.PlotAgg.cor
.. automethod:: HiVis.HiVis_plot.PlotAgg.noise_mean_curve
.. automethod:: HiVis.HiVis_plot.PlotAgg.save

Export
**********************************
.. automethod:: HiVis.Aggregation.Aggregation.export_h5
.. automethod:: HiVis.Aggregation.Aggregation.export_to_matlab

Attributes
************************************
.. autoattribute:: HiVis.Aggregation.Aggregation.adata
.. rubric:: AnnData that stores the data and metadata of the objects
.. autoattribute:: HiVis.Aggregation.Aggregation.viz
.. rubric:: HiVis parent object that is linked to this instance
.. autoattribute:: HiVis.Aggregation.Aggregation.shape
.. rubric:: Returns adata.shape
.. autoattribute:: HiVis.Aggregation.Aggregation.columns
.. rubric:: Returns adata.columns
.. autoattribute:: HiVis.Aggregation.Aggregation.name
.. rubric:: Name of the object
.. autoattribute:: HiVis.Aggregation.Aggregation.path_output
.. rubric:: Path where all plots and files will be saved
