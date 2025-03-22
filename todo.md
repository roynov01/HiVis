

# Todo
* Remove double-assignment in add_agg_stardist()

* subset - crop adata.uns["spatial"]:
	* adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
	* adata.uns["spatial"] = {"your_sample_name": {"images": {"hires": np.array(image)},
                                             "scalefactors": {"tissue_hires_scalef": 1.0},
                                             "metadata": {"source": "your_data"}}}
* Remove double-assignment in add_agg_stardist()
* fix plot_umap multiple - gives error, and also constant warnings on color m6.agg["SC"].plot.umap("leiden",size=20,cmap="tab20")
* Finish Groovy scripts - one for Stardist, one for Cellpose. create python cellpose_add_agg
* Write Qupath tutorial
	* pixel classifier + script
	* Annotations + geojson
	* Stardist/cellpose (+installation)
* Test adding of Aggs
* Update links in:
	* readme
	* notebooks
* Write tests	
* remove garbage files from github
* add citations to RTD and README
* convert noise_mean to plot
* add cor() to HiVis


## future additions
* Add smooth() to HiVis
* Add module_score() 
* Add HiVis objects


