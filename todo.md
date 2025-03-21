

# Todo
* subset - crop adata.uns["spatial"]:
	* adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
	* adata.uns["spatial"] = {"your_sample_name": {"images": {"hires": np.array(image)},
                                             "scalefactors": {"tissue_hires_scalef": 1.0},
                                             "metadata": {"source": "your_data"}}}
* Remove double-assignment in add_agg_stardist()
* Finish Groovy scripts - one for Stardist, one for Cellpose. create python cellpose_add_agg
* Write Qupath tutorial
	* pixel classifier + script
	* Annotations + geojson
	* Stardist/cellpose (+installation)
* Test adding of Aggs
* Update links in:
	* readme
	* notebooks
* Finish readTheDocs (sphinx)
* Write tests	
* remove garbage files from github
* upload to pypi
* add citations to RTD and README


## future additions
* Add smooth() to HiVis
* Add module_score() 
* Add HiVis objects


