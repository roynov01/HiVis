# Changelog

Changes to HiVis since 1.0.0, newest first. Version numbers are the ones in `setup.py`, `pyproject.toml` and `HiVis.__version__`.

## Unreleased

Fixes and a few new options after 1.1.0. Existing scripts keep working with the same results, except for the points under "Changed". The tests in `tests/` compare each change with 1.1.0.

### Changed
- `plot.cor` of one gene colors the points by -log10(q) instead of -ln(q), as its legend title "qval_log10" says. The points' colors are the same. The legend now shows log10 values.
- A dict passed as `scalebar` is no longer modified. Reusing one dict for several plots used to keep the first plot's color, e.g. white from a fluorescence image on a later plot with `image=False`. Each plot now gets its own default color.

### Added
- A matplotlib `Colormap` (e.g. `ListedColormap`) as `cmap` in `plot.spatial` and `plot.cells`, for numeric and categorical values. For numeric values it raised `UnboundLocalError`, and other cmap types now raise `ValueError` instead.
- `repel` in `plot.cor` (HiVis and Aggregation), to move one gene's labels apart. `legend_title` now also titles that plot's legend.
- `scalebar={"length_microns": ...}`, the key the docstrings documented, as another name for `length`.
- `agg_cells`, `agg_from_annotations`, `import_geometry` and the `Aggregation` constructor take the geojson as a `pathlib.Path` or a `GeoDataFrame`.

### Fixed
- pandas 3: `plot.hist` of a string or categorical column raised `TypeError`, and `other_utils.inspect_df` left out the grouping column.
- Subsetting an Aggregation by cell labels or gene names (`agg[labels, :]`, `agg[:, genes]`), or by masks on both axes, failed with anndata 0.13, and with older anndata when the aggregation had layers. Subsets that worked are unchanged, except with anndata 0.13 and positions on both axes (e.g. `agg[[0], [1, 2]]`): X came out dense, so `sub["gene"]` failed, and now stays sparse as with older anndata.
- Docstrings: the `scalebar` keys of `plot.spatial` and `plot.cells` (`length`, not `length_microns`; `text_offset` defaults to `None`). Plot coordinates are pixels of the image resolution used, so plots drawn on one ax need the same `img_resolution`, `xlim` and `ylim`. `dge2` is `dge` as it was before 2025-08-28.

### Performance
- `dge` with `fisher_exact`: about 20x faster, 35 s instead of about 12 minutes for 30,000 cells x 19,000 genes. The results are identical. Each gene's total counts come from one column sum where it gives exactly the same number. Other genes are summed one by one as before.

## 1.1.0 (2026-10-28)

Bug fixes, speedups and a few new options. Existing scripts keep working with the same results, except for the points under "Changed". Pickles saved by older versions load as before. The tests in `tests/` compare each change with 1.0.6. The default plots checked (`plot.spatial`, `plot.cells`, `plot_MA`, `qc`) are pixel-identical to 1.0.6.

### Changed
- A new HiVis object (`new_visiumHD`, or `HiVis(..., plot_qc=True)`) saves its QC plot as `<name>_QC.png` only, not also as PDF and SVG. `analysis.qc(save=True)` still writes all three.
- An Aggregation no longer creates an empty folder in `path_output` when it is made (also by `subset`, `crop` and `copy`). Its folder, at the same path as before, is created when a plot or data is saved into it. Scripts that write their own files into `agg.path_output` should create it first: `os.makedirs(agg.path_output, exist_ok=True)`.
- Every save (`plot.save`, `export_h5`, `export_to_matlab`, `save`) creates the folder it writes into, if it doesn't exist.
- `subset()` of an object whose name ends with `_subset` keeps that name. It used to give the new object an empty name.
- `Aggregation.copy()` stays linked to the same HiVis object. It used to deep-copy the whole HiVis object, images included, twice. `sync()` on a copy now writes to the parent HiVis.
- `agg_from_annotations` and `agg_cells` no longer modify the `obs2agg` list or dict that is passed in. Reusing one across calls no longer adds an earlier annotation's column to later aggregations.
- `update()` also reloads `HiVis_utils`, `Aggregation_utils` and `other_utils`.

### Added
- `vmin` and `vmax` in `plot.spatial` (HiVis and Aggregation) and `plot.cells`: a fixed color range for numeric values, for the objects and the colorbar.
- `layer` in `analysis.score_genes`.
- `object_type` in `agg_cells`, e.g. `"detection"` for cells exported from QuPath as detections. The geojson is checked before the aggregation, and the error lists the object types it contains.
- `genes`, `color` and `color_genes` in `HiVis_plot.plot_MA`.

### Fixed
- Aggregation plots (`plot.spatial`, `plot.cells`): objects were shifted against the image when `xlim` or `ylim` started below 0 (past the image's left or top edge).
- `save` splits large writes into 64 MB chunks. A single multi-GB write could fail on a network drive with `OSError: [Errno 22] Invalid argument`. The saved file is unchanged.
- `HiVis_utils.matnorm(DataFrame, axis="row")` failed under pandas 3.
- `dge` with `fisher_exact` failed on a dense matrix.
- Warnings: `invalid value encountered in cast` when plotting values with NaN, `is_categorical_dtype` deprecation (pandas 3), and `invalid escape sequence` on import.

### Performance
- `dge` with `wilcox` or `t_test`: about 25x faster for 20,000 cells x 19,000 genes. With `fisher_exact`: about 2x faster. The results are identical.
- Aggregation plots no longer copy the whole AnnData, with its images, on every call. `agg.adata_cropped` is a view of `agg.adata`, as it is for the bins. Pickles saved after plotting an aggregation are smaller.

## 1.0.6 (2026-10-25)

### Fixed
- `AnalysisAgg.compute_distances` failed under pandas 3, where the WKT geometry column has the `str` dtype. Objects without geometry get NaN instead of crashing, and targets without geometry are skipped (also in `AnalysisVisium.compute_distances`). Results are otherwise identical to 1.0.5.
- `save` writes to `<path>.tmp` and then replaces the file, so a save that fails midway keeps the previous pickle. A save needs free disk space for both versions.
- `plot.cells(axis_labels=False)` showed "x" and "y" axis labels with geopandas 1.2 or newer.

### Added
- Synthetic tests in `tests/`.

## 1.0.5 (2026-10-24)

### Added
- Space Ranger 3 output layout: the hires and lowres images are also found in `outs/spatial`. New optional `path_image_highres` and `path_image_lowres` arguments in `new_visiumHD`.
- Reading ImageJ TIFFs larger than 4 GB, as written by QuPath and Fiji (`HiVis_utils.read_tiff`).

### Changed
- `new_visiumHD` no longer writes into the input folders. With `crop_images=True`, the cropped images and tissue positions are saved in `path_output` (`<name>_fullres_cropped.tif`, `<name>_highres_cropped.tif`, `<name>_lowres_cropped.tif`, `<name>_tissue_positions_cropped.csv`). Existing cropped TIFFs are not overwritten.
- The data are read with `sc.read_10x_h5` and `tissue_positions.parquet` instead of the deprecated `sc.read_visium`, about 2x faster. The scanpy requirement is relaxed from `==1.11.0` to `>=1.11.0`.

### Fixed
- `new_visiumHD(properties=None)` crashed.
- `new_visiumHD` accepts `pathlib.Path` and paths that end with a slash.

## 1.0.4 (2026-09-06)

### Fixed
- `get()` returns NumPy arrays with `to_numpy()`, for newer pandas versions.
- Printing an Aggregation with a `None` layer.
- `plt.cm.get_cmap`, removed in matplotlib 3.9, replaced by `plt.get_cmap`.

## 1.0.3 (2026-04-22)

- Documentation only: citation, manuscript link and links between the tutorials.

## 1.0.2 (2026-04-01)

### Fixed
- `PlotAgg.umap` coloring by genes and other features. New `vmax` argument.

### Changed
- Updated tutorials, and video tutorial links in the README.

## 1.0.1 (2026-03-28)

### Fixed
- `PlotAgg.spatial` with an ordered categorical column.
- Importing metadata with columns that already exist in the data: the duplicates are dropped.

## 1.0.0 (2026-03-26)

- First stable release.
- `rasterize` argument in `plot.spatial`, `plot.cells` and `plot.umap` (on by default in `umap`), for smaller vector figures.
