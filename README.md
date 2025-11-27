**HiVis** is associated with the following manuscript: [Subcellular mRNA localization patterns across tissues resolved with spatial transcriptomics](https://www.biorxiv.org/content/10.1101/2025.09.07.674688v1)

# HD Integrated Visium Interactive Suite (HiVis)
* **HiVis** is a user-friendly Python tool for analyzing  
  [10X VisiumHD](https://www.10xgenomics.com/products/spatial-gene-expression) data,  
  supporting both H&E and immunofluorescence experiments.
* **HiVis** is built on top of [AnnData](https://anndata.readthedocs.io/en/latest/), 
integrating seamlessly with other spatial transcriptomics tools such as
 [Scanpy](https://scanpy.readthedocs.io/en/stable/) and 
 [Squidpy](https://squidpy.readthedocs.io/en/stable/).  
* **HiVis** provides extensive data visualization with fully customizable plots. 
Plots are created using *matplotlib*, allowing stacking and further customization.  
* **HiVis** works with 
[QuPath](https://qupath.readthedocs.io/en/stable/)
to enable easy manual annotation, pixel classification, and single-cell segmentation using
 [Stardist](https://github.com/stardist/stardist/),  
 [Cellpose](https://github.com/MouseLand/cellpose) and [Instaseg](https://github.com/instanseg/instanseg).
 It links bins and single-cell objects, facilitating seamless information exchange across levels.  

![](docs/HiVis.png)

## Getting started
### Installation

To avoid dependency conflicts, we recommend the use of a 
dedicated 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
environment.  In a terminal run the command:

```bash
conda create -n HiVis python=3.12
conda activate HiVis
```

We recommend two options to then install HiVis in your virtual environment.

Use the package manager [pip](https://pypi.org/project/HiVis/) to install HiVis. 
In a terminal run the command:

```bash
pip install HiVis
```

Or clone the project's [Github repository](https://github.com/roynov01/HiVis) 
and install it manually with the following commands:

```bash
git clone git@github.com:roynov01/HiVis.git
cd HiVis
pip install .
```

To use Qupath features, such as manual annotations and pixel classifiers, download and install 
[QuPath](https://qupath.github.io/) (>= 0.5.1).

To perform cell segmentation in QuPath, download and install [Stardist](https://github.com/qupath/qupath-extension-stardist) 
and/or [Cellpose](https://github.com/BIOP/qupath-extension-cellpose) extensions for QuPath.


### Usage and Documentation
Please refer to the [demo notebooks](https://github.com/roynov01/HiVis/tree/main/tutorials).

For QuPath, please refer to the [tutorial](https://github.com/roynov01/HiVis/tree/main/QuPath).

Function docstrings are available on [ReadTheDocs](https://hivis.readthedocs.io/en/latest/).

## Contact
Bug report/feature request via the [GitHub issue tracker](https://github.com/roynov01/HiVis/issues).

## Citation
Novoselsky R*, Golani O*, Barkai T, Kedmi M, Goliand I, Fine M, Kent I, Nachmany I, Itzkovitz S. *Subcellular mRNA localization patterns across tissues resolved with spatial transcriptomics*. *BioRxiv*, 2025.

