from setuptools import setup, find_packages

setup(
    name="HiVis", 
    version="0.1.0",
    author="Roy Novoselsky",
    author_email="roy.novoselsky@weizmann.ac.il",
    description="A user friendly tool for working with VisiumHD data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/roynov01/HiVis",  
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1.3",
        "scipy>=1.15.2",
        "matplotlib>=3.10.1",
        "pandas>=2.2.3",
        "seaborn>=0.13.2",
        "scanpy==1.11.0", # read_visium will be deprecated, if updating scanpy, change to suidpy.read_visium
        "anndata>=0.11.3",
        "adjustText", # version 1.3.0 is broken
        "pyarrow>=19.0.1",
        "pillow>=11.1.0",
        "tqdm>=4.67.1",
        "shapely>=2.0.7",
        "geopandas>=1.0.1",
        "dill>=0.3.9",
        "tifffile>=2025.3.13",
        "statsmodels>=0.14.4"
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.10",
)



    "Intended Audience :: Science/Research",
	"Topic :: Scientific/Engineering :: Bio-Informatics",
	"Programming Language :: Python :: 3",
	"Programming Language :: Python :: 3.10",
	"License :: OSI Approved :: MIT License",
	"Operating System :: OS Independent",