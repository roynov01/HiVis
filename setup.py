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
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scanpy",
        "squidpy",
        "anndata",
        "adjustText",
        "pyarrow",
        "pillow",
        "tqdm",
        "shapely",
        "geopandas",
        "dill",
        "tifffile",
        "statsmodels"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)