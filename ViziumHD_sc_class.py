# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:57:45 2024

@author: royno
"""

class SingleCell:
     def __init__(self, vizium_instance):
         self.viz = vizium_instance
         self.adata = None
         self.path_output = self.viz.path_output + "/single_cell"

     def __init_img(self):
         pass
     
     def crop(self):
         pass
     
     def plot_spatial(self, ):
         pass
     
     def plot_cells(self, column, celltypes=None, image=None, title=None, cmap=None, 
                   legend=False, alpha=True, figsize=(8, 8), 
                   xlim=None, ylim=None):
         # requires cells annotations geopandas
         pass
     
     def plot_umap(self, features, out_path=None,title=None,size=None,
              figsize=(8,8),file_type='png',legend_loc='right margin'):
         pass
     
     def plot_hist(self):
         pass
     
     
     def export_h5(self, path=None):
         if not path:
             path = f"{self.path_output}/{self.name}_viziumHD.h5ad"
         self.adata.write(path)
         return path
    
     @property
     def shape(self):
         return self.adata.shape
     
     def __str__(self):
         s = f"# single-cells # {self.viz.name} #\n"
         s += f"\tSize: {self.adata.shape[0]} x {self.adata.shape[1]}\n"
         s += '\nobs: '
         s += ', '.join(list(self.adata.obs.columns))
         s += '\n\nvar: '
         s += ', '.join(list(self.adata.var.columns))
         return s
     
     def __repr__(self):
         s = f"SingleCell[{self.viz.name}]"
         return s
     
     def __delitem__(self, key):
         '''deletes metadata'''
         if isinstance(key, str):
             if key in self.adata.obs:
                 del self.adata.obs[key]
             elif key in self.adata.var:
                 del self.adata.var[key]
             else:
                 raise KeyError(f"'{key}' not found in adata.obs")
             self.__init_img()
         else:
             raise TypeError(f"Key must be a string, not {type(key).__name__}")
     
     def head(self, n=5):
         return self.adata.obs.head(n) 
