
project = 'HiVis'
copyright = '2025, Roy Novoselsky'
author = 'Roy Novoselsky'
release = '0.1.0'

import os
import sys
sys.path.insert(0, os.path.abspath('../../HiVis'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    "myst_parser" 
]



html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

#autodoc_typehints = "none"
#autodoc_docstring_signature = False

myst_enable_extensions  = [
    "dollarmath",
    "amsmath", 
]

autodoc_member_order = 'bysource' 
