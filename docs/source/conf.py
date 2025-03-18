
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
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']


myst_enable_extensions  = [
    "dollarmath",
    "amsmath", 
]

autodoc_member_order = 'bysource' 
