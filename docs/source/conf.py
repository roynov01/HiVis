
project = 'HiVis'
copyright = '2025, Roy Novoselsky'
author = 'Roy Novoselsky'
release = os.environ.get('READTHEDOCS_VERSION_NAME', 'latest')

import os
import sys
#sys.path.insert(0, os.path.abspath('../../HiVis'))
sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',  
    'myst_parser',
    'sphinx.ext.linkcode',  
]

html_theme = 'sphinx_rtd_theme'

html_static_path = []

myst_enable_extensions  = [
    'dollarmath',
    'amsmath', 
]
autosummary_generate = True
html_show_sourcelink = True
autodoc_member_order = 'bysource' 
add_module_names = False
html_domain_indices = False
html_use_modindex = False

