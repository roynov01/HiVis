import os
import sys
import inspect
import importlib

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None

    module = info.get('module')
    fullname = info.get('fullname')
    if not module:
        return None

    try:
        obj = importlib.import_module(module)
        for part in fullname.split('.'):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
    except Exception:
        return None

    # repo root as seen from the docs folder (conf.py is in docs/source)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    rel_path = os.path.relpath(fn, repo_root).replace(os.path.sep, '/')

    start = start_line
    end = start_line + len(source_lines) - 1

    return f"https://github.com/roynov01/HiVis/blob/main/{rel_path}#L{start}-L{end}"
    
project = 'HiVis'
copyright = '2025, Roy Novoselsky'
author = 'Roy Novoselsky'
release = os.environ.get('READTHEDOCS_VERSION_NAME', 'latest')

# Mock heavy external dependencies so Sphinx can import HiVis to inspect signatures
autodoc_mock_imports = [
    "numpy", "pandas", "geopandas", "shapely", "anndata", "scanpy",
    "matplotlib", "PIL", "Pillow", "tqdm", "sklearn", "statsmodels",
    "scipy", "pyarrow", "dill","skimage"
]

#sys.path.insert(0, os.path.abspath('../../HiVis'))
sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
#    'sphinx.ext.viewcode',  
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

