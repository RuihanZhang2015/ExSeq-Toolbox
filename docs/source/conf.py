from sphinx.ext.autosummary import Autosummary
import sphinx.ext.autosummary
import sys
import os
import h5py
import pickle
import tempfile
import numpy as np
import SimpleITK as sitk
from scipy import signal
import matplotlib.pyplot as plt
import multiprocessing
import queue

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information

project = 'ExM Toolbox'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

#html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'

# -- Options for EPUB output
epub_show_urls = 'footnote'
