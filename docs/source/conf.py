import os
import sys
import datetime

sys.path.insert(0, os.path.abspath("../.."))

import exm

# import sphinx.domain.python
import sphinx.ext.autosummary
from docutils import nodes
from importlib import import_module
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.ext.autosummary import Autosummary
from inspect import getmembers, isclass, isfunction


# -- Project information

# project = 'ExSeq-Toolbox'
project = ""

# release = '0.1'
release = ""
# version = '0.1.0'
version = ""

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
]

# katex for rendering LaTeX
katex_prerenderer = True

# suffixes of source file names
source_suffix = ".rst"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
intersphinx_disabled_domains = ["std"]

html_theme_options = {
    "display_version": False,
    "logo_only": True,
    "collapse_navigation": True,
}

html_logo = "../exseq-logo.png"

templates_path = ["_templates"]

# name of syntax highlighting style to use
pygments_style = "sphinx"

# -- Options for HTML output

# html_theme = 'sphinx_rtd_theme'
html_theme = "furo"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "navigation_with_keys": True,
}

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Type hints configs ------------------------------------------------------

autodoc_inherit_docstrings = True
autoclass_content = "both"
autodoc_typehints = "description"
napoleon_attr_annotations = True

# -- Options for EPUB output
epub_show_urls = "footnote"

# replaces pending_xref node with desc_type for type annotations
# sphinx.domains.python.type_to_xref = lambda t, e=None: addnodes.desc_type(
#    "", nodes.Text(t)
# )

# -- Autosummary patch to get list of a classes, funcs automatically ----------


class BetterAutosummary(Autosummary):
    """Autosummary with autolisting for modules.
    By default it tries to import all public names (__all__),
    otherwise import all classes and/or functions in a module.
    Options:
    - :autolist: option to get list of classes and functions from currentmodule.
    - :autolist-classes: option to get list of classes from currentmodule.
    - :autolist-functions: option to get list of functions from currentmodule.
    Example Usage:
    .. currentmodule:: ignite.metrics
    .. autosummary::
        :nosignatures:
        :autolist:
    """

    # Add new option
    _option_spec = Autosummary.option_spec.copy()
    _option_spec.update(
        {
            "autolist": directives.unchanged,
            "autolist-classes": directives.unchanged,
            "autolist-functions": directives.unchanged,
        }
    )
    option_spec = _option_spec

    def run(self):
        for auto in ("autolist", "autolist-classes", "autolist-functions"):
            if auto in self.options:
                # Get current module name
                module_name = self.env.ref_context.get("py:module")
                # Import module
                module = import_module(module_name)

                # Get public names (if possible)
                try:
                    names = getattr(module, "__all__")
                except AttributeError:
                    # Get classes defined in the module
                    cls_names = [
                        name[0]
                        for name in getmembers(module, isclass)
                        if name[-1].__module__ == module_name
                        and not (name[0].startswith("_"))
                    ]
                    # Get functions defined in the module
                    fn_names = [
                        name[0]
                        for name in getmembers(module, isfunction)
                        if (name[-1].__module__ == module_name)
                        and not (name[0].startswith("_"))
                    ]
                    names = cls_names + fn_names
                    # It may happen that module doesn't have any defined class or func
                    if not names:
                        names = [name[0] for name in getmembers(module)]

                # Filter out members w/o doc strings
                names = [
                    name for name in names if getattr(module, name).__doc__ is not None
                ]

                if auto == "autolist":
                    # Get list of all classes and functions inside module
                    names = [
                        name
                        for name in names
                        if (
                            isclass(getattr(module, name))
                            or isfunction(getattr(module, name))
                        )
                    ]
                else:
                    if auto == "autolist-classes":
                        # Get only classes
                        check = isclass
                    elif auto == "autolist-functions":
                        # Get only functions
                        check = isfunction
                    else:
                        raise NotImplementedError

                    names = [name for name in names if check(getattr(module, name))]

                # Update content
                self.content = StringList(names)
        return super().run()


# Patch original Autosummary
sphinx.ext.autosummary.Autosummary = BetterAutosummary

# --- autosummary config -----------------------------------------------------
autosummary_generate = False

# -- Type hints configs ------------------------------------------------------

autodoc_inherit_docstrings = True
autoclass_content = "both"
autodoc_typehints = "description"
napoleon_attr_annotations = True
