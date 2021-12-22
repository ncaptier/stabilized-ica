# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#from packaging.version import parse

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath("sphinxext"))

#from github_link import make_linkcode_resolve
#import sphinx_gallery

# -- Project information -----------------------------------------------------

project = 'stabilized-ica'
copyright = 'Copyright 2021, Nicolas Captier (built with a theme provided by the scikit-learn developers (BSD License))'
author = 'Nicolas Captier'

# The full version, including alpha/beta/rc tags
#import sica

# parsed_version = parse(sica.__version__)
# version = ".".join(parsed_version.base_version.split(".")[:2])
# # The full version, including alpha/beta/rc tags.
# # Removes post from release name
# if parsed_version.is_postrelease:
#     release = parsed_version.base_version
# else:
#     release = sica.__version__
release = '1.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc' ,
              'sphinx.ext.autosummary' ,
              'sphinx.ext.coverage',
              'numpydoc',
              'sphinx.ext.linkcode',
              'sphinx.ext.autosectionlabel',
              'nbsphinx',
              'sphinx_gallery.load_style',
              'add_toctree_functions']



#'sphinx_gallery.gen_gallery'
#'sphinx.ext.napoleon' (similar to numpydoc)
    
autosectionlabel_prefix_document = True

numpydoc_class_members_toctree = False

# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
    

autodoc_default_options = {"members": True, "inherited-members": True}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = ".rst"

# The main toctree document.
main_doc = "contents"
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_templates" , "themes" , "includes"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ["themes"]

#html_theme = 'nature'
html_theme = "scikit-learn-modern"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"google_analytics": True, "mathjax_path": mathjax_path}

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "stabilized-ica"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "logos/stabilized-ica-logo_2.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, the reST sources are included in the HTML build as _sources/name.
html_copy_source = True

# The following is used by sphinx.ext.linkcode to provide links to github
# linkcode_resolve = make_linkcode_resolve(
#     package = "sica",
#     url_fmt = "https://github.com/ncaptier/stabilized-ica/"
#               "stabilized-ica/blob/master/"
#               "{package}/{path}#L{lineno}"
#  
# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = "stabilized-icadoc"

import sica
                                     
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        import os
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(sica.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno , lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = 'sica/%s#L%d-L%d' % find_source()
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    return "https://github.com/ncaptier/stabilized-ica/blob/master/%s" % filename

