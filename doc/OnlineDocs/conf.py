#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# pyomo documentation build configuration file, created by
# sphinx-quickstart on Mon Dec 12 16:08:36 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# assumes pyutilib source is next to the pyomo source directory
sys.path.insert(0, os.path.abspath('../../../pyutilib'))
# top-level pyomo source directory
sys.path.insert(0, os.path.abspath('../..'))
# our sphinx extensions
sys.path.insert(0, os.path.abspath('ext'))

# -- Rebuild SPY files ----------------------------------------------------
sys.path.insert(0, os.path.abspath('src'))
try:
    print("Regenerating SPY files...")
    from strip_examples import generate_spy_files

    generate_spy_files(os.path.abspath('src'))
    generate_spy_files(
        os.path.abspath(os.path.join('explanation', 'experimental', 'kernel'))
    )
finally:
    sys.path.pop(0)

# -- Options for intersphinx ---------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '1.8'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx_copybutton',
    # Our version of 'autoenum', designed to work with autosummary.
    # This adds 'sphinx.ext.autosummary', and 'sphinx.ext.autodoc':
    'pyomo_autosummary_autoenum',
    'pyomo_tocref',
]

viewcode_follow_imported_members = True
# napoleon_include_private_with_doc = True

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Pyomo'
copyright = u'2008-2024, Sandia National Laboratories'
author = u'Pyomo Development Team'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import pyomo.version

version = pyomo.version.__version__
# The full version, including alpha/beta/rc tags.
release = pyomo.version.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# These patterns also effect to html_static_path and html_extra_path
# Notes:
#  - _build : this is the Sphinx build (output) dir
#
#  - api/*.tests.* : this matches autosummary RST files generated for
#    test modules.  Note that the _templates/recursive-modules.rst
#    should prevent these file from being generated, so this is not
#    strictly necessary, but including it makes Sphinx throw warnings if
#    the filter in the template ever "breaks"
#
#  - **/tests/** : this matches source files in any tests directory
#    [JDS: I *believe* this is necessary, but am not 100% certain]
#
#  - 'Thumbs.db', '.DS_Store' : these have been included from the
#    beginning.  Unclear if they are still necessary
exclude_patterns = ['_build', 'api/*.tests.*', '**/tests/**', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# If true, doctest flags (comments looking like # doctest: FLAG, ...) at
# the ends of lines and <BLANKLINE> markers are removed for all code
# blocks showing interactive Python sessions (i.e. doctests)
trim_doctest_flags = True

# If true, figures, tables and code-blocks are automatically numbered if
# they have a caption.
numfig = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

html_theme = 'sphinx_rtd_theme'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {'navigation_depth': 6, 'titles_only': True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['theme_overrides.css']

html_favicon = "../logos/pyomo/favicon.ico"


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyomo'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    'preamble': r'''
\usepackage{enumitem}
\setlistdepth{99}
\DeclareUnicodeCharacter{2227}{$\wedge$}
\DeclareUnicodeCharacter{2228}{$\vee$}
\DeclareUnicodeCharacter{22BB}{$\veebar$}
''',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    # necessary for unicode charactacters in pdf output
    'inputenc': '',
    'utf8extra': '',
    # remove blank pages (e.g., between chapters)
    'classoptions': ',openany,oneside',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'pyomo.tex', 'Pyomo Documentation', author, 'manual'),
    ('code', 'pyomo_reference.tex', 'Pyomo Code Reference', author, 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = '../logos/pyomo/PyomoNewBlue.jpg'

# Disable the domain indices (i.e., the module index) for LaTeX targets:
# because we are splitting the socumentation, the module index in the
# main document would be completely broken, and having one in the
# reference document seems redundant (JDS: and I haven't figured out how
# to have it in only one of the documents)
latex_domain_indices = False

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = []


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = []

# autodoc_member_order = 'bysource'
autodoc_member_order = 'groupwise'

autosummary_generate = True
autosummary_ignore_module_all = True

# -- Check which conditional dependencies are available ------------------
# Used for skipping certain doctests
from sphinx.ext.doctest import doctest

doctest_default_flags = (
    doctest.ELLIPSIS
    + doctest.NORMALIZE_WHITESPACE
    + doctest.IGNORE_EXCEPTION_DETAIL
    + doctest.DONT_ACCEPT_TRUE_FOR_1
)


class IgnoreResultOutputChecker(doctest.OutputChecker):
    IGNORE_RESULT = doctest.register_optionflag('IGNORE_RESULT')

    def check_output(self, want, got, optionflags):
        if optionflags & self.IGNORE_RESULT:
            return True
        return super().check_output(want, got, optionflags)


doctest.OutputChecker = IgnoreResultOutputChecker

doctest_global_setup = '''
import os, platform, sys
on_github_actions = bool(os.environ.get('GITHUB_ACTIONS', ''))
system_info = (
    sys.platform,
    platform.machine(),
    platform.python_implementation()
)

# Mark that we are testing code (in this case, testing the documentation)
from pyomo.common.flags import in_testing_environment
in_testing_environment(True)

# We need multiprocessing because some doctests must be skipped if the
# start method is not "fork"
import multiprocessing

# (register plugins, make environ available to tests)
import pyomo.environ as pyo

from pyomo.common.dependencies import (
    attempt_import, numpy_available, scipy_available, pandas_available,
    yaml_available, networkx_available, matplotlib_available,
    pympler_available, dill_available, pint_available,
    numpy as np,
)
from pyomo.contrib.parmest.parmest import parmest_available

# Ensure that the matplotlib import has been resolved (and the backend changed)
bool(matplotlib_available)

# Not using SolverFactory to check solver availability because
# as of June 2020 there is no way to suppress warnings when 
# solvers are not available
import pyomo.opt as _opt
ipopt_available = bool(_opt.check_available_solvers('ipopt'))
sipopt_available = bool(_opt.check_available_solvers('ipopt_sens'))
k_aug_available = bool(_opt.check_available_solvers('k_aug'))
dot_sens_available = bool(_opt.check_available_solvers('dot_sens'))
baron_available = bool(_opt.check_available_solvers('baron'))
glpk_available = bool(_opt.check_available_solvers('glpk'))
gurobipy_available = bool(_opt.check_available_solvers('gurobi_direct'))

baron = _opt.SolverFactory('baron')

if numpy_available:
    # Recent changes on GHA seem to have dropped the default precision
    # from 8 to 4; restore the default.
    np.set_printoptions(precision=8)

if numpy_available and scipy_available:
    import pyomo.contrib.pynumero.asl as _asl
    asl_available = _asl.AmplInterface.available()
    import pyomo.contrib.pynumero.linalg.ma27 as _ma27
    ma27_available = _ma27.MA27Interface.available()
    from pyomo.contrib.pynumero.linalg.mumps_interface import mumps_available
else:
    asl_available = False
    ma27_available = False
    mumps_available = False

# Prevent any Pyomo logs from propagating up to the doctest logger
import logging
logging.getLogger('pyomo').propagate = False
'''
