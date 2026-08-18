"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import datetime

import contrailopt

project = "contrailopt"
copyright = f"{datetime.datetime.now().year}, Contrails.org"
author = "Contrails.org"
version = contrailopt.__version__
release = contrailopt.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]

nb_execution_mode = "off"
nb_merge_streams = True

myst_enable_extensions = ["dollarmath"]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "npt.NDArray[FLOAT_DTYPE]": "numpy.ndarray",
    "npt.NDArray[np.bool_]": "numpy.ndarray",
    "npt.NDArray[np.datetime64]": "numpy.ndarray",
    "npt.NDArray[np.float64]": "numpy.ndarray",
    "npt.NDArray[np.floating]": "numpy.ndarray",
    "npt.NDArray[np.int64]": "numpy.ndarray",
    "pd.DataFrame": "pandas.DataFrame",
    "pd.Timestamp": "pandas.Timestamp",
    "pd.Timedelta": "pandas.Timedelta",
    "xr.DataArray": "xarray.DataArray",
    "xr.Dataset": "xarray.Dataset",
    "FuncAnimation": "~matplotlib.animation.FuncAnimation",
    "Flight": "~pycontrails.Flight",
    "MetDataArray": "~pycontrails.MetDataArray",
    "MetDataset": "~pycontrails.MetDataset",
    "ps_aircraft_params.PSAircraftEngineParams": (
        "~pycontrails.models.ps_model.PSAircraftEngineParams"
    ),
}

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pycontrails": ("https://py.contrails.org/", None),
}

html_theme = "furo"
html_title = f"{project} v{release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "https://py.contrails.org/_static/favicon.svg"
html_logo = "https://py.contrails.org/_static/img/icon-light.svg"
html_last_updated_fmt = "%Y-%m-%d"
