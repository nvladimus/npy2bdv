import importlib.metadata

project = "npy2bdv"
author = "Nikita Vladimirov"
copyright = "Nikita Vladimirov"
release = importlib.metadata.version("npy2bdv")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

autoclass_content = "class"
default_role = "py:obj"  # `name` in docstrings links to the API entry
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

myst_enable_extensions = ["colon_fence"]
suppress_warnings = ["myst.header"]  # CHANGELOG.md starts with H2, not H1

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = []

html_theme = "furo"
html_title = f"npy2bdv {release}"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
