# npy2bdv
 A library for writing/reading HDF5/XML datasets of 
 Fiji BigDataViewer/BigStitcher format as numpy arrays. 
 
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 
 ## Installation
 Run `pip install npy2bdv` in the command line. Use `import npy2bdv` in the code.

 ### From source (development)
 With [mamba](https://github.com/conda-forge/miniforge) (or conda):
 ```
 git clone https://github.com/nvladimus/npy2bdv.git
 cd npy2bdv
 mamba env create -f environment.yml
 mamba activate npy2bdv
 ```
 Or with pip only: `pip install -e .[dev]`.

 Run the tests from the repo root: `python -m pytest test`.

 ### Releasing
 Bump `version` in `pyproject.toml`, then publish a GitHub release tagged `v<version>`.
 The [`publish`](.github/workflows/publish.yml) workflow runs the tests and uploads the package to PyPI.
 The docs are rebuilt and deployed to GitHub Pages on every push to `master`
 ([`docs`](.github/workflows/docs.yml) workflow). To build them locally:
 `pip install -e .[docs]`, then `sphinx-build -b html docs/source docs/build/html`.
 
 ## Documentation
**https://nvladimus.github.io/npy2bdv/**: a user guide with code examples, and a detailed API reference.

Larger, timed examples are in the Jupyter [`notebook`](/docs/examples/examples_h5writing.ipynb).

 ## Supported H5/XML writing options
 * pixel types `uint16` (default), `uint8`, `float32`: `BdvWriter(..., dtype='float32')`.
   Types other than `uint16` need an up-to-date Fiji (BigDataViewer-core >= 10.4.6).
 * compression methods `None`, `gzip`, `lzf` (`None` by default).
 * downsampling options: 
    - any number of mipmap levels
    - computed via averaging, compatible with BigDataViewer/BigStitcher convention.
 * user-defined block sizes for H5 storage (default `4,256,256`)
 * any number of time points, illuminations, channels, tiles, angles.
 * arbitrary affine transformation for each individual view (e.g. translation, rotation, shear).
 * arbitrary voxel calibration for each view, to account for spatial anisotropy.
 * individual views can differ in dimensions, voxel size, voxel units, exposure time, and exposure units.
 * missing views are labeled in XML automatically.
 * support of additional meta-information:
    - camera properties: `name`, `exposureTime`, `exposureUnits`
    - `microscope` (name and version), `user`
 * writing virtual stacks of arbitrary size, by plane or sub-stack. Handy when your stack is larger than your RAM.
    - virtual stacks can be written with multiple subsampling levels and compression.
    
 ## New features
 See [Recent changes](CHANGELOG.md)
 
 ## Writing speed
 Currently speeds between 150 - 320 MB/s were achieved with confidence. Speeds of up to 2300 MB/s reported in earlier versions are likely due to bugs in testing methods in notebook [`examples_h5writing.ipynb`](/docs/examples/examples_h5writing.ipynb). More testing is underway.

 ## Acknowledgements
 This code was inspired by [Talley Lambert's](https://github.com/tlambert03/imarispy) code 
 and improved by generous input from Adam Glaser, [VolkerH](https://github.com/VolkerH), Doug Shepherd, 
 [Peter H](https://github.com/abred), [Mike Taormina](https://github.com/miketaormina) [Pradeep Rajasekhar](https://github.com/pr4deepr) and [aaronkanzer](https://github.com/aaronkanzer).
 
 To report issues or bugs please use the [issues](https://github.com/nvladimus/npy2bdv/issues) tool.
 
 ## Citation
 You can cite it like so
 
 [![DOI](https://zenodo.org/badge/203410946.svg)](https://zenodo.org/badge/latestdoi/203410946)


It will give me some dopamine and motivation to keep improving it :heart: 
