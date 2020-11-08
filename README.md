# npy2bdv
 A minimalistic library for writing image stacks (numpy arrays) into H5 and N5 datasets of 
 Fiji BigDataViewer/BigStitcher format. 
 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 
 ## Installation
 Optional: create a new conda environment
  ```
    conda create --name npy2bdv python=3.6
    conda activate npy2bdv
  ```
Install library [`z5py`](https://github.com/constantinpape/z5) and IPython notebook kernel 
(to run example notebooks):
 ```
    conda install -c conda-forge z5py
    conda install ipykernel
```
 To install latest **stable** version from PyPi, run `pip install npy2bdv`. 
 
 To install latest version from this repo, clone it and run `python setup.py install`
 
 ## Documentation
For a quick start, see examples in the Jupyter [`notebook`](/npy2bdv/docs/examples_h5writing.ipynb).

API [reference](https://nvladimus.github.io/npy2bdv/).

[Recent changes](CHANGELOG.md)
 - N5 support with file-system backend (i.e. the dataset is a tree of folders on a disk)
 - The H5/XML datasets can be edited in-place, e.g. by cropping selected views in `x,y,z`, 
 and appending affine transforms to XML. 
 


 ## Supported writing options
 * H5 compression methods `None`, `gzip`, `lzf`.
 * N5 compression methods `None`, `gzip`, `xz`.
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
 * writing virtual stacks of arbitrary size plane-by-plane (currently H5 only). Useful when your stack is larger than your RAM.
    - virtual stacks can be written with multiple subsampling levels and compression.
    
 ## Writing speed
Writing speeds up to 2300 MB/s can be achieved on a PC with SSD drive. 
The speed of writing for long time series (>100 stacks) is typically about 700-900 MB/s. 
This is in the range of full-speed camera acquisition 
of Hamamatsu Orca Flash4, e.g. 840 MB/s (2048x2048 px at 100 Hz).

 ## Acknowledgements
 This code was inspired by [Talley Lambert's](https://github.com/tlambert03/imarispy) code 
 and further input from Adam Glaser, [VolkerH](https://github.com/VolkerH), Doug Shepherd and 
 [Peter H](https://github.com/abred).
 
 To report issues or bugs please use the [issues](https://github.com/nvladimus/npy2bdv/issues) tool.
 
 ## Citation
 If you find this library useful, please cite it. Thanks!
 
 [![DOI](https://zenodo.org/badge/203410946.svg)](https://zenodo.org/badge/latestdoi/203410946)
