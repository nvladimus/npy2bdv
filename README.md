# npy2bdv
 A library for writing/reading HDF5/XML datasets of 
 Fiji BigDataViewer/BigStitcher format as numpy arrays. 
 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 
 ## Installation
 Run `pip install npy2bdv` in the command line. Use `import npy2bdv` in the code.
 
 ## Documentation
For a quick start, see examples in the Jupyter 
[`notebook`](/docs/examples/examples_h5writing.ipynb).

API [documentation](https://nvladimus.github.io/npy2bdv/).

 ## Supported H5/XML writing options
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
Writing speeds up to 2300 MB/s can be achieved on a PC with SSD drive. 
The speed of writing for long time series (>100 stacks) is typically about 700-900 MB/s. 
This is in the range of full-speed camera acquisition 
of Hamamatsu Orca Flash4, e.g. 840 MB/s (2048x2048 px at 100 Hz).

 ## Acknowledgements
 This code was inspired by [Talley Lambert's](https://github.com/tlambert03/imarispy) code 
 and improved by generous input from Adam Glaser, [VolkerH](https://github.com/VolkerH), Doug Shepherd, 
 [Peter H](https://github.com/abred), [Mike Taormina](https://github.com/miketaormina) and [Pradeep Rajasekhar](https://github.com/pr4deepr).
 
 To report issues or bugs please use the [issues](https://github.com/nvladimus/npy2bdv/issues) tool.
 
 ## Citation
 You can cite it like so
 
 [![DOI](https://zenodo.org/badge/203410946.svg)](https://zenodo.org/badge/latestdoi/203410946)

## Support
If you find this library useful, send me some love :blush:

<a href="https://www.paypal.com/paypalme/NikitaVladimirov"><img src="/assets/donate-paypal-button-blue.svg" height="72"></a>

It will give me some dopamine and motivation to keep improving it :heart: 
