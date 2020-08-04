# npy2bdv
 A minimalistic library for writing image stacks (numpy 3d-arrays) into HDF5 files in 
 Fiji BigDataViewer/BigStitcher format. Also supports basic reading from HDF5 into numpy arrays.
 
 Default options are optimized for high-speed writing, 
 to save microscopy images on the fly at full camera speed.
 
 Python 3.6.
 
 ## Installation
 Run `pip install` from Anaconda prompt, and insert `import npy2bdv` in your code.
 
 ## Input
 Any 3d numpy arrays in (z,y,x) axis order as `stack`, or 2d array (y,x) as `plane`. 
 The array type is converted to `uint16` automatically.
 
 ## Pipeline
 When a writer object is created, it opens a new h5 file 
 and requires info about setups and saving options: 
 number of setup attributes (e.g. channels, angles), compression and subsampling (if any). 
 
 The image stacks (3d) are appended to h5 file 
 as views by `.append_view(stack, ...)`. 
 Time point `time` and attributes such as `channel`, `angle` etc must be specified 
 for each view.
 
 Stacks that are too huge to fit RAM can be written plane by plane using `.append_plane()` method. 
 Before calling this method, virtual stack must be initialized with 
 `.append_view(stack=None, virtual_stack_dim=stack_dim,...)` method, see Example6.
 
 The XML file is created in the end by calling `.write_xml_file(ntimes, ...)`.
 The total number of time points `ntimes` must be specified at this step. 
 So, it may be unknown in the beginning of acquisition.
  
 Writing is finalized by calling `.close()`.
 
 See [`examples.py`](/npy2bdv/examples.py) for details.
 
 ## Supported writing options
 * compression methods `None`, `gzip`, `lzf` (`None` by default).
 * downsampling possible for any number of mipmap levels (no downsampling by default). 
 Downsampling is done by averaging, compatible with BigDataViewer/BigStitcher convention.
 * block sizes for H5 storage (default `4,256,256`)
 * any number of time points, illuminations, channels, tiles, angles.
 * arbitrary affine transformation for each individual view (e.g. translation, rotation, shear).
 * arbitrary voxel calibration for each view, to account for spatial anisotropy.
 * individual views can differ in dimensions, voxel size, voxel units, 
 exposure time, and exposure units.
 * missing views are labeled automatically.
 * support of camera properties: `name`, `exposureTime`, `exposureUnits`
 * support of `generatedBy` meta-information: `microscope` (name and version), `user`
 * writing virtual stacks of arbitrary size plane-by-plane. Handy when your stack is larger than your RAM.
 
 ## Recent changes
 * Missing views handling.
 * Basic reader into numpy: `npy2bdv.BdvReader('file.h5') `
 
 ## Writing speed
Writing speeds up to 2300 MB/s can be achieved on a PC with SSD drive. 
The amount of available RAM does not seem to play role: 
writing on laptop with 16 GB RAM can be faster than on 64 GB RAM mainframe machine, if laptop SSD is faster.

The speed of writing for long time series (>100 stacks) is typically about 700-900 MB/s. 
This is in the range of full-speed camera acquisition 
of Hamamatsu Orca Flash4, 840 MB/s (2048x2048 px at 100 Hz).

 ## Acknowledgements
 This code was inspired by [Talley Lambert's](https://github.com/tlambert03/imarispy) code 
 and further input from Adam Glaser, [VolkerH](https://github.com/VolkerH), Doug Shepherd and 
 [Peter H](https://github.com/abred).
 
 To report issues or bugs please use the [issues](https://github.com/nvladimus/npy2bdv/issues) tool.
 
 ## Citation
 If you find this library useful, please cite it. Thanks!
 
 [![DOI](https://zenodo.org/badge/203410946.svg)](https://zenodo.org/badge/latestdoi/203410946)
