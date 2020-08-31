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
 1. a writer object is created: it opens a new h5 file and requires info about setups and saving options: 
 number of setup attributes (e.g. channels, angles), compression, subsampling. 
 2. stacks (3d arrays) can be appended to the h5 file 
 as views by `BdvWriter.append_view(stack, ...)`. 
 Stacks can be normal (whole stack appended at once), or virtual, appended plane by plane.
 The time point `time` and attributes such as `channel`, `angle` etc must be specified 
 for each view.
 
 3. an XML file with all meta information is created in the end by `BdvWriter.write_xml_file(ntimes, ...)`.
 The total number of time points `ntimes` must be specified at this step 
 (it may be unknown in the beginning of acquisition).
  
 4. Writing is finalized by calling `BdvWriter.close()`.
 
 See [`notebook with examples`](/npy2bdv/examples.ipynb) for details.
 
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
 
 ## Recent changes (Aug. 2020)
 * virtual stacks can be written with multiple subsampling levels and compression.
 
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
