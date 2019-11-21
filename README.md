# npy2bdv
 A minimalistic library for writing image stacks (numpy 3d-arrays) into HDF5 files in 
 Fiji BigDataViewer/BigStitcher format.
 
 Default options are optimized for high-speed writing, 
 to save microscopy images on the fly at full camera speed.
 
 Python 3.6.
 
 ## Installation
 Copy [`npy2bdv.py`](npy2bdv.py) into your code folder and use `import npy2bdv`.
 
 ## Data input
 Any 3d numpy arrays in (z,y,x) axis order as `stack`, or 2d array (y,x) as `plane`. 
 The array type is converted to `uint16` inside npy2bdv.
 
 ## Pipeline
 When a writer object is created, it opens a new h5 file 
 and requires info about setups and saving options: 
 number of setup attributes (e.g. channels, angles), compression and subsampling (if any). 
 File name must be new to avoid accidental data loss due to file re-writing.
 
 The image stacks (3d numpy arrays) are appended to h5 file 
 as new views by `.append_view(stack, ...)`. 
 Time point `time` and attributes (e.g. `channel`, `angle`) must be specified 
 for each view.
 
 Stacks that are too huge to fit RAM can be written plane by plane using `.append_plane()` method. 
 Before calling this method, virtual stack must be initialized with 
 `.append_view(stack=None, virtual_stack_dim=stack_dim,...)` method, see Example6.
 
 The XML file is created in the end by calling `.write_xml_file(ntimes, ...)`.
 The total number of time points `ntimes` must be specified at this step. 
 So, it may be unknown in the beginning of acquisition.
  
 Writing is finalized by calling `.close()`.
 
 See [`examples.py`](examples.py) for details.
 
 ## Supported options
 * compression methods `None`, `gzip`, `lzf` (`None` by default).
 * downsampling possible for any number of mipmap levels (no downsampling by default). 
 Downsampling is done by averaging, compatible with BigDataViewer/BigStitcher convention.
 * block sizes for H5 storage (default `4,256,256`)
 * any number of time points, illuminations, channels, tiles, angles.
 * arbitrary affine transformation for each individual view (e.g. translation, rotation, shear).
 * arbitrary voxel calibration for each view, to account for spatial anisotropy.
 * individual views can differ in dimensions, voxel size, voxel units, 
 exposure time, and exposure units.
 * writing of camera properties into XML (new):
    * `name`
    * `exposureTime`
    * `exposureUnits`
 * writing of `generatedBy` meta-information into XML (new):
    * `microscope` (name and version),
    * `user`.
 * writing virtual stacks of arbitrary size plane-by-plane. Handy when your stack is larger than your RAM.
 
 ## ToDo
 Adding automatic calculation of missing views IDs into XML file, or tile positions.. Maybe.
 
 ## Writing speed
Writing speeds up to 2300 MB/s can be achieved on a PC with SSD drive. 
The amount of available RAM does not seem to play role: 
writing on laptop with 16 GB RAM can be faster than on 64 GB RAM mainframe machine, if laptop SSD is faster.

The speed of writing for long time series (>100 stacks) is typically about 700-900 MB/s. 
This is in the range of full-speed camera acquisition 
of Hamamatsu Orca Flash4, 840 MB/s (2048x2048 px at 100 Hz).

 ## Acknowledgements
 This code was inspired by [Talley Lambert's](https://github.com/tlambert03/imarispy) code 
 and further input from Adam Glaser, [VolkerH](https://github.com/VolkerH) and Doug Shepherd.
 
 To report issues or bugs please use the [issues](https://github.com/nvladimus/npy2bdv/issues) tool. 