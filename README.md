# npy2bdv
 A minimalistic library for writing image stacks (numpy 3d-arrays) into HDF5 files in 
 Fiji BigDataViewer/BigStitcher format.
 
 Default options are optimized for high-speed writing, 
 to save microscopy images on the fly at full camera speed.
 
 Python 3.6.
 
 ## Installation
 Copy [`npy2bdv.py`](npy2bdv.py) into your code folder and use `import npy2bdv`.
 
 ## Data input
 Any 3-dimensional numpy arrays in (z,y,x) axis order, 
 data type `uint16`.
 
 ## Pipeline
 When a writer object is created, it opens a new h5 file 
 and requires info about setups and saving options: 
 number of setup attributes (e.g. channels, angles), compression and subsampling (if any). 
 File name must be new to avoid accidental data loss due to file re-writing.
 
 The image stacks (3d numpy arrays) are appended to h5 file 
 as new views by `.append_view()`. 
 Time point `time` and attributes (e.g. `channel`, `angle`) must be specified 
 for each view.
 
 The XML file is created at the end by calling `.write_xml_file()`.
 Total number of time points in the experiment `ntimes` 
 must be specified here (so, it may be unknown in the beginning of acquisition).
  
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
 
 ## ToDo
 Add automatic calculation of missing views IDs into XML file.
 
 ## Writing speed
Writing speeds up to 1300 MB/s can be achieved on a PC with SSD drive. 
The amount of available RAM does not seem to play role: 
writing on laptop with 16 GB RAM can be faster than on 64 GB RAM mainframe machine, if laptop SSD is faster.

The speed of writing for long time series (>100 stacks) is typically about 700-900 MB/s. This is in the range of full-speed camera acquisition 
of Hamamatsu Orca Flash4, 840 MB/s (2048x2048 px at 100 Hz).
