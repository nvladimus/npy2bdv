# npy2bdv
 A minimalistic library for writing image stacks (numpy 3d-arrays) into HDF5 files in 
 Fiji BigDataViewer/BigStitcher format.
 
 Default options are optimized for high-speed writing, 
 to save microscopy images on the fly at full camera speed.
 
 ## Installation
 Copy `npy2bdv.py` in your code folder and use `import npy2bdv`.
 
 ## Data input
 Any 3-dimensional numpy arrays in (z,y,x) axis order, 
 data type `uint16`.
 
 ## Pipeline
 When a writer object is created, it opens a new h5 file and writes basic 
 info about setups (total number, subdivisions and resolutions). 
 File name must be new to avoid accidental data loss.
 
 The 3d image stacks (numpy arrays) are appended to h5 file 
 as new views by `.append_view()`. 
 Time point `itime` and setup ID `isetup` need to be specified 
 for each view.
 
 The XML file is created by calling `.write_xml_file()`.
 Total number of time points, illuminations, channels, 
 tiles and angles must be specified and match the total 
 number of setups in h5 file.
  
 Writing is finalized by calling `.close()`.
 
 See `examples.py` for details.
 
 ## Supported options
 * compression methods 'gzip', 'lzf' (no compression by default)
 * down-sampling for any number of mipmap levels (no down-sampling by default)
 * any number of time points, illuminations, channels, tiles, angles.
 
 ## Limitations
 At the moment, all h5 chunk sizes have the same size 
 (4, 256, 256) for any mipmap level. 
 This makes h5 writing fast, but may be non-optimal for reading 
 or minimizing file size.
 
 ## Writing speed
Writing speeds up to 1200 MB/s can be achieved on a PC with SSD drive. 
The amount of available RAM does not seem to play role: 
writing on laptop with 16 GB RAM can be faster than on 64 GB RAM mainframe machine.

The speed of h5 writing using this library is typically about 700-900 MB/s 
(uncomressed data, no downsampling). This is in the range of full-speed camera acquisition 
of Hamamatsu Orca Flash4, 840 MB/s (2048x2048 px at 100 Hz).
