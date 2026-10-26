Pyramids and compression
========================

.. code-block:: python

   import numpy as np
   import npy2bdv

   rng = np.random.default_rng(0)
   stack = rng.integers(0, 4000, (64, 256, 256), dtype=np.uint16)

Resolution levels
-----------------

BigDataViewer streams data at the resolution the screen needs. For big views, store
downsampled copies ("pyramid levels") next to the full-resolution data, so zoomed-out views load fast.
Levels are given as subsampling factors in **(z, y, x)** order; downsampling averages blocks of pixels.

.. code-block:: python

   writer = npy2bdv.BdvWriter("pyramid.h5",
                              subsamp=((1, 1, 1), (2, 4, 4), (4, 8, 8)),
                              blockdim=((16, 64, 64), (16, 64, 64), (16, 64, 64)))
   writer.append_view(stack)
   writer.write_xml()
   writer.close()

   editor = npy2bdv.BdvEditor("pyramid.xml")
   print(editor.read_view(ilevel=1).shape)   # (32, 64, 64)
   editor.finalize()

Good levels halve or quarter the size in each step and stop at a few hundred pixels per axis.
Subsampling only in (y, x), as in ``(1, 4, 4)``, is common for thin stacks.

Lazy pyramids: fast acquisition first
-------------------------------------

Computing levels while writing slows writing down. During acquisition, write raw data at full speed,
and compute the pyramid afterwards with :meth:`~npy2bdv.BdvWriter.create_pyramids`:

.. code-block:: python

   writer = npy2bdv.BdvWriter("lazy.h5", nchannels=2)
   for ch in range(2):
       writer.append_view(stack, channel=ch)          # fast: raw data only
   writer.create_pyramids(subsamp=((2, 4, 4), (4, 8, 8)),
                          blockdim=((16, 64, 64), (16, 64, 64)),
                          compression='gzip')          # slower, after acquisition
   writer.write_xml()
   writer.close()

The full-resolution level stays uncompressed; the new levels use the given compression.

Block size
----------

HDF5 stores each level in blocks (chunks). BigDataViewer loads whole blocks, so block size affects
both writing speed and browsing. Aim for 10 KB to 1 MB per block: the default ``(4, 256, 256)`` of
uint16 is 0.5 MB. If ``blockdim`` has fewer entries than ``subsamp``, the first block size is used for all levels.

Compression
-----------

``compression='gzip'`` makes files smaller, especially for data with a dark background, but writing
slower. ``'lzf'`` is faster with less compression, but it is an h5py-specific filter that Fiji's HDF5
reader does not support: use lzf only for files read back by Python. The default ``None`` is the fastest.

.. code-block:: python

   writer = npy2bdv.BdvWriter("compressed.h5", subsamp=((1, 1, 1), (2, 4, 4)), compression='gzip')
   writer.append_view(stack)
   writer.write_xml()
   writer.close()

Writing speed
-------------

Uncompressed writing without pyramids reached 150-320 MB/s in our tests. Downsampling and compression
on the fly slow writing down considerably (about 5x for virtual stacks in the example notebook), which
is why lazy pyramids are recommended during acquisition. Measure on your own hardware: disk speed dominates.
