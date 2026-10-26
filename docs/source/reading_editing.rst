Reading and editing
===================

:class:`~npy2bdv.BdvEditor` opens an existing dataset, written by npy2bdv, Fiji, BigStitcher or any
other tool using the BigDataViewer HDF5 format. First, a dataset to work with:

.. code-block:: python

   import numpy as np
   import npy2bdv

   rng = np.random.default_rng(0)
   writer = npy2bdv.BdvWriter("to_edit.h5", nchannels=2, ntiles=2, subsamp=((1, 1, 1), (2, 4, 4)))
   for ch in range(2):
       for tile in range(2):
           writer.append_view(rng.integers(0, 4000, (32, 128, 128), dtype=np.uint16),
                              channel=ch, tile=tile, voxel_size_xyz=(0.5, 0.5, 2), voxel_units='um')
   writer.write_xml()
   writer.close()

Reading views
-------------

Open with ``mode='r'`` when you only read, so the files cannot be changed by accident.

.. code-block:: python

   editor = npy2bdv.BdvEditor("to_edit.xml", mode='r')
   print(editor.ntimes, editor.nchannels, editor.ntiles)     # 1 2 2

   stack = editor.read_view(channel=1, tile=0)               # (z, y, x) array
   preview = editor.read_view(channel=1, tile=0, ilevel=1)   # downsampled level
   plane = editor.read_view(channel=1, tile=0, z=10)         # one plane only, fast

   print(editor.get_view_property('voxel_size'))   # (0.5, 0.5, 2.0), in (x, y, z) order
   print(editor.get_view_property('view_shape'))   # (128, 128, 32), in (x, y, z) order
   editor.finalize()

Either file of the pair can be given. With the XML file, the H5 file it refers to is used,
so several XML files can share one H5 file, e.g. different registrations of the same data.

Transformations
---------------

Read the transformations of a view, or add new ones. Added transformations go on top of the list,
so BigDataViewer applies them last; the XML file is updated immediately.

.. code-block:: python

   editor = npy2bdv.BdvEditor("to_edit.xml")
   print(editor.read_affine(tile=1))     # the 'calibration' transform written by BdvWriter

   shift = np.eye(3, 4)
   shift[0, 3] = 112                     # move tile 1 by 112 px in x
   for ch in range(2):
       editor.append_affine(shift, name_affine="tile position", channel=ch, tile=1)
   print(editor.read_affine(channel=0, tile=1))    # index 0 is the newest
   editor.finalize()

Cropping
--------

:meth:`~npy2bdv.BdvEditor.crop_view` crops a view in place, at all time points. Each axis takes
``(start, stop)`` like a Python slice, or None to keep it whole:

.. code-block:: python

   editor = npy2bdv.BdvEditor("to_edit.xml")
   editor.crop_view(bbox_xyz=((10, -10), None, (0, 16)), channel=0, tile=0)
   editor.finalize()                     # writes the new size to the XML file

   editor = npy2bdv.BdvEditor("to_edit.xml", mode='r')
   print(editor.read_view(channel=0, tile=0).shape)   # (16, 128, 108)
   editor.finalize()

.. warning::

   Cropping cannot be undone. ``finalize`` keeps the previous XML file as ``to_edit.xml~1``, but the
   H5 data is changed for good. Only the given resolution level is cropped (default 0).
