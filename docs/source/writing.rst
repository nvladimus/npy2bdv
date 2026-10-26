Writing datasets
================

All examples on this page use a small test stack:

.. code-block:: python

   import numpy as np
   import npy2bdv

   rng = np.random.default_rng(0)
   stack = rng.integers(0, 4000, (32, 128, 128), dtype=np.uint16)   # (z, y, x)

The writing workflow
--------------------

1. Create a :class:`~npy2bdv.BdvWriter`, declaring the number of channels, illuminations,
   tiles and angles, and optionally resolution levels, compression and pixel type.
2. Call :meth:`~npy2bdv.BdvWriter.append_view` once per view.
3. Call :meth:`~npy2bdv.BdvWriter.write_xml` to write the XML file.
4. Call :meth:`~npy2bdv.BdvWriter.close` to close the H5 file.

Multi-channel, multi-view data
------------------------------

.. code-block:: python

   writer = npy2bdv.BdvWriter("multiview.h5", nchannels=2, nilluminations=2, nangles=2)

   for t in range(2):
       for ch in range(2):
           for ill in range(2):
               for angle in range(2):
                   writer.append_view(stack, time=t, channel=ch, illumination=ill, angle=angle)
   writer.write_xml()
   writer.close()

Naming channels and other attributes
------------------------------------

By default BigDataViewer shows attribute values as 0, 1, 2… Give them names before ``write_xml``:

.. code-block:: python

   writer = npy2bdv.BdvWriter("labeled.h5", nchannels=2, nilluminations=2, nangles=2)
   writer.set_attribute_labels('channel', ('488', '561'))
   writer.set_attribute_labels('illumination', ('left', 'right'))
   writer.set_attribute_labels('angle', ('0', '90'))
   for ch in range(2):
       for ill in range(2):
           for angle in range(2):
               writer.append_view(stack, channel=ch, illumination=ill, angle=angle)
   writer.write_xml()
   writer.close()

Voxel size and experiment metadata
----------------------------------

Voxel size, exposure and microscope details are stored in the XML file:

.. code-block:: python

   writer = npy2bdv.BdvWriter("metadata.h5")
   writer.append_view(stack,
                      voxel_size_xyz=(0.5, 0.5, 2.0), voxel_units='um',
                      exposure_time=10, exposure_units='ms')
   writer.write_xml(camera_name="Hamamatsu ORCA-Flash4.0",
                    microscope_name="mesoSPIM", microscope_version="v6",
                    user_name="nikita")
   writer.close()

.. note::

   ``voxel_size_xyz`` is metadata. What BigDataViewer uses to *display* the data is the list of
   transformations of each view. To show anisotropic voxels at the right aspect ratio, also pass
   a ``calibration`` with the same ratios, e.g. ``calibration=(1, 1, 4)`` for a z-step four
   times the pixel size.

.. _missing-views:

Missing views
-------------

If some views were not acquired, just skip them. They are marked as missing in the XML file:

.. code-block:: python

   writer = npy2bdv.BdvWriter("missing.h5", nchannels=2)
   writer.append_view(stack, time=0, channel=0)
   writer.append_view(stack, time=1, channel=1)   # (t=0, ch=1) and (t=1, ch=0) are missing
   writer.write_xml()
   writer.close()

Tiles and affine transformations
--------------------------------

Each view can carry an affine transformation: a (3, 4) matrix whose first three columns are
the linear part (rotation, scaling, shear) and last column the translation (x, y, z).
For tiled acquisitions, the translation places each tile in the global coordinate system,
so BigStitcher starts from the right layout:

.. code-block:: python

   n_tiles_x, n_tiles_y = 3, 2
   tile_h, tile_w = 128, 128
   overlap = 16   # px

   writer = npy2bdv.BdvWriter("tiles.h5", ntiles=n_tiles_x * n_tiles_y)
   for j in range(n_tiles_y):
       for i in range(n_tiles_x):
           affine = np.eye(3, 4)
           affine[0, 3] = i * (tile_w - overlap)   # x translation, px
           affine[1, 3] = j * (tile_h - overlap)   # y translation, px
           writer.append_view(stack, tile=i + j * n_tiles_x,
                              m_affine=affine, name_affine="tile position")
   writer.write_xml()
   writer.close()

The same mechanism can correct a shear, e.g. for light-sheet data acquired with a tilted stage.
This shifts each plane by ``-shear_px`` pixels in x per z-plane:

.. code-block:: python

   shear_px = 0.5
   shear = np.array([[1.0, 0.0, -shear_px, 0.0],
                     [0.0, 1.0, 0.0,       0.0],
                     [0.0, 0.0, 1.0,       0.0]])
   writer = npy2bdv.BdvWriter("sheared.h5")
   writer.append_view(stack, m_affine=shear, name_affine="deskew")
   writer.write_xml()
   writer.close()

Transformations can also be added after writing, see :doc:`reading_editing`.
