File format
===========

npy2bdv writes the `BigDataViewer HDF5/XML format <https://imagej.net/plugins/bdv/>`_.
You don't need to know it to use the library, but it helps when inspecting files with other tools,
e.g. ``h5py`` or HDFView.

The H5 file
-----------

.. code-block:: text

   dataset.h5
   ├── s00/                      setup 0
   │   ├── resolutions           (nlevels, 3) float: subsampling factors, (x, y, z) order
   │   ├── subdivisions          (nlevels, 3) int: block sizes, (x, y, z) order
   │   └── @dataType             only for uint8 / float32: "uint8" or "float32"
   ├── s01/ ...                  one group per setup
   ├── t00000/                   time point 0
   │   ├── s00/                  setup 0
   │   │   ├── 0/cells           full resolution, (z, y, x)
   │   │   ├── 1/cells           resolution level 1
   │   │   └── ...
   │   └── s01/ ...
   └── t00001/ ...

* uint16 data is stored as **int16**, the original BigDataViewer convention, and reinterpreted
  as unsigned on reading. There is no ``dataType`` attribute.
* Other types are stored natively, and the setup group carries a ``dataType`` string attribute,
  as written by BigDataViewer's own exporter.
* Missing views have no group in the H5 file.

Inspecting a file with h5py:

.. code-block:: python

   import numpy as np
   import h5py
   import npy2bdv

   writer = npy2bdv.BdvWriter("inspect.h5", subsamp=((1, 1, 1), (2, 2, 2)))
   writer.append_view(np.zeros((8, 32, 32), dtype=np.uint16))
   writer.write_xml()
   writer.close()

   with h5py.File("inspect.h5", "r") as f:
       f.visit(print)
       print(f["s00/resolutions"][()])          # [[1. 1. 1.] [2. 2. 2.]]
       print(f["t00000/s00/1/cells"].shape)     # (4, 16, 16)

The XML file
------------

The XML file describes the dataset and points to the H5 file by relative path:

* ``<SequenceDescription>``: the ``ImageLoader`` (format ``bdv.hdf5`` and the H5 file name),
  ``ViewSetups`` with size, voxel size and attribute values of each setup, the attribute names,
  the ``Timepoints`` range and the ``MissingViews``.
* ``<ViewRegistrations>``: the list of affine transformations of each view, applied bottom to top.
  npy2bdv always writes a ``calibration`` transform, plus the ``m_affine`` passed to ``append_view``.
* ``<generatedBy>``: npy2bdv version, microscope name and version, and user name.

Because the XML file only refers to the H5 file, several XML files can point to the same data,
e.g. before and after registration in BigStitcher.
