Getting started
===============

Installation
------------

npy2bdv needs Python 3.9 or newer (tested on 3.12) and installs ``h5py``, ``numpy``,
``scikit-image`` and ``tqdm``.

.. code-block:: bash

   pip install npy2bdv

To work on the library itself, install from source with
`mamba <https://github.com/conda-forge/miniforge>`_ (or conda):

.. code-block:: bash

   git clone https://github.com/nvladimus/npy2bdv.git
   cd npy2bdv
   mamba env create -f environment.yml
   mamba activate npy2bdv

Your first dataset
------------------

A dataset is a pair of files: an ``.h5`` file with the pixel data and an ``.xml`` file that
describes it. Arrays are always in **(z, y, x)** order.

.. code-block:: python

   import numpy as np
   import npy2bdv

   # a test stack: a bright Gaussian blob, 64 planes of 256 x 256 pixels
   z, y, x = np.mgrid[-1:1:64j, -1:1:256j, -1:1:256j]
   stack = (60000 * np.exp(-(x**2 + y**2 + z**2) / 0.1)).astype(np.uint16)

   writer = npy2bdv.BdvWriter("blob.h5")   # creates blob.h5
   writer.append_view(stack)               # writes the stack as view 0
   writer.write_xml()                      # creates blob.xml
   writer.close()

The writer refuses to replace an existing file. Pass ``overwrite=True`` to allow it.

Opening it in Fiji
------------------

In Fiji, choose *Plugins › BigDataViewer › Open XML/HDF5* and select ``blob.xml``.
For stitching and registration, start *Plugins › BigStitcher › BigStitcher* and select the
same XML file. No *Define dataset* step is needed: the XML file already defines it.

.. tip::

   In BigDataViewer, press :kbd:`S` to adjust brightness and contrast. A uint16 stack with
   low values may look black until the display range is set.

Reading it back
---------------

.. code-block:: python

   editor = npy2bdv.BdvEditor("blob.xml")
   data = editor.read_view()
   print(data.shape, data.dtype)       # (64, 256, 256) uint16
   assert np.array_equal(data, stack)
   editor.finalize()

Next steps
----------

* :doc:`concepts`: how time points, channels and other attributes map to views.
* :doc:`writing`: multi-channel datasets, labels, voxel sizes, tiles.
* :doc:`pyramids`: downsampled levels for smooth browsing of big data.
* The example notebook
  `examples_h5writing.ipynb <https://github.com/nvladimus/npy2bdv/blob/master/docs/examples/examples_h5writing.ipynb>`_
  with larger, timed examples.
