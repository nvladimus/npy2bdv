Datasets and views
==================

BigDataViewer organizes a dataset as a set of **views**. A view is one 3d stack, identified by:

* a **time point**, and
* a **setup**: one combination of the four *view attributes*
  **illumination**, **channel**, **tile** and **angle**.

For example, a light-sheet experiment with 2 channels, 2 illumination sides and 3 tiles has
2 × 2 × 3 = 12 setups. Imaged at 10 time points, it has 120 views.

How npy2bdv maps this
---------------------

You declare how many values each attribute has when creating the writer. Everything not
declared defaults to 1:

.. code-block:: python

   import numpy as np
   import npy2bdv

   writer = npy2bdv.BdvWriter("experiment.h5", nchannels=2, nilluminations=2, ntiles=3)
   print(writer.nsetups)   # 12

Then each view is written with its time point and attribute indices, all zero-based:

.. code-block:: python

   stack = np.zeros((16, 64, 64), dtype=np.uint16)
   for t in range(2):
       for ch in range(2):
           for ill in range(2):
               for tile in range(3):
                   writer.append_view(stack, time=t, channel=ch, illumination=ill, tile=tile)
   writer.write_xml()
   writer.close()

The number of time points is not declared: it grows with the largest ``time`` written.

Rules worth knowing
-------------------

* **Order does not matter.** Views can be written in any order, including time points out of order.
* **Missing views are fine.** Views never written are listed as *missing* in the XML file, and
  BigDataViewer/BigStitcher skips them. See :ref:`missing-views`.
* **Writing a view twice replaces it.**
* **Setups can differ** in shape, voxel size and exposure. Within one setup, the shape must be the
  same at all time points, because the XML file stores one shape per setup: ``append_view`` raises
  ``ValueError`` otherwise. The pixel type and the resolution levels are shared by all views in a file.
* **Arrays are (z, y, x)**, while BigDataViewer and the XML file use (x, y, z). Parameters
  named ``..._xyz`` follow the XML order.

Setup numbering
---------------

Internally, setups are numbered 0 … ``nsetups - 1`` with the angle changing fastest, then tile,
channel, and illumination. You rarely need this number: all methods take attribute indices.
It appears in the file layout (``s00``, ``s01``, …), see :doc:`file_format`.
