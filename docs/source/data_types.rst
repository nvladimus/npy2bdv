Pixel types
===========

All views in a file share one pixel type, chosen with the ``dtype`` argument of
:class:`~npy2bdv.BdvWriter`:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``dtype``
     - Range
     - Use for
   * - ``'uint16'`` (default)
     - 0 … 65535
     - Raw data from most scientific cameras.
   * - ``'uint8'``
     - 0 … 255
     - 8-bit data, e.g. many confocal microscopes; half the size of uint16.
   * - ``'float32'``
     - any real value
     - Processed data: filtered, deconvolved, normalized, or sums that exceed 65535.

.. code-block:: python

   import numpy as np
   import npy2bdv

   rng = np.random.default_rng(0)
   processed = rng.normal(0, 100, (32, 128, 128)).astype(np.float32)   # has negative values

   writer = npy2bdv.BdvWriter("processed.h5", dtype='float32', subsamp=((1, 1, 1), (2, 2, 2)))
   writer.append_view(processed)
   writer.write_xml()
   writer.close()

   editor = npy2bdv.BdvEditor("processed.xml")
   data = editor.read_view()
   print(editor.dtype, data.dtype)    # float32 float32
   assert np.array_equal(data, processed)
   editor.finalize()

Casting
-------

Input arrays are cast to the writer's ``dtype`` with numpy rules. If the cast changes values (out of
range, fractions lost, or float32 precision exceeded), a ``UserWarning`` is issued, and the data is
written anyway. Casts that cannot lose anything, such as uint16 data into a uint16 file, are not
checked, so they cost no extra time. Scale data yourself when changing the type:

.. code-block:: python

   stack16 = rng.integers(0, 65535, (32, 128, 128), dtype=np.uint16)
   stack8 = (stack16 / 65535 * 255).astype(np.uint8)       # rescale to 0..255 first

   writer = npy2bdv.BdvWriter("eight_bit.h5", dtype='uint8')
   writer.append_view(stack8)        # writer.append_view(stack16) would warn: values > 255
   writer.write_xml()
   writer.close()

To turn the warning into an error, e.g. in a pipeline that must not alter data:

.. code-block:: python

   import warnings

   writer = npy2bdv.BdvWriter("strict.h5", dtype='uint8')
   with warnings.catch_warnings():
       warnings.simplefilter("error")
       try:
           writer.append_view(stack16)
       except UserWarning as e:
           print("refused:", e)
   writer.close()

Downsampled levels are averages cast back to the file's type, so for integer types they are rounded down.

Fiji compatibility
------------------

* **uint16** files open in every Fiji version. They are stored as int16, the original
  BigDataViewer convention, and read back as uint16.
* **uint8** and **float32** are stored natively, with a ``dataType`` attribute that BigDataViewer reads
  (see :doc:`file_format`). This needs **BigDataViewer-core 10.4.6 or newer**, included in any Fiji
  updated since 2023 (*Help › Update…*). Older versions ignore the attribute and read the data as
  16-bit integers: float32 values then lose their fractions and negative values are wrong.
