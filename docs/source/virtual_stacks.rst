Virtual stacks
==============

When a view does not fit in RAM, or arrives plane by plane from a camera, allocate it as a
**virtual stack** and fill it in pieces. Call :meth:`~npy2bdv.BdvWriter.append_view` with
``stack=None`` and the full shape, then write planes or blocks into it.

.. code-block:: python

   import numpy as np
   import npy2bdv

   nz, ny, nx = 100, 256, 256
   rng = np.random.default_rng(0)

Plane by plane
--------------

Typical for acquisition: each camera frame is written as soon as it arrives.

.. code-block:: python

   writer = npy2bdv.BdvWriter("by_plane.h5", nchannels=2, blockdim=((1, 256, 256),))
   for ch in range(2):
       writer.append_view(stack=None, virtual_stack_dim=(nz, ny, nx), channel=ch)
       for z in range(nz):
           frame = rng.integers(0, 4000, (ny, nx), dtype=np.uint16)   # e.g. camera.get_frame()
           writer.append_plane(frame, z=z, channel=ch)
   writer.write_xml()
   writer.close()

A block depth of 1 in z, as in ``blockdim=((1, 256, 256),)``, keeps per-plane writing fast.

**Pyramids:** each plane is downsampled on its own, so subsampling in z must be 1:

.. code-block:: python

   writer = npy2bdv.BdvWriter("by_plane_pyramid.h5",
                              subsamp=((1, 1, 1), (1, 4, 4)),
                              blockdim=((1, 128, 128),))
   writer.append_view(stack=None, virtual_stack_dim=(nz, ny, nx))
   for z in range(nz):
       writer.append_plane(rng.integers(0, 4000, (ny, nx), dtype=np.uint16), z=z)
   writer.write_xml()
   writer.close()

For z-downsampling, use substacks, or write raw planes and call
:meth:`~npy2bdv.BdvWriter.create_pyramids` at the end.

Block by block
--------------

:meth:`~npy2bdv.BdvWriter.append_substack` writes a 3d block at a (z, y, x) offset. Use it to convert
data that is processed or loaded in chunks. Blocks can come in any order and need not align with
the subsampling factors; the pyramid is identical to writing the whole stack at once.

.. code-block:: python

   writer = npy2bdv.BdvWriter("by_substack.h5",
                              subsamp=((1, 1, 1), (2, 4, 4)),
                              blockdim=((16, 64, 64),))
   writer.append_view(stack=None, virtual_stack_dim=(nz, ny, nx))
   for z0 in range(0, nz, 30):                     # 30-plane chunks; the last one is shorter
       chunk = rng.integers(0, 4000, (min(30, nz - z0), ny, nx), dtype=np.uint16)
       writer.append_substack(chunk, z_start=z0)
   writer.write_xml()
   writer.close()

Blocks can also be offset in y and x, e.g. to assemble a view from a grid of sub-volumes:

.. code-block:: python

   writer = npy2bdv.BdvWriter("by_xy_blocks.h5")
   writer.append_view(stack=None, virtual_stack_dim=(nz, ny, nx))
   for y0 in (0, 128):
       for x0 in (0, 128):
           block = rng.integers(0, 4000, (nz, 128, 128), dtype=np.uint16)
           writer.append_substack(block, z_start=0, y_start=y0, x_start=x0)
   writer.write_xml()
   writer.close()

.. note::

   Parts of a virtual stack that are never written read as zeros.
