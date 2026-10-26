API reference
=============

The package exports two classes:

* :class:`~npy2bdv.BdvWriter` creates a new dataset.
* :class:`~npy2bdv.BdvEditor` reads and edits an existing one.

.. currentmodule:: npy2bdv

BdvWriter
---------

.. autoclass:: BdvWriter
   :members: append_view, append_plane, append_substack, set_attribute_labels, create_pyramids,
             write_xml, close, append_affine, read_affine

BdvEditor
---------

.. autoclass:: BdvEditor
   :members: read_view, get_view_property, get_attribute_count, read_affine, append_affine,
             crop_view, finalize
