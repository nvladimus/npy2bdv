npy2bdv
=======

|pypi| |python| |license| |doi|

.. |pypi| image:: https://img.shields.io/pypi/v/npy2bdv.svg
   :target: https://pypi.org/project/npy2bdv/

.. |python| image:: https://img.shields.io/badge/python-3.12-blue.svg
   :target: https://www.python.org/downloads/

.. |license| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0

.. |doi| image:: https://zenodo.org/badge/203410946.svg
   :target: https://zenodo.org/badge/latestdoi/203410946

**npy2bdv** writes numpy arrays into the HDF5/XML format of
`Fiji BigDataViewer <https://imagej.net/plugins/bdv/>`_ and
`BigStitcher <https://imagej.net/plugins/bigstitcher/>`_, and reads them back.
It is built for light-sheet and other multi-view microscopy: many time points, channels,
illuminations, tiles and angles, fast writing during acquisition, and datasets larger than RAM.

.. code-block:: python

   import numpy as np
   import npy2bdv

   stack = np.random.randint(0, 1000, (64, 256, 256), dtype=np.uint16)  # (z, y, x)
   writer = npy2bdv.BdvWriter("first_dataset.h5")
   writer.append_view(stack)
   writer.write_xml()
   writer.close()

Open ``first_dataset.xml`` in Fiji with *Plugins › BigDataViewer › Open XML/HDF5*.

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Getting started
      :link: getting_started
      :link-type: doc

      Install, write a first dataset, open it in Fiji, and read it back.

   .. grid-item-card:: :octicon:`stack` Datasets and views
      :link: concepts
      :link-type: doc

      Time points, channels, illuminations, tiles, angles: how views are organized.

   .. grid-item-card:: :octicon:`pencil` Writing datasets
      :link: writing
      :link-type: doc

      Multi-channel and multi-view data, labels, metadata, missing views, tiles.

   .. grid-item-card:: :octicon:`zoom-in` Pyramids and compression
      :link: pyramids
      :link-type: doc

      Downsampled resolution levels, block sizes, compression, and writing speed.

   .. grid-item-card:: :octicon:`database` Virtual stacks
      :link: virtual_stacks
      :link-type: doc

      Write datasets larger than RAM, plane by plane or block by block.

   .. grid-item-card:: :octicon:`number` Pixel types
      :link: data_types
      :link-type: doc

      uint16, uint8 and float32 data, and which Fiji versions open them.

   .. grid-item-card:: :octicon:`search` Reading and editing
      :link: reading_editing
      :link-type: doc

      Read views, crop, and add transformations to existing datasets.

   .. grid-item-card:: :octicon:`file-binary` File format
      :link: file_format
      :link-type: doc

      What is inside the .h5 and .xml files.

   .. grid-item-card:: :octicon:`code` API reference
      :link: api/index
      :link-type: doc

      Every class and method with parameters and examples.

   .. grid-item-card:: :octicon:`history` Changelog
      :link: changelog
      :link-type: doc

      Release notes and version history.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User guide

   getting_started
   concepts
   writing
   pyramids
   virtual_stacks
   data_types
   reading_editing
   file_format

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api/index
   changelog
   contributing
