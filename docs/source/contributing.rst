Contributing
============

Bug reports and pull requests are welcome on
`GitHub <https://github.com/nvladimus/npy2bdv>`_.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/nvladimus/npy2bdv.git
   cd npy2bdv
   mamba env create -f environment.yml
   mamba activate npy2bdv
   pip install -e .[docs]

Tests
-----

Run from the repository root:

.. code-block:: bash

   python -m pytest test

``test/test_docs.py`` runs every Python code block of this documentation, so a change that breaks
an example fails the tests.

Documentation
-------------

The docs are built with Sphinx from ``docs/source``:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

They are rebuilt and deployed to GitHub Pages on every push to ``master``.
API pages come from the docstrings, written in
`NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Releasing
---------

1. Bump ``version`` in ``pyproject.toml`` and add the release notes to ``CHANGELOG.md``.
2. Publish a GitHub release tagged ``v<version>``, e.g. ``v1.1.0``.

The ``publish`` workflow checks the tag against the version, runs the tests, and uploads the
package to PyPI.
