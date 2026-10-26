## v.1.1.0
:gem: New pixel types `uint8` and `float32`: `BdvWriter(..., dtype='uint8')` (issue #10). `BdvEditor` detects the type
from the H5 file. The H5 file follows the BigDataViewer convention (`dataType` attribute of each setup), readable in
Fiji with BigDataViewer-core >= 10.4.6. `uint16` files are unchanged and still open in older Fiji.

:bug: *Bugfix*: `BdvWriter.append_view()` can skip time points or write them out of order (PR #19 by @aaronkanzer)

:bug: *Bugfix*: pyramid levels of virtual stacks written by `append_substack()` are correct when substack edges
do not align with subsampling factors (PR #20 by @aaronkanzer)

:wrench: API docs are built and deployed to GitHub Pages automatically; publishing a GitHub release uploads to PyPI.

## October 2026
:wrench: Tested on Python 3.12 (numpy 2.5, h5py 3.16, scikit-image 0.26). Packaging moved from `setup.py` to `pyproject.toml`;
`environment.yml` added for mamba/conda installs. `tqdm` added to `requirements.txt`.

## July 2024
:bug: Speeds reported earlier in [`examples_h5writing.ipynb`](/docs/examples/examples_h5writing.ipynb) up to 2300 MB/s are likely due to bugs in testing. Currently speeds of 200 MB/s are more realistic.

## v.1.0.9
:bug: *Bugfix*: `BdvWriter.append_view()` now works correctly when downsampling odd-sized datasets, e.g. (31, 299, 301) (PR #15 by @pr4deepr)

:gem: `BdvEditor` can now open datasets in read-only mode, e.g. `BdvEditor('file.h5', mode='r')` (PR #15 by @pr4deepr)

:gem: `BdvEditor.read_view(z=...)` now supports reading a single plane from a view (PR #15 by @pr4deepr)

## v.1.0.7
:exclamation: **API change**: a mandatory parameter for the number of time points is dropped (redundancy), and the function 
name is shortened: `BdvWriter.write_xml(...)` instead of `BdvWriter.write_xml_file(ntimes=...)`.

:gem: Lazy downsampling and compression is added: `BdvWriter.create_pyramids()`. 
This allows high-speed writing of raw data during acquisition runtime, followed by slower
generation of downsampled and compressed image pyramids (when the time is less critical).

## v.1.0.6
:gem: Multiple XML files pointing to the same H5 data file are supported. 
This allows having several versions of registration or other processing (defined in XML files). 
See [PR #9](https://github.com/nvladimus/npy2bdv/pull/9).

:gem: Set labels for view attributes that will be visible in BDV/BigSticher, 
e.g. `.set_attribute_labels('channel', ('488', '561'))`.

## v.1.0.4

:exclamation: **API change**: The `BdvReader` class was replaced with `BdvEditor`, which allows e.g. streamlined reading and cropping views in H5 and XML files.

:gem: new function `BdvWriter.append_substack()` to write substacks into a virtual stack. 
This allows saving virtual stacks with multi-resolution pyramids downsampled in `z` as well as `(x,y)`.

:gem: `BdvWriter` and `BdvEditor` both have method `.read_affine(...)`, which reads an affine transformation of
a view from the XML file as numpy (3,4) array.

:gem: `BdvWriter` and `BdvEditor` both have method `.append_affine(...)`, which appends affine transformation matrix
 to an existing dataset. 
 This allows e.g. defining unshearing transform and tile position (via translation transform) after the data has been written.
 Before, only one transformation could be defined when creating a new dataset in `BdvWriter.append_view()`. 

:mag: **Test coverage**: The tests became much more comprehensive and cover most of the functionality.

## v.1.0.1 stable
:bug: *Bugfix*: affine transformations of different views were mixed due to using mutable argument `m_affine` in `append_view()`

:gem: New example: writing multiple tiles on a grid, with (optionally) some tiles missing.

:book: API [documentation](https://nvladimus.github.io/npy2bdv/) is added.

:warning: Release notation changed from YYYY.mm.dd to 1.0.0.