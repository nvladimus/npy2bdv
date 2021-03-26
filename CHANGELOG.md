## v.1.0.7
:exclamation: **API change**: number of time points `ntimes` is dropped from the 
`BdvWriter.write_xml_file(...)` parameters, it is now calculated automatically.

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