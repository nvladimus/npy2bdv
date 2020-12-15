## v.1.0.3 candidate

:exclamation: **API change**: The `BdvReader` class was replaced with `BdvEditor`, which allows e.g. streamlined reading and cropping views in H5 and XML files.

:gem: new function `BdvWriter.append_substack()` to write substacks into a virtual stack. 
This allows saving virtual stacks with multi-resolution pyramids downsampled in `z` as well as `(x,y)`.

:gem: `BdvEditor.append_affine()`: affine transformation can be appended to an existing dataset (before, it was only possible when creating a new dataset via `BdvWriter.write_xml_file()`). 

:mag: **Test coverage**: The tests became much more comprehensive and cover most of the functionality.

## v.1.0.1 stable
:bug: *Bugfix*: affine transformations of different views were mixed due to using mutable argument `m_affine` in `append_view()`

:gem: New example: writing multiple tiles on a grid, with (optionally) some tiles missing.

:book: API [documentation](https://nvladimus.github.io/npy2bdv/) is added.

:warning: Release notation changed from YYYY.mm.dd to 1.0.0.