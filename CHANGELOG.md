## v.1.0.2 candidate
:exclamation: **API change**: The `BdvReader` class was replaced with `BdvEditor`, which allows e.g. streamlined reading and cropping views in H5 and XML files.
:mag: **Test coverage**: The tests became much more comprehensive and cover most of the functionality.

## v.1.0.1 stable
:bug: *Bugfix*: affine transformations of different views were mixed due to using mutable argument `m_affine` in `append_view()`

:gem: New example: writing multiple tiles on a grid, with (optionally) some tiles missing.

:book: API [documentation](https://nvladimus.github.io/npy2bdv/) is added.

:warning: Release notation changed from YYYY.mm.dd to 1.0.0.