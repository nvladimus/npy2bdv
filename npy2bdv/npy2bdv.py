# Fast writing of numpy arrays to HDF5 format compatible with Fiji/BigDataViewer and BigStitcher
# Author: Nikita Vladimirov
# License: GPL-3.0
import os
import importlib.metadata
import h5py
import numpy as np
from xml.etree import ElementTree as ET
import skimage.transform
import shutil
import warnings
from pathlib import Path
from tqdm import trange


class BdvBase:
    try:
        __version__ = importlib.metadata.version('npy2bdv')
    except importlib.metadata.PackageNotFoundError:  # running from source without pip install
        __version__ = 'unknown'

    def __init__(self, filename):
        """Base class for `BdvWriter` and `BdvEditor`. Not intended for direct use.

        Parameters
        ----------
        filename : str
            Path to either the .h5 or the .xml file. The other file of the pair is in the same folder.
        """
        self._fmt = 't{:05d}/s{:02d}/{}'
        if filename[-2:] == 'h5':
            self.filename_h5 = filename
            self.filename_xml = filename[:-2] + 'xml'
        elif filename[-3:] == 'xml':
            if os.path.exists(filename):  # if the XML file already exists (editor mode)
                try:
                    et = ET.parse(filename)
                    root = et.getroot()
                    sq = root.find('SequenceDescription')
                    iml = sq.find('ImageLoader')
                    hdf5_ = iml.find('hdf5')
                    image_file_name = hdf5_.text
                    image_file = Path(filename).parent.joinpath(image_file_name)
                    self.filename_h5 = image_file
                    self.filename_xml = filename
                except Exception as e:
                    raise ValueError(f"Could no parse XML file {filename}")
            else:  # to create a new file pair H5/XML
                self.filename_h5 = filename[:-3] + 'h5'
                self.filename_xml = filename
        self._root = None
        self.nlevels = None
        self.ntimes = self.nilluminations = self.nchannels = self.ntiles = self.nangles = self.nsetups = 0
        self.compression = None
        self.compressions_supported = (None, 'gzip', 'lzf')
        self.dtype = np.dtype('uint16')
        self.dtypes_supported = ('uint16', 'uint8', 'float32')

    @property
    def _h5_dtype(self):
        """Storage type in H5: uint16 is stored as int16 (legacy BDV convention), other types natively."""
        return np.dtype('int16') if self.dtype == np.uint16 else self.dtype

    def _determine_setup_id(self, illumination=0, channel=0, tile=0, angle=0):
        """Takes the view attributes (illumination, channel, tile, angle) and converts them into unique setup_id.
        Parameters:
        -----------
            illumination: int
            channel: int
            tile: int
            angle: int

        Returns:
        --------
            setup_id: int, >=0 (first setup)
            """
        if self.nsetups is not None:
            setup_id_matrix = np.arange(self.nsetups)
            setup_id_matrix = setup_id_matrix.reshape((self.nilluminations, self.nchannels, self.ntiles, self.nangles))
            setup_id = setup_id_matrix[illumination, channel, tile, angle]
        else:
            setup_id = None
        return setup_id

    def _get_xml_root(self):
        """Load the meta-information information from XML header file"""
        assert os.path.exists(self.filename_xml), f"Error: {self.filename_xml} file not found"
        if self._root is None:
            with open(self.filename_xml, 'r') as file:
                self._root = ET.parse(file).getroot()
        else:
            pass

    def read_affine(self, time=0, illumination=0, channel=0, tile=0, angle=0, index=0):
        """Read an affine transformation of a view from the XML file.

        A view can have several transformations (e.g. calibration, then registration).
        They are stored as a list in the XML file; BigDataViewer applies them bottom to top.

        Parameters
        ----------
        time : int
            Time point index, >= 0.
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0.
        index : int
            Index of the transformation in the list. Default 0 is the top one, which is applied last.

        Returns
        -------
        numpy.ndarray
            Float array of shape (3, 4). The first three columns are the linear part,
            the last column is the translation (x, y, z).

        Examples
        --------
        >>> editor = npy2bdv.BdvEditor("dataset.xml")
        >>> affine = editor.read_affine(time=0, tile=1)
        """
        self._get_xml_root()
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        found = False
        for node in self._root.findall('./ViewRegistrations/ViewRegistration'):
            if int(node.attrib['setup']) == isetup and int(node.attrib['timepoint']) == time:
                found = True
                break
        assert found, f'Node not found: <ViewRegistration setup="{isetup}" timepoint="{time}">'
        assert index < len(node), f'Index {index} out of range, only {len(node)} transforms found.'
        affine_str = node[index].find('affine').text
        affine_mx = np.fromstring(affine_str, sep='\n').reshape(3,4)
        return affine_mx

    def append_affine(self, m_affine, name_affine="Appended affine transformation using npy2bdv.",
                      time=0, illumination=0, channel=0, tile=0, angle=0):
        """Append an affine transformation to a view, on top of the existing ones.

        The new transformation is placed at the top of the list, so BigDataViewer/BigStitcher
        applies it last. The XML file is written immediately.
        With `BdvWriter`, call `BdvWriter.write_xml` first, so that the XML file exists.

        Parameters
        ----------
        m_affine : numpy.ndarray
            Array of shape (3, 4). The first three columns are the images of the x, y, z unit vectors
            (rotation, scaling, shear), the last column is the translation (x, y, z), in pixels.
        name_affine : str, optional
            Name of the transformation, shown in BigStitcher.
        time : int
            Time point index, >= 0.
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0.

        Examples
        --------
        Shift tile 1 by 100 px in x:

        >>> shift = np.array([[1., 0, 0, 100], [0, 1, 0, 0], [0, 0, 1, 0]])
        >>> editor.append_affine(shift, name_affine="tile shift", tile=1)
        """
        self._get_xml_root()
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        assert m_affine.shape == (3,4), "m_affine must be a numpy array of shape (3,4)"
        found = False
        for node in self._root.findall('./ViewRegistrations/ViewRegistration'):
            if int(node.attrib['setup']) == isetup and int(node.attrib['timepoint']) == time:
                found = True
                break
        assert found, f'Node not found: <ViewRegistration setup="{isetup}" timepoint="{time}">'
        vt = ET.Element('ViewTransform')
        node.insert(0, vt)
        vt.set('type', 'affine')
        ET.SubElement(vt, 'Name').text = name_affine
        mx_string = np.array2string(m_affine.flatten(), formatter={'float':lambda x: "%.6f" % x})
        ET.SubElement(vt, 'affine').text = mx_string[1:-1].strip()
        self._xml_indent(self._root)
        tree = ET.ElementTree(self._root)
        tree.write(self.filename_xml, xml_declaration=True, encoding='utf-8', method="xml")

    def _xml_indent(self, elem, level=0):
        """Pretty printing function"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._xml_indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _subsample_stack(self, stack, subsamp_level):
        """Subsampling of a 3d stack.

        Parameters:
        -----------
            stack, numpy 3d array (z,y,x)
            subsamp_level, array-like with 3 elements, eg (2,4,4) for downsampling z(x2), x and y (x4).

        Returns:
        --------
            down-scaled stack, of type self.dtype.
        """
        if all(subsamp_level[:] == 1):
            stack_sub = stack
        else:
            stack_sub = skimage.transform.downscale_local_mean(stack, tuple(subsamp_level)).astype(self.dtype)
        return stack_sub

    def _write_pyramids_header(self):
        """Write resolutions and subdivisions for all setups into h5 file."""
        for isetup in range(self.nsetups):
            group_name = 's{:02d}'.format(isetup)
            if group_name in self._file_object_h5:
                grp = self._file_object_h5[group_name]
                flipped_subsamp = np.flip(self.subsamp, 1)
                flipped_blockdim = np.flip(self.chunks, 1)
                res_dataset = grp['resolutions']
                subdiv_dataset = grp['subdivisions']
                res_dataset.resize(flipped_subsamp.shape)
                subdiv_dataset.resize(flipped_blockdim.shape)
                res_dataset[:] = flipped_subsamp
                subdiv_dataset[:] = flipped_blockdim
            else:
                raise ValueError(f"Group name {group_name} not found in the H5 file.")

    def create_pyramids(self, subsamp=((4, 8, 8),), blockdim=((8, 128, 128),), compression=None) -> None:
        """Compute and write downsampled resolution levels (pyramids) of all views already written.

        Use it for "lazy" downsampling: write raw full-resolution data at full speed during
        acquisition, then call this method once before `write_xml`. The writer must have been
        created with a single resolution level (the default `subsamp`).

        Parameters
        ----------
        subsamp : tuple of tuples
            Additional subsampling levels in (z, y, x) order, integers >= 1, e.g. ``((2, 4, 4), (4, 8, 8))``.
            The full-resolution level is kept and does not need to be listed.
        blockdim : tuple of tuples
            HDF5 block (chunk) size of each new level, in (z, y, x) order, same length as `subsamp`.
        compression : {None, 'gzip', 'lzf'}
            HDF5 compression of the new levels. Default None.

        Examples
        --------
        >>> writer = npy2bdv.BdvWriter("raw.h5")
        >>> writer.append_view(stack)
        >>> writer.create_pyramids(subsamp=((2, 4, 4),), blockdim=((16, 64, 64),))
        >>> writer.write_xml()
        >>> writer.close()
        """
        assert len(self.subsamp) == 1, f"Image pyramids already exist, len(self.subsamp) = {len(self.subsamp)}"
        assert len(self.chunks) == 1, f"Image pyramids already exist, len(self.chunks) = {len(self.chunks)}"
        assert self.nsetups > 0, f"Dataset has no views! self.nsetups = {self.nsetups}"
        assert self._file_object_h5 is not None, "H5 file object (self._file_object_h5) is None!"
        assert len(self.subsamp) == len(self.chunks), f"Length of subsampling tuple {len(subsamp)} must " \
                                              f"be == length of block dimensions {len(blockdim)}."
        for isub in range(len(subsamp)):
            assert sum(subsamp[isub]) > 3, f"At least one subsampling factor from {subsamp[isub]} must be > 1."
        assert compression in self.compressions_supported, f'Unknown compression, must be one of' \
                                                           f' {self.compressions_supported}'

        self.subsamp = np.asarray([self.subsamp[0]] + list(subsamp))
        self.chunks = np.asarray([self.chunks[0]] + list(blockdim))
        self.nlevels = len(self.subsamp)

        self._write_pyramids_header()
        for time in trange(self.ntimes, desc='time points'):
            for isetup in trange(self.nsetups, desc='views'):
                for ilevel in range(1, self.nlevels):
                    full_res_group_name = self._fmt.format(time, isetup, 0)
                    if full_res_group_name in self._file_object_h5:
                        raw_data = self._file_object_h5[full_res_group_name]['cells'][()].astype(self.dtype)
                        pyramid_group_name = self._fmt.format(time, isetup, ilevel)
                        grp = self._file_object_h5.create_group(pyramid_group_name)
                        subdata = self._subsample_stack(raw_data, self.subsamp[ilevel]).astype(self._h5_dtype)
                        grp.create_dataset('cells', data=subdata, chunks=tuple(self.chunks[ilevel]),
                                           maxshape=(None, None, None), compression=compression, dtype=self._h5_dtype)


class BdvWriter(BdvBase):
    """Write numpy arrays into a new HDF5/XML file pair readable by Fiji BigDataViewer and BigStitcher.

    A dataset consists of *views*: 3d stacks, each identified by its time point and four attributes
    (illumination, channel, tile, angle). The number of each attribute is fixed when the writer is
    created; views are then added with `append_view` in any order, and missing views are allowed.

    Typical workflow: create the writer, call `append_view` for every view, then `write_xml`
    and `close`.

    Parameters
    ----------
    filename : str
        Path of the .h5 file to create (or of the .xml file; the other file of the pair gets the
        same name). The XML file is written by `write_xml`.
    subsamp : tuple of tuples
        Resolution levels as subsampling factors in (z, y, x) order, integers >= 1.
        The first level is normally full resolution. Default ``((1, 1, 1),)``: full resolution only.
        Example with two downsampled levels: ``((1, 1, 1), (2, 4, 4), (4, 8, 8))``.
    blockdim : tuple of tuples
        HDF5 block (chunk) size of each level, in pixels, (z, y, x) order. If fewer entries than
        `subsamp` are given, the first one is used for all levels. Default ``((4, 256, 256),)``.
    compression : {None, 'gzip', 'lzf'}
        HDF5 compression. Default None, the fastest for writing.
    nilluminations, nchannels, ntiles, nangles : int
        Number of illuminations, channels, tiles and angles, each >= 1. Default 1.
    overwrite : bool
        If True, an existing file is replaced. If False (default), an existing file raises `FileExistsError`.
    dtype : {'uint16', 'uint8', 'float32'}
        Pixel type of all views, default 'uint16'. Input arrays are cast to this type. If that changes
        values (out of range, or fractions lost), a `UserWarning` is issued. Types other than uint16 need
        BigDataViewer-core >= 10.4.6 (any Fiji updated since 2023).

    Notes
    -----
    Recommended HDF5 block size is between 10 KB and 1 MB: e.g. ``(4, 256, 256)`` px of uint16 is 0.5 MB
    and writes fast. Blocks can be larger than the stack.

    Examples
    --------
    >>> import numpy as np, npy2bdv
    >>> stack = np.random.randint(0, 1000, (64, 512, 512), dtype=np.uint16)
    >>> writer = npy2bdv.BdvWriter("dataset.h5", nchannels=2)
    >>> writer.append_view(stack, time=0, channel=0)
    >>> writer.append_view(stack, time=0, channel=1)
    >>> writer.write_xml()
    >>> writer.close()
    """

    def __init__(self, filename,
                 subsamp=((1, 1, 1),),
                 blockdim=((4, 256, 256),),
                 compression=None,
                 nilluminations=1, nchannels=1, ntiles=1, nangles=1,
                 overwrite=False, dtype='uint16'):
        super().__init__(filename)
        assert nilluminations >= 1, "Total number of illuminations must be at least 1."
        assert nchannels >= 1, "Total number of channels must be at least 1."
        assert ntiles >= 1, "Total number of tiles must be at least 1."
        assert nangles >= 1, "Total number of angles must be at least 1."
        assert compression in self.compressions_supported, f'Unknown compression, must be one of' \
                                                           f' {self.compressions_supported}'
        assert all([isinstance(element, int) for tupl in subsamp for element in
                    tupl]), 'subsamp values should be integers >= 1.'
        assert dtype in self.dtypes_supported, f'Unknown dtype {dtype}, must be one of {self.dtypes_supported}'
        self.dtype = np.dtype(dtype)
        if len(blockdim) < len(subsamp):
            print(f"INFO: blockdim levels ({len(blockdim)}) < subsamp levels ({len(subsamp)}):"
                  f" First-level block size {blockdim[0]} will be used for all levels")
        self.nsetups = nilluminations * nchannels * ntiles * nangles
        self.nilluminations = nilluminations
        self.nchannels = nchannels
        self.ntiles = ntiles
        self.nangles = nangles
        self.attribute_counts = {'illumination': self.nilluminations, 'channel': self.nchannels,
                                 'angle': self.nangles, 'tile': self.ntiles}
        self.subsamp = np.asarray(subsamp)
        self.nlevels = len(subsamp)
        self.chunks = self._compute_chunk_size(blockdim)
        self.stack_shapes = {}
        self.affine_matrices = {}
        self.affine_names = {}
        self.calibrations = {}
        self.voxel_size_xyz = {}
        self.voxel_units = {}
        self.exposure_time = {}
        self.exposure_units = {}
        self.attribute_labels = {}
        self.compression = compression
        if os.path.exists(self.filename_h5):
            if overwrite:
                os.remove(self.filename_h5)
                print("Warning: H5 file already exists, overwriting.")
            else:
                raise FileExistsError(f"File {self.filename_h5} already exists.")
        self._file_object_h5 = h5py.File(self.filename_h5, 'a')
        self._write_setups_header()
        self.virtual_stacks = False
        self.setup_id_present = [[False] * self.nsetups]

    def set_attribute_labels(self, attribute: str, labels: tuple) -> None:
        """Set names of attribute values, shown in BigDataViewer/BigStitcher instead of 0, 1, 2...

        Call before `write_xml`.

        Parameters
        ----------
        attribute : {'illumination', 'channel', 'tile', 'angle'}
            The view attribute to label.
        labels : tuple of str
            One label per attribute value, e.g. ``('488', '561')`` for 2 channels or ``('left', 'right')``
            for 2 illuminations. Length must equal the number of values of this attribute.

        Examples
        --------
        >>> writer.set_attribute_labels('channel', ('488', '561'))
        """

        assert attribute in self.attribute_counts.keys(), f'Attribute must be one of {self.attribute_counts.keys()}'
        assert len(labels) == self.attribute_counts[attribute], f'Length of labels {len(labels)} must ' \
                                                   f'match the number of attributes {self.attribute_counts[attribute]}'
        self.attribute_labels[attribute] = labels

    def _compute_chunk_size(self, blockdim):
        """Populate the size of h5 chunks (blocks).
        Use first-level chunk size if there are more subsampling levels than chunk size levels.
        """
        chunks = []
        base_level = blockdim[0]
        if len(blockdim) < len(self.subsamp):
            for ilevel in range(len(self.subsamp)):
                chunks.append(base_level)
            chunks_tuple = tuple(chunks)
        else:
            chunks_tuple = blockdim
        return chunks_tuple

    def _write_setups_header(self):
        """Write resolutions and subdivisions for all setups into h5 file."""
        for isetup in range(self.nsetups):
            group_name = 's{:02d}'.format(isetup)
            if group_name in self._file_object_h5:
                del self._file_object_h5[group_name]
            grp = self._file_object_h5.create_group(group_name)
            data_subsamp = np.flip(self.subsamp, 1)
            data_chunks = np.flip(self.chunks, 1)
            grp.create_dataset('resolutions', data=data_subsamp, dtype='<f8', maxshape=(None, 3))
            grp.create_dataset('subdivisions', data=data_chunks, dtype='<i4', maxshape=(None, 3))
            if self.dtype != np.uint16:  # BDV reads this attribute; without it, data is assumed (u)int16
                grp.attrs.create('dataType', np.bytes_(self.dtype.name))

    def append_plane(self, plane, z, time=0, illumination=0, channel=0, tile=0, angle=0):
        """Write one plane into a virtual stack.

        The virtual stack must first be allocated with
        ``append_view(stack=None, virtual_stack_dim=(nz, ny, nx), ...)`` for the same view.
        Planes can be written in any order. Downsampled levels are computed per plane, so all
        subsampling factors in z must be 1, e.g. ``subsamp=((1, 1, 1), (1, 4, 4))``.

        Parameters
        ----------
        plane : numpy.ndarray
            2d array of shape (ny, nx), equal to the (y, x) size of the virtual stack.
        z : int
            Plane index in the virtual stack, 0 <= z < nz.
        time : int
            Time point index, >= 0.
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0.

        Examples
        --------
        >>> writer.append_view(stack=None, virtual_stack_dim=(1000, 2048, 2048))
        >>> for z in range(1000):
        ...     writer.append_plane(camera.get_frame(), z=z)
        """
        
        assert self.virtual_stacks, "Appending planes requires initialization with virtual stack, " \
                                    "see append_view(stack=None,...)"
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        self._check_cast(plane)
        self._update_setup_id_present(isetup, time)
        assert plane.shape == self.stack_shapes[isetup][1:], f"Plane dimensions {plane.shape} do not match (y,x) size" \
                                                             f" of virtual stack {self.stack_shapes[isetup][1:]}."
        assert z < self.stack_shapes[isetup][0], f"Plane index {z} must be less than " \
                                                 f"virtual stack z-dimension {self.stack_shapes[isetup][0]}."
        for ilevel in range(self.nlevels):
            group_name = self._fmt.format(time, isetup, ilevel)
            dataset = self._file_object_h5[group_name]["cells"]
            dataset[z, :, :] = self._subsample_plane(plane, self.subsamp[ilevel]).astype(self._h5_dtype)

    def append_substack(self, substack, z_start, y_start=0, x_start=0,
                        time=0, illumination=0, channel=0, tile=0, angle=0):
        """Write a 3d block (substack) into a virtual stack.

        The virtual stack must first be allocated with
        ``append_view(stack=None, virtual_stack_dim=(nz, ny, nx), ...)`` for the same view.
        Substacks can be written in any order and need not align with the subsampling factors:
        downsampled levels are recomputed from the full-resolution data, so the final pyramid is the
        same as for `append_view` with the whole stack. Unlike `append_plane`, subsampling in z is allowed.

        Parameters
        ----------
        substack : numpy.ndarray
            3d array in (z, y, x) order. It must fit inside the virtual stack at the given offsets.
        z_start, y_start, x_start : int
            Offsets of the substack in the virtual stack, in pixels. `y_start` and `x_start` default to 0.
        time : int
            Time point index, >= 0.
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0.

        Examples
        --------
        >>> writer.append_view(stack=None, virtual_stack_dim=(400, 2048, 2048))
        >>> for z0 in range(0, 400, 100):
        ...     writer.append_substack(load_chunk(z0), z_start=z0)
        """

        assert self.virtual_stacks, "Appending substack requires initialization with virtual stack, " \
                                    "see append_view(stack=None,...)"
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        self._check_cast(substack)
        self._update_setup_id_present(isetup, time)
        assert z_start + substack.shape[0] <= self.stack_shapes[isetup][0], \
            f"Substack offset {z_start} + z-dim {substack.shape[0]} > virtual stack z-dim {self.stack_shapes[isetup][0]}."
        assert y_start + substack.shape[1] <= self.stack_shapes[isetup][1], \
            f"Substack offset {y_start} + y-dim {substack.shape[1]} > virtual stack y-dim {self.stack_shapes[isetup][1]}."
        assert x_start + substack.shape[2] <= self.stack_shapes[isetup][2], \
            f"Substack offset {x_start} + x-dim {substack.shape[2]} > virtual stack x-dim {self.stack_shapes[isetup][2]}."
        starts = (z_start, y_start, x_start)
        full_res = self._file_object_h5[self._fmt.format(time, isetup, 0)]["cells"]
        for ilevel in range(self.nlevels):
            group_name = self._fmt.format(time, isetup, ilevel)
            dataset = self._file_object_h5[group_name]["cells"]
            subsamp = self.subsamp[ilevel]
            if ilevel > 0 and all(self.subsamp[0] == 1):
                # Recompute every block of this level that the substack touches from the
                # full-resolution data, so that blocks shared with neighbouring substacks
                # are not averaged with zeros or overwritten.
                lo = [start // f for start, f in zip(starts, subsamp)]
                hi = [-(-(start + n) // f) for start, n, f in zip(starts, substack.shape, subsamp)]
                block = full_res[lo[0] * subsamp[0]:hi[0] * subsamp[0],
                                 lo[1] * subsamp[1]:hi[1] * subsamp[1],
                                 lo[2] * subsamp[2]:hi[2] * subsamp[2]].astype(self.dtype)
                dataset[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = \
                    self._subsample_stack(block, subsamp).astype(self._h5_dtype)
                continue
            subdata = self._subsample_stack(substack, subsamp).astype(self._h5_dtype)
            sub_z_start = int(z_start/subsamp[0])
            sub_y_start = int(y_start/subsamp[1])
            sub_x_start = int(x_start/subsamp[2])
            dataset[sub_z_start : sub_z_start + subdata.shape[0],
                    sub_y_start : sub_y_start + subdata.shape[1],
                    sub_x_start : sub_x_start + subdata.shape[2]] = subdata

    def append_view(self, stack, virtual_stack_dim=None,
                    time=0, illumination=0, channel=0, tile=0, angle=0,
                    m_affine=None, name_affine='manually defined',
                    voxel_size_xyz=(1, 1, 1), voxel_units='px', calibration=(1, 1, 1),
                    exposure_time=0, exposure_units='s'):
        """Write a 3d stack as one view, or allocate a virtual stack to be filled later.

        A view is identified by `time` and the attribute indices. Writing the same view twice replaces it.
        All resolution levels given in `subsamp` are computed and written immediately.

        Parameters
        ----------
        stack : numpy.ndarray or None
            3d array in (z, y, x) order. It is cast to the writer's `dtype`, with a `UserWarning`
            if that changes values.
            If None, an empty *virtual stack* of shape `virtual_stack_dim` is allocated, to be filled
            with `append_plane` or `append_substack` (for data larger than RAM).
        virtual_stack_dim : tuple of int, optional
            (nz, ny, nx) shape of the virtual stack. Required if `stack` is None.
        time : int
            Time point index, >= 0.
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0 and below the numbers given to the writer.
        m_affine : numpy.ndarray, optional
            Affine transformation of shape (3, 4) applied to this view in BigDataViewer, e.g. the tile
            position (last column: translation x, y, z in pixels) or a shear correction.
        name_affine : str, optional
            Name of the affine transformation, shown in BigStitcher.
        voxel_size_xyz : tuple of float
            Physical voxel size (x, y, z) in `voxel_units`. Default (1, 1, 1).
        voxel_units : str
            Unit of the voxel size, e.g. 'um'. Default 'px'.
        calibration : tuple of float
            Scaling (x, y, z) written as the 'calibration' transform. Default (1, 1, 1).
            To display anisotropic voxels at the correct aspect ratio, set e.g. ``(1, 1, 4)``
            for z-steps 4x larger than the pixel size.
        exposure_time : float
            Camera exposure time, stored as metadata. Default 0.
        exposure_units : str
            Unit of the exposure time. Default 's'.

        Raises
        ------
        ValueError
            If the shape differs from the shape of the same setup at another time point:
            the XML file stores one shape per setup.

        Examples
        --------
        >>> writer.append_view(stack, time=0, channel=1, voxel_size_xyz=(0.5, 0.5, 2), voxel_units='um')
        """
        
        assert len(calibration) == 3, "Calibration must be a tuple of 3 elements (x, y, z)."
        assert len(voxel_size_xyz) == 3, "Voxel size must be a tuple of 3 elements (x, y, z)."
        if stack is not None:
            assert len(stack.shape) == 3, "Stack should be a 3-dimensional numpy array (z,y,x)"
            shape = tuple(stack.shape)
        else:
            assert virtual_stack_dim is not None and len(virtual_stack_dim) == 3, \
                "Stack is virtual, so parameter virtual_stack_dim must be defined."
            shape = tuple(virtual_stack_dim)
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        # The XML file stores one shape per setup, so all time points of a setup must share it.
        other_times = [t for t, present in enumerate(self.setup_id_present) if t != time and present[isetup]]
        if other_times and tuple(self.stack_shapes[isetup]) != shape:
            raise ValueError(f"View shape {shape} differs from shape {tuple(self.stack_shapes[isetup])} of the same "
                             f"setup at time point {other_times[0]}. All time points of a setup must have one shape.")
        if stack is not None:
            self._check_cast(stack)
        if time > self.ntimes - 1:
            self.ntimes = time + 1
        self._update_setup_id_present(isetup, time)
        self.stack_shapes[isetup] = shape
        if stack is None:
            self.virtual_stacks = True

        for ilevel in range(self.nlevels):
            group_name = self._fmt.format(time, isetup, ilevel)
            if group_name in self._file_object_h5:
                print(f"Overwriting H5 group {group_name}")
                del self._file_object_h5[group_name]
            grp = self._file_object_h5.create_group(group_name)
            if stack is not None:
                subdata = self._subsample_stack(stack, self.subsamp[ilevel]).astype(self._h5_dtype)
                grp.create_dataset('cells', data=subdata, chunks=self.chunks[ilevel],
                                   maxshape=(None, None, None), compression=self.compression, dtype=self._h5_dtype)
            else:  # a virtual stack initialized
                grp.create_dataset('cells', chunks=self.chunks[ilevel],
                                   shape=tuple(np.ceil(virtual_stack_dim / self.subsamp[ilevel]).astype(int)),
                                   maxshape=(None, None, None), compression=self.compression, dtype=self._h5_dtype)
        if m_affine is not None:
            self.affine_matrices[isetup] = m_affine.copy()
            self.affine_names[isetup] = name_affine
        self.calibrations[isetup] = calibration
        self.voxel_size_xyz[isetup] = voxel_size_xyz
        self.voxel_units[isetup] = voxel_units
        self.exposure_time[isetup] = exposure_time
        self.exposure_units[isetup] = exposure_units

    def _subsample_plane(self, plane, subsamp_level):
        """Subsampling of a 2d plane.
        
        Parameters:
        -----------
            plane: numpy 2d array (y,x)
            subsamp_level: array-like with 3 elements, eg (1,4,4) for downsampling x and y (x4).
            
        Returns:
        --------
            down-scaled plane, of type self.dtype.
        """
        assert subsamp_level[0] == 1, "z-subsampling must be == 1 for virtual stacks."
        if all(subsamp_level[:] == 1):
            plane_sub = plane
        else:
            plane_sub = skimage.transform.downscale_local_mean(plane, tuple(subsamp_level[1:])).astype(self.dtype)
        return plane_sub

    def write_xml(self, camera_name="default",  microscope_name="default",
                       microscope_version="0.0", user_name="user"):
        """Write the XML file that describes the dataset. Call it after all views are written.

        The XML file lists views, their attributes, voxel sizes, transformations and missing views,
        and points to the .h5 file by relative path. Fiji opens the dataset via this file.

        Parameters
        ----------
        camera_name : str
            Camera name, stored as metadata for all views.
        microscope_name, microscope_version, user_name : str
            Stored as metadata in the ``<generatedBy>`` element.
        """
        assert self.ntimes >= 1, "Total number of time points must be at least 1."
        root = ET.Element('SpimData')
        root.set('version', '0.2')
        bp = ET.SubElement(root, 'BasePath')
        bp.set('type', 'relative')
        bp.text = '.'
        # new XML data, added by @nvladimus
        generator = ET.SubElement(root, 'generatedBy')
        library = ET.SubElement(generator, 'library')
        library.set('version', self.__version__)
        library.text = "npy2bdv"
        microscope = ET.SubElement(generator, 'microscope')
        ET.SubElement(microscope, 'name').text = microscope_name
        ET.SubElement(microscope, 'version').text = microscope_version
        ET.SubElement(microscope, 'user').text = user_name
        # end of new XML data

        seqdesc = ET.SubElement(root, 'SequenceDescription')
        imgload = ET.SubElement(seqdesc, 'ImageLoader')
        imgload.set('format', 'bdv.hdf5')
        el = ET.SubElement(imgload, 'hdf5')
        el.set('type', 'relative')
        el.text = os.path.basename(self.filename_h5)
        # write ViewSetups
        viewsets = ET.SubElement(seqdesc, 'ViewSetups')
        for iillumination in range(self.nilluminations):
            for ichannel in range(self.nchannels):
                for itile in range(self.ntiles):
                    for iangle in range(self.nangles):
                        isetup = self._determine_setup_id(iillumination, ichannel, itile, iangle)
                        if any([self.setup_id_present[t][isetup] for t in range(len(self.setup_id_present))]):
                            vs = ET.SubElement(viewsets, 'ViewSetup')
                            ET.SubElement(vs, 'id').text = str(isetup)
                            ET.SubElement(vs, 'name').text = 'setup ' + str(isetup)
                            nz, ny, nx = tuple(self.stack_shapes[isetup])
                            ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)
                            vox = ET.SubElement(vs, 'voxelSize')
                            ET.SubElement(vox, 'unit').text = self.voxel_units[isetup]
                            dx, dy, dz = self.voxel_size_xyz[isetup]
                            ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)
                            # new XML data, added by @nvladimus
                            cam = ET.SubElement(vs, 'camera')
                            ET.SubElement(cam, 'name').text = camera_name
                            ET.SubElement(cam, 'exposureTime').text = '{}'.format(self.exposure_time[isetup])
                            ET.SubElement(cam, 'exposureUnits').text = self.exposure_units[isetup]
                            # end of new XML data
                            a = ET.SubElement(vs, 'attributes')
                            ET.SubElement(a, 'illumination').text = str(iillumination)
                            ET.SubElement(a, 'channel').text = str(ichannel)
                            ET.SubElement(a, 'tile').text = str(itile)
                            ET.SubElement(a, 'angle').text = str(iangle)

        # write Attributes
        for attribute in self.attribute_counts.keys():
            attrs = ET.SubElement(viewsets, 'Attributes')
            attrs.set('name', attribute)
            for i_attr in range(self.attribute_counts[attribute]):
                att = ET.SubElement(attrs, attribute.capitalize())
                ET.SubElement(att, 'id').text = str(i_attr)
                if attribute in self.attribute_labels.keys() and i_attr < len(self.attribute_labels[attribute]):
                    name = str(self.attribute_labels[attribute][i_attr])
                else:
                    name = str(i_attr)
                ET.SubElement(att, 'name').text = name

        # Time points
        tpoints = ET.SubElement(seqdesc, 'Timepoints')
        tpoints.set('type', 'range')
        ET.SubElement(tpoints, 'first').text = str(0)
        ET.SubElement(tpoints, 'last').text = str(self.ntimes - 1)

        # missing views
        if any(True in l for l in self.setup_id_present):
            miss_views = ET.SubElement(seqdesc, 'MissingViews')
            for t in range(len(self.setup_id_present)):
                for i in range(len(self.setup_id_present[t])):
                    if not self.setup_id_present[t][i]:
                        miss_view = ET.SubElement(miss_views, 'MissingView')
                        miss_view.set('timepoint', str(t))
                        miss_view.set('setup', str(i))

        # Transformations of coordinate system
        vregs = ET.SubElement(root, 'ViewRegistrations')
        for itime in range(self.ntimes):
            for isetup in range(self.nsetups):
                if self.setup_id_present[itime][isetup]:
                    vreg = ET.SubElement(vregs, 'ViewRegistration')
                    vreg.set('timepoint', str(itime))
                    vreg.set('setup', str(isetup))
                    # write arbitrary affine transformation, specific for each view
                    if isetup in self.affine_matrices.keys():
                        vt = ET.SubElement(vreg, 'ViewTransform')
                        vt.set('type', 'affine')
                        ET.SubElement(vt, 'Name').text = self.affine_names[isetup]
                        mx_string = np.array2string(self.affine_matrices[isetup].flatten(), formatter={'float':lambda x: "%.6f" % x})
                        ET.SubElement(vt, 'affine').text = mx_string[1:-1].strip()

                    # write registration transformation (calibration)
                    vt = ET.SubElement(vreg, 'ViewTransform')
                    vt.set('type', 'affine')
                    ET.SubElement(vt, 'Name').text = 'calibration'
                    calx, caly, calz = self.calibrations[isetup]
                    ET.SubElement(vt, 'affine').text = \
                        '{} 0.0 0.0 0.0 0.0 {} 0.0 0.0 0.0 0.0 {} 0.0'.format(calx, caly, calz)

        self._xml_indent(root)
        tree = ET.ElementTree(root)
        tree.write(self.filename_xml, xml_declaration=True, encoding='utf-8', method="xml")

    def _check_cast(self, data):
        """Warn if casting `data` to the writer's dtype changes its values (out of range, or fractions lost).
        Safe casts, e.g. uint16 data into a uint16 writer, skip the value check and cost nothing."""
        if np.can_cast(data.dtype, self.dtype, casting='safe'):
            return
        with np.errstate(invalid='ignore', over='ignore'):
            unchanged = np.array_equal(data.astype(self.dtype), data, equal_nan=True)
        if not unchanged:
            warnings.warn(f"Values of {data.dtype} data change when cast to {self.dtype} (out of range or not "
                          f"integer). Rescale the data, or create BdvWriter with a matching dtype.", stacklevel=3)

    def _update_setup_id_present(self, isetup, itime):
        """Update the lookup table (list of lists) for missing setups"""
        while len(self.setup_id_present) <= itime:
            self.setup_id_present.append([False] * self.nsetups)
        self.setup_id_present[itime][isetup] = True

    def close(self):
        """Flush and close the H5 file. Call it last, after `write_xml`."""
        self._file_object_h5.flush()
        self._file_object_h5.close()


class BdvEditor(BdvBase):
    """Read and edit an existing BigDataViewer HDF5/XML file pair.

    Works with files written by npy2bdv, Fiji/BigStitcher, or any other tool using the
    BigDataViewer HDF5 format. The pixel type (uint16, uint8, float32) is detected from the H5 file.

    .. warning::
        Edits (`crop_view`, `append_affine`) change the files in place, without undo.
        `finalize` keeps a backup of the previous XML file as ``<name>.xml~1``.

    Parameters
    ----------
    filename : str
        Path to the .xml or the .h5 file. Both files must exist. Given an .xml file, the .h5 file
        it points to is used, so several XML files sharing one H5 file are supported.
    mode : {'r+', 'r'}
        'r+' (default) for reading and editing, 'r' for read-only access.

    Attributes
    ----------
    ntimes, nilluminations, nchannels, ntiles, nangles : int
        Number of time points and view attributes, read from the XML file.
    dtype : numpy.dtype
        Pixel type of the views.

    Examples
    --------
    >>> editor = npy2bdv.BdvEditor("dataset.xml", mode='r')
    >>> stack = editor.read_view(time=0, channel=1)
    >>> editor.finalize()
    """

    def __init__(self, filename, mode='r+'):
        super().__init__(filename)
        assert os.path.exists(self.filename_h5), f"Error: {self.filename_h5} file not found"
        assert os.path.exists(self.filename_xml), f"Error: {self.filename_xml} file not found"
        self._file_object_h5 = h5py.File(self.filename_h5, mode=mode)
        self._root = None
        self.mode = mode
        self.ntimes, self.nilluminations, self.nchannels, self.ntiles, self.nangles = self.get_attribute_count()
        self.nsetups = self.nilluminations * self.nchannels * self.ntiles * self.nangles
        data_type = self._file_object_h5['s00'].attrs.get('dataType', b'uint16')
        self.dtype = np.dtype(data_type.decode() if isinstance(data_type, bytes) else data_type)

    def get_attribute_count(self):
        """Count time points and view attributes in the XML file.

        Returns
        -------
        tuple of int
            ``(ntimes, nilluminations, nchannels, ntiles, nangles)``. The same values are available
            as editor attributes.
        """
        with open(self.filename_xml, 'r') as file:
            root = ET.parse(file).getroot()
            element = root.find("./SequenceDescription/Timepoints[@type='range']")
            nt = int(element.find('last').text) - int(element.find('first').text) + 1 if element is not None else 0
            ni = len(root.findall("./SequenceDescription/ViewSetups/Attributes[@name='illumination']/Illumination"))
            nch = len(root.findall("./SequenceDescription/ViewSetups/Attributes[@name='channel']/Channel"))
            ntiles = len(root.findall("./SequenceDescription/ViewSetups/Attributes[@name='tile']/Tile"))
            nang = len(root.findall("./SequenceDescription/ViewSetups/Attributes[@name='angle']/Angle"))
        return nt, ni, nch, ntiles, nang

    def read_view(self, time=0, illumination=0, channel=0, tile=0, angle=0, ilevel=0, z=None):
        """Read a view, or a single plane of it, into a numpy array.

        Parameters
        ----------
        time : int
            Time point index, >= 0.
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0.
        ilevel : int
            Resolution level: 0 (default) is full resolution, 1, 2... are the downsampled levels.
        z : int, optional
            If given, read only this plane (faster than reading the whole stack).

        Returns
        -------
        numpy.ndarray
            3d array (z, y, x), or 2d array (y, x) if `z` is given, of type `dtype`.

        Raises
        ------
        KeyError
            If the view or level does not exist in the H5 file (e.g. a missing view).

        Examples
        --------
        >>> stack = editor.read_view(time=0, channel=1)
        >>> preview = editor.read_view(ilevel=2)
        >>> plane = editor.read_view(z=100)
        """
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        group_name = self._fmt.format(time, isetup, ilevel)
        if self._file_object_h5:
            if z is None:
                dataset = self._file_object_h5[group_name]["cells"][()].astype(self.dtype)
            elif z >= 0:
                dataset = self._file_object_h5[group_name]["cells"][z, ...].astype(self.dtype)
            else:
                raise ValueError('z must be >= 0')
            return dataset
        else:
            raise ValueError('File object is None')

    def crop_view(self, bbox_xyz=((1, -1), (1, -1), None), illumination=0, channel=0, tile=0, angle=0, ilevel=0):
        """Crop a view in place, for all time points, in the H5 file and the XML file.

        The H5 data is cropped immediately; the new view size is written to the XML file by `finalize`.
        Only the given resolution level is cropped.

        Parameters
        ----------
        bbox_xyz : tuple
            Crop ranges ``((x_start, x_stop), (y_start, y_stop), (z_start, z_stop))``, each used as a
            Python slice (negative values count from the end). None leaves that axis uncropped.
            Default ``((1, -1), (1, -1), None)`` removes a 1-px border in x and y.
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0.
        ilevel : int
            Resolution level to crop, default 0.

        Examples
        --------
        >>> editor = npy2bdv.BdvEditor("dataset.xml")
        >>> editor.crop_view(bbox_xyz=((100, -100), None, (10, 200)), channel=0)
        >>> editor.finalize()
        """
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        if self.mode == 'r':
            raise ValueError('File is open in read-only mode')
        elif self._file_object_h5:
            for time in range(self.ntimes):
                group_name = self._fmt.format(time, isetup, ilevel)
                view_dataset = self._file_object_h5[group_name]["cells"]
                view_arr = view_dataset[()]
                if bbox_xyz[0]:
                    view_arr = view_arr[:, :, slice(*bbox_xyz[0])]
                if bbox_xyz[1]:
                    view_arr = view_arr[:, slice(*bbox_xyz[1]), :]
                if bbox_xyz[2]:
                    view_arr = view_arr[slice(*bbox_xyz[2]), :, :]
                view_dataset.resize(view_arr.shape)
                view_dataset[:] = view_arr # Always use braces here! A common mistake to omit them.
                self._file_object_h5.flush()
            # Edit the XML file as well.
            with open(self.filename_xml, self.mode) as file:
                self._get_xml_root()
                for elem in self._root.findall("./SequenceDescription/ViewSetups/ViewSetup"):
                    elem_id = elem.find("id")
                    if int(elem_id.text) == isetup:
                        nz, ny, nx = tuple(view_arr.shape)
                        elem_size = elem.find("size")
                        elem_size.text = '{} {} {}'.format(nx, ny, nz)
        else:
            raise FileNotFoundError(self.filename_h5)

    def get_view_property(self, key, illumination=0, channel=0, tile=0, angle=0) -> tuple:
        """Read a property of a view setup from the XML file.

        Setups do not change over time, so no time index is needed.

        Parameters
        ----------
        key : {'voxel_size', 'view_shape'}
            'voxel_size': physical voxel size (floats); 'view_shape': view size in pixels (ints).
        illumination, channel, tile, angle : int
            Indices of the view attributes, >= 0.

        Returns
        -------
        tuple
            Value in (x, y, z) order, as stored in the XML file. Note that `read_view` returns
            arrays in (z, y, x) order.

        Examples
        --------
        >>> editor.get_view_property('voxel_size')
        (0.5, 0.5, 2.0)
        """
        accepted_keys = ['voxel_size', 'view_shape']
        assert key in accepted_keys, f"Key {key} not recognized, must be one of: {accepted_keys}."
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        self._get_xml_root()
        if key == 'voxel_size':
            path = "./SequenceDescription/ViewSetups/ViewSetup/voxelSize/size"
            type_caster = float
        elif key == 'view_shape':
            path = "./SequenceDescription/ViewSetups/ViewSetup/size"
            type_caster = int
        props_list = self._root.findall(path)
        # Todo: possible bug here, if the views are not in setupID order.
        assert 0 <= isetup < len(props_list), f"Setup index {isetup} out of range 0..{len(props_list)-1}"
        value = tuple([type_caster(val) for val in props_list[isetup].text.split()])
        return value

    def finalize(self):
        """Save pending changes and close the files. Call it when done with the editor.

        The H5 file is flushed and closed. If the XML tree was loaded (by `read_affine`,
        `append_affine`, `crop_view` or `get_view_property`), the XML file is rewritten, and the
        previous version is kept as ``<name>.xml~1``.
        """
        if self._file_object_h5 is not None:
            self._file_object_h5.flush()
            self._file_object_h5.close()
        if self._root is not None:
            self._xml_indent(self._root)
            tree = ET.ElementTree(self._root)
            shutil.copy(self.filename_xml, self.filename_xml + '~1') # backup the previous XML file.
            tree.write(self.filename_xml, xml_declaration=True, encoding='utf-8', method="xml")


