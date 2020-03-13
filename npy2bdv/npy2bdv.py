# Fast writing of numpy arrays to HDF5 format compatible with Fiji/BigDataViewer and BigStitcher
# Author: Nikita Vladimirov
# License: GPL-3.0
import os
import h5py
import numpy as np
from xml.etree import ElementTree as ET
import skimage.transform


class BdvWriter:
    def __init__(self, filename,
                 subsamp=((1, 1, 1),),
                 blockdim=((4, 256, 256),),
                 compression=None,
                 nilluminations=1, nchannels=1, ntiles=1, nangles=1):
        """Class for writing multiple numpy 3d-arrays into BigDataViewer/BigStitcher HDF5 file.

        Parameters:
            filename: (string) full path to a new file
            subsamp: (tuple of tuples),
                Subsampling levels in (z,y,x) order. Integers >= 1, default value ((1, 1, 1),)
            blockdim: (tuple of tuples),
                Block size for h5 storage, in pixels, in (z,y,x) order. Default ((4,256,256),), see notes.
            compression: str
                (None, 'gzip', 'lzf'), HDF5 compression method. Default is None for high-speed writing.
            nilluminations, nchannels, ntiles, nangles, (int)
                number of view attributes, default 1.

        Notes:
        Input stacks and output files are assumed uint16 type.

        The h5 recommended block (chunk) size should be between 10 KB and 1 MB, larger for large arrays.
        For example, block dimensions (4,256,256)px gives ~0.5MB block size for type int16 (2 bytes) and writes very fast.
        Block size can be larger than stack dimension.
        """
        assert nilluminations >= 1, "Total number of illuminations must be at least 1."
        assert nchannels >= 1, "Total number of channels must be at least 1."
        assert ntiles >= 1, "Total number of tiles must be at least 1."
        assert nangles >= 1, "Total number of angles must be at least 1."
        assert compression in (None, 'gzip', 'lzf'), 'Unknown compression type'
        assert all([isinstance(element, int) for tupl in subsamp for element in
                    tupl]), 'subsamp values should be integers >= 1.'
        if len(blockdim) < len(subsamp):
            print("Number of blockdim levels (" + str (len(blockdim)) +
                  ") is less than subsamp levels (" + str(len(subsamp)) + ")\n" +
                  "First-level block size " + str(blockdim[0]) + " will be used for all levels\n")

        self.nsetups = nilluminations * nchannels * ntiles * nangles
        self.nilluminations = nilluminations
        self.nchannels = nchannels
        self.ntiles = ntiles
        self.nangles = nangles
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
        self.compression = compression
        self.filename = filename
        self.file_object = h5py.File(filename, 'a')
        self._write_setups_header()
        self.virtual_stacks = False
        self.setup_id_present = [[False] * self.nsetups]
        self.__version__ = "2020.03.13"

    def _write_setups_header(self):
        """Write resolutions and subdivisions for all setups into h5 file."""
        for isetup in range(self.nsetups):
            group_name = 's{:02d}'.format(isetup)
            if group_name in self.file_object:
                del self.file_object[group_name]
            grp = self.file_object.create_group(group_name)
            data_subsamp = np.flip(self.subsamp, 1)
            data_chunks = np.flip(self.chunks, 1)
            grp.create_dataset('resolutions', data=data_subsamp, dtype='<f8')
            grp.create_dataset('subdivisions', data=data_chunks, dtype='<i4')

    def append_plane(self, plane, plane_index, time=0, illumination=0, channel=0, tile=0, angle=0):
        """Append a plane to initialized virtual stack. Requires stack initialization by calling e.g.
        append_view(stack=None, virtual_stack_dim=(1000,2048,2048)).
        Parameters:
            plane: numpy array(uint16)
                A 2d array of (y,x) pixel values.
            plane_index: (int)
                Plane z-position in the virtual stack.
            time: (int)
                time index, starting from 0.
            illumination, channel, view, angle: (int)
                indices of the view attributes, starting from 0.
        """
        assert self.virtual_stacks, "Appending planes requires initialization with virtual stack, " \
                                    "see append_view(stack=None,...)"
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        self.update_setup_id_present(isetup, time)
        assert plane.shape == self.stack_shapes[isetup][1:], "Plane dimensions must match (y,x) size of virtual stack."
        assert plane_index < self.stack_shapes[isetup][0], "Plane index must be less than virtual stack z-dimension."
        assert self.nlevels == 1, "No subsampling currently implemented for virtual stack writing."
        fmt = 't{:05d}/s{:02d}/{}'
        for ilevel in range(self.nlevels):
            group_name = fmt.format(time, isetup, ilevel)
            dataset = self.file_object[group_name]["cells"]
            dataset[plane_index, :, :] = plane.astype('int16')

    def append_view(self, stack, virtual_stack_dim=None,
                    time=0, illumination=0, channel=0, tile=0, angle=0,
                    m_affine=None, name_affine='manually defined',
                    voxel_size_xyz=(1, 1, 1), voxel_units='px', calibration=(1, 1, 1),
                    exposure_time=0, exposure_units='s'):
        """Write numpy 3-dimensional array (stack) to h5 file at specified timepint (itime) and setup number (isetup).
        Parameters:
            stack: numpy array (uint16) or None
                A 3-dimensional stack of uint16 data in (z,y,x) axis order.
                If None, creates an empty dataset of size huge_stack_dim.
            virtual_stack_dim: None or tuple of (z,y,x) dimensions, optional
                Dimensions to allocate a huge stack and fill it later by individual planes.
            time: (int)
                time index, starting from 0.
            illumination, channel, view, angle: (int)
                indices of the view attributes, starting from 0.
            m_affine: a (3,4) numpy array, optional
                Coefficients of affine transformation matrix (m00, m01m ...)
            name_affine: str, optional
                Name of affine transformation
            voxel_size_xyz: tuple of 3 elements, optional
                The physical size of voxel, in voxel_units. Default (1, 1, 1).
            voxel_units: str, optional
                spatial units (default is 'px').
            calibration: tuple of 3 elements, optional
                The anisotropy factors for (x,y,z) voxel calibration. Default (1, 1, 1).
                Leave it default unless you are expert in BigStitcher and know how it affects transformations.
            exposure_time: scalar, optional
                Camera exposure time for this view, default 0.
            exposure_units: str, optional
                Time units for this view, default "s".

        """
        assert len(calibration) == 3, "Calibration must be a tuple of 3 elements (x, y, z)."
        assert len(voxel_size_xyz) == 3, "Voxel size must be a tuple of 3 elements (x, y, z)."
        fmt = 't{:05d}/s{:02d}/{}'
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        self.update_setup_id_present(isetup, time)
        if stack is not None:
            assert len(stack.shape) == 3, "Stack should be a 3-dimensional numpy array (z,y,x)"
            self.stack_shapes[isetup] = stack.shape
        else:
            assert len(virtual_stack_dim) == 3, "Stack is virtual, so parameter virtual_stack_dim must be defined."
            self.stack_shapes[isetup] = virtual_stack_dim
            self.virtual_stacks = True

        for ilevel in range(self.nlevels):
            group_name = fmt.format(time, isetup, ilevel)
            if group_name in self.file_object:
                del self.file_object[group_name]
            grp = self.file_object.create_group(group_name)
            if stack is not None:
                subdata = self._subsample_stack(stack, self.subsamp[ilevel]).astype('int16')
                grp.create_dataset('cells', data=subdata, chunks=self.chunks[ilevel],
                                   maxshape=(None, None, None), compression=self.compression, dtype='int16')
            else:  # a huge virtual stack can be initialized
                assert self.nlevels == 1, "No subsampling currently implemented for virtual stacks."
                grp.create_dataset('cells', chunks=self.chunks[ilevel],
                                   shape=virtual_stack_dim, compression=self.compression, dtype='int16')
        if m_affine is not None:
            self.affine_matrices[isetup] = m_affine
            self.affine_names[isetup] = name_affine
        self.calibrations[isetup] = calibration
        self.voxel_size_xyz[isetup] = voxel_size_xyz
        self.voxel_units[isetup] = voxel_units
        self.exposure_time[isetup] = exposure_time
        self.exposure_units[isetup] = exposure_units

    def _compute_chunk_size(self, blockdim):
        """Populate the size of h5 chunks.
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

    def _subsample_stack(self, stack, subsamp_level):
        """Subsampling of 3d stack.
        Parameters:
            stack, numpy 3d array (z,y,x) of int16
            subsamp_level, array-like with 3 elements, eg (2,4,4) for downsampling z(x2), x and y (x4).
        Return:
            down-scaled stack, unit16 type.
        """
        if all(subsamp_level[:] == 1):
            stack_sub = stack
        else:
            stack_sub = skimage.transform.downscale_local_mean(stack, tuple(subsamp_level)).astype(np.uint16)
        return stack_sub

    def write_xml_file(self, ntimes=1,
                       camera_name="default",  microscope_name="default",
                       microscope_version="0.0", user_name="user"):
        """
        Write XML header file for the HDF5 file.

        Parameters:
            ntimes: int
                number of time points
            camera_name: str, optional
                Name of the camera (same for all setups at the moment, ToDo)
            microscope_name: str, optional
            microscope_version: str, optional
            user_name: str, optional
        """
        assert ntimes >= 1, "Total number of time points must be at least 1."
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
        el.text = os.path.basename(self.filename)
        # write ViewSetups
        viewsets = ET.SubElement(seqdesc, 'ViewSetups')
        for iillumination in range(self.nilluminations):
            for ichannel in range(self.nchannels):
                for itile in range(self.ntiles):
                    for iangle in range(self.nangles):
                        isetup = self._determine_setup_id(iillumination, ichannel, itile, iangle)
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

        # write Attributes (range of values)
        attrs_illum = ET.SubElement(viewsets, 'Attributes')
        attrs_illum.set('name', 'illumination')
        for iilumination in range(self.nilluminations):
            illum = ET.SubElement(attrs_illum, 'Illumination')
            ET.SubElement(illum, 'id').text = str(iilumination)
            ET.SubElement(illum, 'name').text = 'illumination ' + str(iilumination)

        attrs_chan = ET.SubElement(viewsets, 'Attributes')
        attrs_chan.set('name', 'channel')
        for ichannel in range(self.nchannels):
            chan = ET.SubElement(attrs_chan, 'Channel')
            ET.SubElement(chan, 'id').text = str(ichannel)
            ET.SubElement(chan, 'name').text = 'channel ' + str(ichannel)

        attrs_tile = ET.SubElement(viewsets, 'Attributes')
        attrs_tile.set('name', 'tile')
        for itile in range(self.ntiles):
            tile = ET.SubElement(attrs_tile, 'Tile')
            ET.SubElement(tile, 'id').text = str(itile)
            ET.SubElement(tile, 'name').text = 'tile ' + str(itile)

        attrs_ang = ET.SubElement(viewsets, 'Attributes')
        attrs_ang.set('name', 'angle')
        for iangle in range(self.nangles):
            ang = ET.SubElement(attrs_ang, 'Angle')
            ET.SubElement(ang, 'id').text = str(iangle)
            ET.SubElement(ang, 'name').text = 'angle ' + str(iangle)

        # Time points
        tpoints = ET.SubElement(seqdesc, 'Timepoints')
        tpoints.set('type', 'range')
        ET.SubElement(tpoints, 'first').text = str(0)
        ET.SubElement(tpoints, 'last').text = str(ntimes - 1)

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
        for itime in range(ntimes):
            for iset in range(self.nsetups):
                vreg = ET.SubElement(vregs, 'ViewRegistration')
                vreg.set('timepoint', str(itime))
                vreg.set('setup', str(iset))
                # write arbitrary affine transformation, specific for each view
                if iset in self.affine_matrices.keys():
                    vt = ET.SubElement(vreg, 'ViewTransform')
                    vt.set('type', 'affine')
                    ET.SubElement(vt, 'Name').text = self.affine_names[iset]
                    n_prec = 6
                    mx_string = np.array2string(self.affine_matrices[iset].flatten(), separator=' ',
                                                precision=n_prec, floatmode='fixed',
                                                max_line_width=(n_prec+5)*4)
                    ET.SubElement(vt, 'affine').text = mx_string[1:-1].strip()

                # write registration transformation (calibration)
                vt = ET.SubElement(vreg, 'ViewTransform')
                vt.set('type', 'affine')
                ET.SubElement(vt, 'Name').text = 'calibration'
                calx, caly, calz = self.calibrations[iset]
                ET.SubElement(vt, 'affine').text = \
                    '{} 0.0 0.0 0.0 0.0 {} 0.0 0.0 0.0 0.0 {} 0.0'.format(calx, caly, calz)

        self._xml_indent(root)
        tree = ET.ElementTree(root)
        tree.write(os.path.splitext(self.filename)[0] + ".xml", xml_declaration=True, encoding='utf-8', method="xml")
        return

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

    def _determine_setup_id(self, illumination=0, channel=0, tile=0, angle=0):
        """Takes the view attributes (illumination, channel, tile, angle) and converts them into unique setup_id.
        Parameters:
            illumination (int) >=0
            channel (int) >= 0
            tile (int) >= 0
            angle (int) >= 0
        Returns
            setup_id (int), starting from 0 (first setup)
            """
        setup_id_matrix = np.arange(self.nsetups)
        setup_id_matrix = setup_id_matrix.reshape((self.nilluminations, self.nchannels, self.ntiles, self.nangles))
        setup_id = setup_id_matrix[illumination, channel, tile, angle]
        return setup_id

    def update_setup_id_present(self, isetup, itime):
        """Update the lookup table (list of lists) for missing setups"""
        if len(self.setup_id_present) <= itime:
            self.setup_id_present.append([False] * self.nsetups)
        self.setup_id_present[itime][isetup] = True

    def close(self):
        """Close the file object."""
        self.file_object.close()


class BdvReader:
    def __init__(self, filename_h5=None):
        """Class for reading a BigDataViewer/BigStitcher HDF5 file into numpy array.

        Constructor parameters
            filename_h5: (string), optional, full path to a HDF5 file (default None).
        """
        self.__version__ = "2020.01.07"
        self.__fmt = 't{:05d}/s{:02d}/{}'
        self.filename_h5 = filename_h5
        if filename_h5 is not None:
            assert os.path.exists(filename_h5), "Error: HDF5 file not found"
            self.__file_object = h5py.File(filename_h5, 'r')
        else:
            self.__file_object = None

    def set_path_h5(self, filename_h5):
        """Set the file path to HDF5 file. If another file was already open, it closes it before proceeding"""
        assert os.path.exists(filename_h5), "Error: HDF5 file not found"
        if self.__file_object is not None:
            self.__file_object.close()
        self.__file_object = h5py.File(filename_h5, 'r')

    def read_view(self, time=0, isetup=0, ilevel=0):
        """Read a view (stack) specified by its time, setup ID, and downsampling level.

        Parameters
            time: (int) index of time point (default 0).
            isetup: (int), index of setup ID (default 0)
            ilevel: (int), level of subsampling, if available (default 0, no subsampling)

        Returns
            dataset: a numpy array (ndim=3)"""
        group_name = self.__fmt.format(time, isetup, ilevel)
        dataset = self.__file_object[group_name]["cells"]
        return dataset

    def close(self):
        """Close the file object."""
        self.__file_object.close()
