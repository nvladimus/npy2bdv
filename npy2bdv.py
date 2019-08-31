# Fast writing of numpy arrays to HDF5 format compatible with Fiji/BigDataViewer and BigStitcher
# Inspired by https://github.com/tlambert03/imarispy and Adam Glaser code
# Author: Nikita Vladimirov
# MIT license
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
        assert not os.path.exists(filename), "File already exists, writing terminated"
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
        self.chunks = self.compute_chunk_size(blockdim)
        self.stack_shape = None
        self.compression = compression
        self.filename = filename
        self.file_object = h5py.File(filename, 'a')
        self.write_setups_header()
        self.__version__ = "2019.08.31"

    def write_setups_header(self):
        """Write resolutions and subdivisions for all setups into h5 file."""
        for isetup in range(self.nsetups):
            grp = self.file_object.create_group('s{:02d}'.format(isetup))
            data_subsamp = np.flip(self.subsamp, 1)
            data_chunks = np.flip(self.chunks, 1)
            grp.create_dataset('resolutions', data=data_subsamp, dtype='<f8')
            grp.create_dataset('subdivisions', data=data_chunks, dtype='<i4')

    def append_view(self, stack, time, illumination=0, channel=0, tile=0, angle=0):
        """Write numpy 3-dimensional array (stack) to h5 file at specified timepint (itime) and setup number (isetup).
        Parameters:
            stack: 3d numpy array in (z,y,x) order, type 'uint16'.
            time: (int) time index, starting from 0.
            illumination, channel, view, angle: (int) view attributes, starting from 0.
        """
        assert len(stack.shape) == 3, "Stack should be a 3-dimensional numpy array (z,y,x)"
        self.stack_shape = stack.shape
        fmt = 't{:05d}/s{:02d}/{}'
        nlevels = len(self.subsamp)
        isetup = self.determine_setup_id(illumination, channel, tile, angle)
        for ilevel in range(nlevels):
            grp = self.file_object.create_group(fmt.format(time, isetup, ilevel))
            subdata = self.subsample_stack(stack, self.subsamp[ilevel])
            grp.create_dataset('cells', data=subdata, chunks=self.chunks[ilevel],
                               maxshape=(None, None, None), compression=self.compression)

    def compute_chunk_size(self, blockdim):
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

    def subsample_stack(self, stack, subsamp_level):
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

    def write_xml_file(self, ntimes=1, units='px', dx=1, dy=1, dz=1,
                       m_affine=None, name_affine="Manually defined (Rigid/Affine by matrix)",
                       camera_name="default", exposure_time=None, exposure_units="s",
                       microscope_name="default", microscope_version="0.0", user_name="default"):
        """
        Write XML header file for the HDF5 file.

        Parameters:
            ntimes: int
                number of time points
            units: str, optional
                spatial units (default is 'px').
            dx, dy, dz: scalars, optional
                voxel dimensions in 'units'.
            m_affine: a (3,4) numpy array, optional
                Coefficients of affine transformation matrix (m00, m01m ...)
            name_affine: str, optional
                Name of affine transformation
            camera_name: str, optional
                Name of the camera (same for all setups at the moment, ToDo)
            exposure_time: scalar, optional
                Camera exposure time (same for all setups at the moment, ToDo)
            exposure_units: str, optional
                Time units, default "s".
            microscope_name: str, optional
            microscope_version: str, optional
            user_name: str, optional

        """
        assert ntimes >= 1, "Total number of time points must be at least 1."
        nz, ny, nx = tuple(self.stack_shape)
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
                        isetup = self.determine_setup_id(iillumination, ichannel, itile, iangle)
                        vs = ET.SubElement(viewsets, 'ViewSetup')
                        ET.SubElement(vs, 'id').text = str(isetup)
                        ET.SubElement(vs, 'name').text = 'setup ' + str(isetup)
                        ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)
                        vox = ET.SubElement(vs, 'voxelSize')
                        ET.SubElement(vox, 'unit').text = units
                        ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)
                        # new XML data, added by @nvladimus
                        cam = ET.SubElement(vs, 'camera')
                        ET.SubElement(cam, 'name').text = camera_name
                        ET.SubElement(cam, 'exposureTime').text = '{}'.format(exposure_time)
                        ET.SubElement(cam, 'exposureUnits').text = exposure_units
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

        # Transformations of coordinate system
        vregs = ET.SubElement(root, 'ViewRegistrations')

        for itime in range(ntimes):
            for iset in range(self.nsetups):
                vreg = ET.SubElement(vregs, 'ViewRegistration')
                vreg.set('timepoint', str(itime))
                vreg.set('setup', str(iset))
                # write arbitrary affine transformation
                if m_affine is not None:
                    vt = ET.SubElement(vreg, 'ViewTransform')
                    vt.set('type', 'affine')
                    ET.SubElement(vt, 'Name').text = name_affine
                    n_prec = 6
                    mx_string = np.array2string(m_affine.flatten(), separator=' ',
                                                precision=n_prec, floatmode='fixed',
                                                max_line_width=(n_prec+5)*4)
                    ET.SubElement(vt, 'affine').text = mx_string[2:-1]

                # write registration transformation (calibration)
                vt = ET.SubElement(vreg, 'ViewTransform')
                vt.set('type', 'affine')
                ET.SubElement(vt, 'Name').text = 'calibration'
                ET.SubElement(vt, 'affine').text = '{} 0.0 0.0 0.0 0.0 {} 0.0 0.0 0.0 0.0 {} 0.0'.format(dx, dy, dz)

        self.xml_indent(root)
        tree = ET.ElementTree(root)
        tree.write(os.path.splitext(self.filename)[0] + ".xml", xml_declaration=True, encoding='utf-8', method="xml")
        return

    def xml_indent(self, elem, level=0):
        """Pretty printing function"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.xml_indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def determine_setup_id(self, illumination=0, channel=0, tile=0, angle=0):
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

    def close(self):
        """Close the file object."""
        self.file_object.close()

