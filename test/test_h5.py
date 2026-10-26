# python -m test.test_h5
import npy2bdv
import os
import shutil
import unittest
import warnings
import numpy as np
import skimage.transform
from .sample import generate_test_image

TOY_DATASET_PATH = './test/toy_datasets/'
FILE_EXTENSION = 'h5' # set either 'h5' or 'xml'. This should be parsed correctly either way.

np.random.seed(1)

class TestReadWrite(unittest.TestCase):
    """Write a dataset with multiples views, and load it back. Compare the loaded dataset vs expectations.
    """
    def setUp(self) -> None:
        """"This will automatically call for EVERY single test we run."""
        self.test_dir = "./test/test_files/"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.fname = self.test_dir + "test_real_stack." + FILE_EXTENSION
        self.fname1 = self.test_dir + "test_real_subsampling_dims_odd." + FILE_EXTENSION
        self.NZ, self.NY, self.NX = 16, 64, 64
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 2, 2, 2, 3, 2
        self.N_VIEWS = self.N_T*self.N_CH*self.N_ILL*self.N_TILES*self.N_ANGLES
        self.affine = np.random.uniform(0, 1, (3, 4))
        self.probe_t_ch_ill_tile_angle = (0, 1, 1, 0, 1) # pick a random index of a view to probe
        self.subsamp = ((1, 1, 1), (2, 4, 4),)  # OPTIONAL param
        self.blockdim = ((8, 32, 32), (4, 8, 8),) # OPTIONAL param
        self.stacks = []
        # generate random views (stacks)
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            stack = np.empty((self.NZ, self.NY, self.NX), "uint16")
                            peak = np.random.randint(100, 65535)
                            for z in range(self.NZ):
                                stack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ, peak=peak)
                            self.stacks.append(stack)

        bdv_writer = npy2bdv.BdvWriter(self.fname,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES,
                                       )
        i = 0
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            bdv_writer.append_view(self.stacks[i], time=t,
                                                   channel=i_ch,
                                                   illumination=i_illum,
                                                   tile=i_tile,
                                                   angle=i_angle,
                                                   voxel_size_xyz=(1, 1, 4))
                            i += 1
        bdv_writer.create_pyramids(subsamp=self.subsamp[1:], blockdim=self.blockdim[1:])
        bdv_writer.write_xml()
        bdv_writer.append_affine(self.affine, 'test affine transform', *self.probe_t_ch_ill_tile_angle)
        bdv_writer.close()

    def test_pixels(self):
        """Check if the reader imports full uint16 range correctly"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        editor = npy2bdv.BdvEditor(self.fname)
        i = 0
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            view = editor.read_view(time=t,
                                                    channel=i_ch,
                                                    illumination=i_illum,
                                                    tile=i_tile,
                                                    angle=i_angle)
                            self.assertTrue((view == self.stacks[i]).all(), "Written stack differs from the loaded one.")
                            i += 1
        editor.finalize()
        
    def test_read_z_plane(self):
        """Check if the reader imports each z slice correctly"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        editor = npy2bdv.BdvEditor(self.fname,mode='r')
        i = 0
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            for z_plane in range(self.NZ):
                                view = editor.read_view(time=t,
                                                        channel=i_ch,
                                                        illumination=i_illum,
                                                        tile=i_tile,
                                                        angle=i_angle,
                                                        z=z_plane)
                                self.assertTrue((view == self.stacks[i][z_plane]).all(), "Written stack differs from the loaded one.")
                            i += 1
        editor.finalize()
        
    def test_view_properties(self):
        """"BdvReader(): does the meta-info in XML file have expected values?"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        editor = npy2bdv.BdvEditor(self.fname)
        ntimes, nilluminations, nchannels, ntiles, nangles = editor.get_attribute_count()
        self.assertEqual(ntimes, self.N_T, f"ntimes is incorrect: {ntimes}.")
        self.assertEqual(nilluminations, self.N_ILL, f"nilluminations is incorrect: {nilluminations}.")
        self.assertEqual(nchannels, self.N_CH, f"nchannels is incorrect: {nchannels}.")
        self.assertEqual(ntiles, self.N_TILES, f"ntiles is incorrect: {ntiles}.")
        self.assertEqual(nangles, self.N_ANGLES, f"nangles is incorrect: {nangles}.")

        i = 0
        for i_ch in range(self.N_CH):
            for i_illum in range(self.N_ILL):
                for i_tile in range(self.N_TILES):
                    for i_angle in range(self.N_ANGLES):
                        vox_size = editor.get_view_property("voxel_size",
                                                            channel=i_ch,
                                                            illumination=i_illum,
                                                            tile=i_tile,
                                                            angle=i_angle)
                        view_shape = editor.get_view_property("view_shape",
                                                              channel=i_ch,
                                                              illumination=i_illum,
                                                              tile=i_tile,
                                                              angle=i_angle)
                        self.assertEqual(vox_size, (1, 1, 4), f"Voxel size is incorrect: {vox_size}.")
                        self.assertEqual(view_shape, self.stacks[i].shape[::-1], f"View shape incorrect: {view_shape}.")
                        i += 1
        affine_read = editor.read_affine(*self.probe_t_ch_ill_tile_angle, index=0)
        self.assertAlmostEqual((affine_read - self.affine).sum(), 0, places=4,
                               msg=f"Affine matrix incorrect: {affine_read} vs {self.affine}.")
        editor.finalize()

    def test_cropping(self):
        """"BdvEditor: crop a view in-place for all time points,
         and check if new H5 view size matches the XML view size."""
        editor = npy2bdv.BdvEditor(self.fname)
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            original_view = editor.read_view(time=t,
                                                            channel=i_ch,
                                                            illumination=i_illum,
                                                            tile=i_tile,
                                                            angle=i_angle)
                            h5_shape_before = original_view.shape
                            xml_shape_before = editor.get_view_property(key='view_shape',
                                                                        channel=i_ch,
                                                                        illumination=i_illum,
                                                                        tile=i_tile,
                                                                        angle=i_angle)
                            editor.crop_view(bbox_xyz=((3, -3), (2, -2), (1, -1)),
                                             channel=i_ch,
                                             illumination=i_illum,
                                             tile=i_tile,
                                             angle=i_angle)
                            cropped_view = editor.read_view(time=t,
                                                            channel=i_ch,
                                                            illumination=i_illum,
                                                            tile=i_tile,
                                                            angle=i_angle)
                            h5_shape_after = cropped_view.shape
                            xml_shape_after = editor.get_view_property(key='view_shape',
                                                                        channel=i_ch,
                                                                        illumination=i_illum,
                                                                        tile=i_tile,
                                                                        angle=i_angle)
                            h5_shape_expected = tuple(np.subtract(h5_shape_before, (2, 4, 6)))
                            xml_shape_expected = tuple(np.subtract(xml_shape_before[::-1], (2, 4, 6)))
                            self.assertEqual(h5_shape_before, xml_shape_before[::-1],
                                             f"View shapes before crop mismatch between H5 and XML:"
                                             f" {h5_shape_before} vs {xml_shape_before[::-1]}")
                            self.assertEqual(h5_shape_after, xml_shape_after[::-1],
                                             f"View shapes after crop mismatch between H5 and XML:"
                                             f" {h5_shape_after} vs {xml_shape_after[::-1]}")
                            self.assertEqual(h5_shape_after, h5_shape_expected,
                                             f"View shapes  mismatch in H5 before and after crop:"
                                             f"{h5_shape_after} vs {h5_shape_expected}")
                            self.assertEqual(xml_shape_after[::-1], xml_shape_expected,
                                             f"View shapes  mismatch in XML before and after crop:"
                                             f"{xml_shape_after[::-1]} vs {xml_shape_expected}")
                            self.assertTrue((original_view[1:-1, 2:-2, 3:-3] == cropped_view).all(),
                                            "Cropped view is not equal the substack of original view")
        editor.finalize()

    def test_write_odd_dims(self):
        img = np.zeros((33, 301, 299))
        bdv_writer = npy2bdv.BdvWriter(self.fname1, nchannels=1, subsamp=((1, 2, 2), (1, 8, 8), (1, 16, 16)),
                                       blockdim=((8, 16, 16), (8, 16, 16), (8, 16, 16)), overwrite=True)
        bdv_writer.append_view(stack=img, channel=0)
        bdv_writer.write_xml()
        bdv_writer.close()

    def test_write_sparse_timepoints(self):
        """Time points can be skipped or written out of order"""
        fname = self.test_dir + "test_sparse_timepoints." + FILE_EXTENSION
        stack = np.arange(4 * 8 * 8, dtype=np.uint16).reshape((4, 8, 8))
        bdv_writer = npy2bdv.BdvWriter(fname, nchannels=1, overwrite=True)
        for t in (2, 0):
            bdv_writer.append_view(stack + t, time=t)
        bdv_writer.write_xml()
        bdv_writer.close()

        editor = npy2bdv.BdvEditor(fname)
        self.assertEqual(editor.ntimes, 3)
        for t in (2, 0):
            self.assertTrue(np.array_equal(editor.read_view(time=t), stack + t))
        editor.finalize()
        with open(self.test_dir + "test_sparse_timepoints.xml") as f:
            self.assertIn('<MissingView timepoint="1" setup="0" />', f.read())

    def tearDown(self) -> None:
        """"Tidies up after EACH test method has been run."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


class TestReadWriteVirtual(unittest.TestCase):
    """Write virtual stacks. """
    def setUp(self) -> None:
        """"This will automatically call for EVERY single test we run."""
        self.test_dir = "./test/test_files/"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.fname0 = self.test_dir + "test_virtual_by_plane." + FILE_EXTENSION
        self.fname1 = self.test_dir + "test_virtual_by_substack." + FILE_EXTENSION
        self.fname2 = self.test_dir + "test_virtual_subsampling_dims_odd." + FILE_EXTENSION

        self.NZ, self.NY, self.NX = 8, 256, 256
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 2, 2, 2, 3, 2
        self.N_VIEWS = self.N_T * self.N_CH * self.N_ILL * self.N_TILES * self.N_ANGLES
        self.N_SUBSTACKS = 4

        self.stacks = []
        for i_stack in range(self.N_VIEWS):
            peak = np.random.randint(100, 65535)
            stack = np.empty((self.NZ, self.NY, self.NX), "uint16")
            for z in range(self.NZ):
                stack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ, peak=peak)
            self.stacks.append(stack)

        self.write_virtual_by_plane()
        self.write_virtual_by_substack()

    def write_virtual_by_plane(self):
        bdv_writer = npy2bdv.BdvWriter(self.fname0,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES)
        i = 0
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            bdv_writer.append_view(stack=None,
                                                   virtual_stack_dim=(self.NZ, self.NY, self.NX),
                                                   time=t,
                                                   channel=i_ch,
                                                   illumination=i_illum,
                                                   tile=i_tile,
                                                   angle=i_angle)
                            for iz in range(self.NZ):
                                bdv_writer.append_plane(plane=self.stacks[i][iz, :, :],
                                                        z=iz,
                                                        time=t, channel=i_ch, illumination=i_illum,
                                                        tile=i_tile, angle=i_angle)
                            i += 1
        bdv_writer.write_xml()
        bdv_writer.close()

    def test_write_virtual_odd_dims_by_plane(self):
        img = np.zeros((33, 301, 299))
        nz, ny, nx = img.shape

        bdv_writer = npy2bdv.BdvWriter(self.fname2, nchannels=1, subsamp=((1, 2, 2), (1, 8, 8), (1, 16, 16)),
                                       blockdim=((8, 16, 16), (8, 16, 16), (8, 16, 16)),
                                       compression='gzip', overwrite=True)
        bdv_writer.append_view(stack=None, virtual_stack_dim=(nz, ny, nx), channel=0)
        for iz in range(nz):
            bdv_writer.append_plane(plane=img[iz, :, :], z=iz)
        bdv_writer.write_xml()
        bdv_writer.close()

    def write_virtual_by_substack(self):
        ''''Write a virtual stack by substacks, with downsampling pyramid.'''
        bdv_writer = npy2bdv.BdvWriter(self.fname1,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES,
                                       subsamp=((1, 1, 1), (2, 4, 4)),
                                       blockdim=((4, 16, 16), (4, 16, 16)))
        # Initialize virtual stacks
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            bdv_writer.append_view(stack=None,
                                                   virtual_stack_dim=(self.NZ, self.NY, self.NX),
                                                   time=t, channel=i_ch, illumination=i_illum,
                                                   tile=i_tile, angle=i_angle)
        # Populate the virtual stacks
        y_offset = 16
        i = 0
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            for isub in range(self.N_SUBSTACKS):
                                zslice = slice(isub*(self.NZ//self.N_SUBSTACKS), (isub+1)*(self.NZ//self.N_SUBSTACKS))
                                # first half-stack
                                bdv_writer.append_substack(substack=self.stacks[i][zslice, :y_offset, :],
                                                           z_start=zslice.start, y_start=0, x_start=0,
                                                           time=t, channel=i_ch, illumination=i_illum,
                                                           tile=i_tile, angle=i_angle)
                                # second half-stack
                                bdv_writer.append_substack(substack=self.stacks[i][zslice, y_offset:, :],
                                                           z_start=zslice.start, y_start=y_offset, x_start=0,
                                                           time=t, channel=i_ch, illumination=i_illum,
                                                           tile=i_tile, angle=i_angle)
                            i += 1
        bdv_writer.write_xml()
        bdv_writer.close()

    def compare_pixels(self, filename):
        editor = npy2bdv.BdvEditor(filename)
        i_stack = 0
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            view = editor.read_view(time=t, channel=i_ch, illumination=i_illum,
                                                    tile=i_tile, angle=i_angle)
                            self.assertTrue((view == self.stacks[i_stack]).all(),
                                            "Written stack differs from the loaded one.")
                            i_stack += 1
        editor.finalize()

    def test_read_voxels(self):
        """"Read the voxel values back and compare to expected values"""
        self.compare_pixels(self.fname0)
        self.compare_pixels(self.fname1)

    def test_substack_pyramid_matches_view(self):
        """Substacks that do not line up with the subsampling factors give the same pyramid as the whole view"""
        subsamp = ((1, 1, 1), (4, 2, 2))
        blockdim = ((2, 16, 16), (2, 16, 16))
        stack = self.stacks[0]
        fname_view = self.test_dir + "test_pyramid_view." + FILE_EXTENSION
        fname_sub = self.test_dir + "test_pyramid_substacks." + FILE_EXTENSION
        bdv_writer = npy2bdv.BdvWriter(fname_view, subsamp=subsamp, blockdim=blockdim)
        bdv_writer.append_view(stack)
        bdv_writer.write_xml()
        bdv_writer.close()
        bdv_writer = npy2bdv.BdvWriter(fname_sub, subsamp=subsamp, blockdim=blockdim)
        bdv_writer.append_view(stack=None, virtual_stack_dim=stack.shape)
        for z_start, z_stop in ((0, 3), (3, 6), (6, 8)):
            for y_start, y_stop in ((101, 256), (0, 101)):
                bdv_writer.append_substack(stack[z_start:z_stop, y_start:y_stop],
                                           z_start=z_start, y_start=y_start)
        bdv_writer.write_xml()
        bdv_writer.close()
        editor_view = npy2bdv.BdvEditor(fname_view)
        editor_sub = npy2bdv.BdvEditor(fname_sub)
        for ilevel in range(len(subsamp)):
            self.assertTrue(np.array_equal(editor_view.read_view(ilevel=ilevel),
                                           editor_sub.read_view(ilevel=ilevel)),
                            f"Pyramid level {ilevel} differs.")
        editor_view.finalize()
        editor_sub.finalize()

    def tearDown(self) -> None:
        """"Tidies up after EACH test method has been run."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


class TestDtypes(unittest.TestCase):
    """Write and read back uint8 and float32 views (issue #10): in memory, virtual, and with lazy pyramids."""
    def setUp(self) -> None:
        self.test_dir = "./test/test_files_dtypes/"
        os.makedirs(self.test_dir, exist_ok=True)
        self.subsamp = ((1, 1, 1), (2, 2, 2))
        self.blockdim = ((4, 16, 16), (4, 16, 16))
        rng = np.random.default_rng(0)
        self.stacks = {'uint8': rng.integers(0, 256, (8, 32, 33), dtype=np.uint8),
                       'float32': rng.normal(0, 1000, (8, 32, 33)).astype(np.float32)}

    def expected_level(self, stack, ilevel):
        if ilevel == 0:
            return stack
        return skimage.transform.downscale_local_mean(stack, self.subsamp[ilevel]).astype(stack.dtype)

    def check_file(self, fname, stack):
        dtype = stack.dtype.name
        editor = npy2bdv.BdvEditor(fname)
        self.assertEqual(editor.dtype, stack.dtype)
        self.assertEqual(editor._file_object_h5['s00'].attrs['dataType'], dtype.encode())
        for ilevel in range(len(self.subsamp)):
            view = editor.read_view(ilevel=ilevel)
            self.assertEqual(view.dtype, stack.dtype)
            self.assertEqual(editor._file_object_h5[f't00000/s00/{ilevel}/cells'].dtype, stack.dtype)
            self.assertTrue(np.array_equal(view, self.expected_level(stack, ilevel)),
                            f"{dtype}: level {ilevel} differs.")
        editor.finalize()

    def test_append_view(self):
        for dtype, stack in self.stacks.items():
            with self.subTest(dtype=dtype):
                fname = self.test_dir + f"view_{dtype}.h5"
                writer = npy2bdv.BdvWriter(fname, subsamp=self.subsamp, blockdim=self.blockdim, dtype=dtype)
                writer.append_view(stack)
                writer.write_xml()
                writer.close()
                self.check_file(fname, stack)

    def test_virtual_stack_by_plane(self):
        for dtype, stack in self.stacks.items():
            with self.subTest(dtype=dtype):
                fname = self.test_dir + f"virtual_{dtype}.h5"
                subsamp = ((1, 1, 1), (1, 2, 2))
                writer = npy2bdv.BdvWriter(fname, subsamp=subsamp, blockdim=self.blockdim, dtype=dtype)
                writer.append_view(stack=None, virtual_stack_dim=stack.shape)
                for z in range(stack.shape[0]):
                    writer.append_plane(stack[z], z=z)
                writer.write_xml()
                writer.close()
                editor = npy2bdv.BdvEditor(fname)
                self.assertTrue(np.array_equal(editor.read_view(), stack))
                expected = skimage.transform.downscale_local_mean(stack, (1, 2, 2)).astype(stack.dtype)
                self.assertTrue(np.array_equal(editor.read_view(ilevel=1), expected))
                editor.finalize()

    def test_create_pyramids(self):
        for dtype, stack in self.stacks.items():
            with self.subTest(dtype=dtype):
                fname = self.test_dir + f"pyramids_{dtype}.h5"
                writer = npy2bdv.BdvWriter(fname, blockdim=(self.blockdim[0],), dtype=dtype)
                writer.append_view(stack)
                writer.create_pyramids(subsamp=(self.subsamp[1],), blockdim=(self.blockdim[1],), compression='gzip')
                writer.write_xml()
                writer.close()
                self.check_file(fname, stack)
                editor = npy2bdv.BdvEditor(fname, mode='r')
                self.assertEqual(editor._file_object_h5['t00000/s00/1/cells'].compression, 'gzip')
                editor.finalize()

    def test_uint16_is_legacy_int16(self):
        """Default uint16 stays stored as int16 without a dataType attribute, so older Fiji can read it."""
        fname = self.test_dir + "legacy_uint16.h5"
        stack = np.full((2, 4, 4), 60000, dtype=np.uint16)
        writer = npy2bdv.BdvWriter(fname)
        writer.append_view(stack)
        writer.write_xml()
        writer.close()
        editor = npy2bdv.BdvEditor(fname)
        self.assertNotIn('dataType', editor._file_object_h5['s00'].attrs)
        self.assertEqual(editor._file_object_h5['t00000/s00/0/cells'].dtype, np.int16)
        self.assertTrue(np.array_equal(editor.read_view(), stack))
        editor.finalize()

    def test_unsupported_dtype(self):
        with self.assertRaises(AssertionError):
            npy2bdv.BdvWriter(self.test_dir + "bad.h5", dtype='uint32')

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class TestInputChecks(unittest.TestCase):
    """Warnings on lossy casts, and errors on inconsistent shapes within a setup."""
    def setUp(self) -> None:
        self.test_dir = "./test/test_files_checks/"
        os.makedirs(self.test_dir, exist_ok=True)

    def writer(self, name, **kwargs):
        return npy2bdv.BdvWriter(self.test_dir + name + ".h5", overwrite=True, **kwargs)

    def test_lossy_cast_warns(self):
        cases = [('uint16', np.full((2, 4, 4), 70000, dtype=np.int64)),     # out of range
                 ('uint16', np.full((2, 4, 4), 1.5)),                         # fractions
                 ('uint8', np.full((2, 4, 4), 300, dtype=np.uint16)),
                 ('float32', np.full((2, 4, 4), 2**24 + 1, dtype=np.int64))]  # precision
        for dtype, stack in cases:
            with self.subTest(dtype=dtype, input=stack.dtype.name):
                writer = self.writer("lossy", dtype=dtype)
                with self.assertWarns(UserWarning):
                    writer.append_view(stack)
                writer.close()

    def test_lossless_cast_is_silent(self):
        cases = [('uint16', np.full((2, 4, 4), 65535, dtype=np.uint16)),
                 ('uint16', np.full((2, 4, 4), 1000, dtype=np.int64)),   # e.g. np.random.randint default
                 ('uint16', np.full((2, 4, 4), 1000.0)),
                 ('uint8', np.full((2, 4, 4), 255, dtype=np.uint16)),
                 ('float32', np.full((2, 4, 4), np.nan))]
        for dtype, stack in cases:
            with self.subTest(dtype=dtype, input=stack.dtype.name):
                writer = self.writer("lossless", dtype=dtype)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    writer.append_view(stack)
                writer.close()

    def test_lossy_plane_and_substack_warn(self):
        writer = self.writer("lossy_virtual", dtype='uint8')
        writer.append_view(stack=None, virtual_stack_dim=(4, 8, 8))
        with self.assertWarns(UserWarning):
            writer.append_plane(np.full((8, 8), 256, dtype=np.uint16), z=0)
        with self.assertWarns(UserWarning):
            writer.append_substack(np.full((2, 8, 8), -1, dtype=np.int16), z_start=1)
        writer.close()

    def test_shape_change_within_setup_raises(self):
        writer = self.writer("shapes", nchannels=2)
        writer.append_view(np.zeros((4, 8, 8), np.uint16), time=0, channel=0)
        writer.append_view(np.zeros((6, 8, 8), np.uint16), time=0, channel=1)   # other setup: fine
        writer.append_view(np.zeros((5, 8, 8), np.uint16), time=0, channel=0)   # rewrite same view: fine
        with self.assertRaises(ValueError):
            writer.append_view(np.zeros((4, 8, 8), np.uint16), time=1, channel=0)
        with self.assertRaises(ValueError):
            writer.append_view(stack=None, virtual_stack_dim=(4, 8, 8), time=1, channel=0)
        writer.append_view(np.zeros((5, 8, 8), np.uint16), time=1, channel=0)   # same shape: fine
        writer.write_xml()
        writer.close()
        editor = npy2bdv.BdvEditor(self.test_dir + "shapes.xml", mode='r')
        self.assertEqual(editor.get_view_property('view_shape', channel=0), (8, 8, 5))
        self.assertEqual(editor.ntimes, 2)
        editor.finalize()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)


class TestToyDatasets(unittest.TestCase):
    """Test on existing (small) datasets.
    """

    def test_kidney(self):
        fname_xml = TOY_DATASET_PATH + 'kidney/kidney.xml'
        fname_xml1 = TOY_DATASET_PATH + 'kidney/kidney_v1.xml'
        fname_h5 = TOY_DATASET_PATH + 'kidney/kidney.h5'

        for fname in (fname_xml, fname_xml1, fname_h5):
            assert os.path.exists(fname), f'File {fname} not found.'

        editor = npy2bdv.BdvEditor(fname_xml)
        editor1 = npy2bdv.BdvEditor(fname_xml1)
        view, view1 = editor.read_view(), editor1.read_view()
        self.assertTrue((view == view1).all(), f"Views loaded using files {fname_xml}, {fname_xml1} are different.")


if __name__ == '__main__':
    unittest.main()

