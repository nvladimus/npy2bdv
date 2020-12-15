# python -m test.test_h5
import npy2bdv
import os
import shutil
import unittest
import numpy as np
from .sample import generate_test_image


class TestReadWrite(unittest.TestCase):
    """Write a dataset with multiples views, and load it back. Compare the loaded dataset vs expetations.
    """
    def setUp(self) -> None:
        """"This will automatically call for EVERY single test we run."""
        self.test_dir = "./test/test_files/"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.fname = self.test_dir + "test_real_stack.h5"
        self.NZ, self.NY, self.NX = 8, 35, 35 # XY dims must be odd to get nominal 65535 peak value.
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 2, 2, 4, 6, 4
        
        self.stack = np.empty((self.NZ, self.NY, self.NX), "uint16")
        for z in range(self.NZ):
            self.stack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ)

        bdv_writer = npy2bdv.BdvWriter(self.fname,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES)
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            bdv_writer.append_view(self.stack, time=t,
                                                   channel=i_ch,
                                                   illumination=i_illum,
                                                   tile=i_tile,
                                                   angle=i_angle,
                                                   voxel_size_xyz=(1, 1, 4))
        bdv_writer.write_xml_file(ntimes=self.N_T)
        bdv_writer.close()

    def test_range_uint16(self):
        """Check if the reader imports full uint16 range correctly"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        editor = npy2bdv.BdvEditor(self.fname)
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
                            self.assertTrue(view.min() >= 0, f"Min() value incorrect: {view.min()}")
                            self.assertTrue(view.max() == 65535, f"Max() value incorrect: {view.max()}")
                            self.assertTrue((view == self.stack).all(), "Written stack differs from the loaded stack.")
        editor.finalize()

    def test_view_properties(self):
        """"BdvReader(): does the meta-info in XML file have expected values?"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        editor = npy2bdv.BdvEditor(self.fname)
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
                        self.assertEqual(view_shape, self.stack.shape[::-1], f"View shape incorrect: {view_shape}.")
        editor.finalize()

    def test_attribute_counts(self):
        """"BdvEditor(): do the attribute total counts have expected values?"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        editor = npy2bdv.BdvEditor(self.fname)
        ntimes, nilluminations, nchannels, ntiles, nangles = editor.get_attribute_count()
        self.assertEqual(ntimes, self.N_T, f"ntimes is incorrect: {ntimes}.")
        self.assertEqual(nilluminations, self.N_ILL, f"nilluminations is incorrect: {nilluminations}.")
        self.assertEqual(nchannels, self.N_CH, f"nchannels is incorrect: {nchannels}.")
        self.assertEqual(ntiles, self.N_TILES, f"ntiles is incorrect: {ntiles}.")
        self.assertEqual(nangles, self.N_ANGLES, f"nangles is incorrect: {nangles}.")
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
        self.fname0 = self.test_dir + "test_virtual_by_plane.h5"
        self.fname1 = self.test_dir + "test_virtual_by_substack.h5"

        self.NZ, self.NY, self.NX = 8, 256, 256  # XY dims must be odd to get nominal 65535 peak value.
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 2, 2, 2, 2, 2
        self.N_SUBSTACKS = 4

        self.plane = np.empty((self.NY, self.NX), "uint16")
        self.substack = np.empty((self.NZ//self.N_SUBSTACKS, self.NY, self.NX), "uint16")
        self.plane = generate_test_image((self.NY, self.NX), 1, 2)
        for z in range(self.NZ//self.N_SUBSTACKS):
            self.substack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ//self.N_SUBSTACKS)

        self.write_virtual_by_plane()
        self.write_virtual_by_substack()

    def write_virtual_by_plane(self):
        bdv_writer = npy2bdv.BdvWriter(self.fname0,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES)
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
                                bdv_writer.append_plane(plane=self.plane,
                                                        z=iz,
                                                        time=t, channel=i_ch, illumination=i_illum,
                                                        tile=i_tile, angle=i_angle)
        bdv_writer.write_xml_file(ntimes=self.N_T)
        bdv_writer.close()

    def write_virtual_by_substack(self):
        ''''Write a virtual stack by substacks, with downsampling pyramid.'''
        bdv_writer = npy2bdv.BdvWriter(self.fname1,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES,
                                       subsamp=((1, 1, 1), (2, 4, 4)),
                                       blockdim=((4, 16, 16), ))
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
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            for isub in range(self.N_SUBSTACKS):
                                bdv_writer.append_substack(substack=self.substack,
                                                           z_start=isub*self.substack.shape[0],
                                                           time=t, channel=i_ch, illumination=i_illum,
                                                           tile=i_tile, angle=i_angle)
        bdv_writer.write_xml_file(ntimes=self.N_T)
        bdv_writer.close()

    def test_read_voxels(self):
        """"Read the voxel values back and compare to expected values"""
        editor = npy2bdv.BdvEditor(self.fname1)
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            view = editor.read_view(time=t, channel=i_ch, illumination=i_illum,
                                                    tile=i_tile, angle=i_angle)
                            self.assertAlmostEqual(np.mean(view), np.mean(self.substack), 6, "Mean differs.")
                            self.assertAlmostEqual(np.std(view), np.std(self.substack), 6, "Std differs.")
        editor.finalize()

    def tearDown(self) -> None:
        """"Tidies up after EACH test method has been run."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()

