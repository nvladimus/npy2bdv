import npy2bdv
import os
import shutil
import unittest
import numpy as np


def generate_test_image(dim_yx, iz, nz):
    """Gaussian blob spanning the range of uint16 type."""
    x = np.linspace(-3, 3, dim_yx[1])
    y = np.linspace(-3, 3, dim_yx[0])
    sigma = 1.0 - abs(iz - nz/2) / nz
    x, y = np.meshgrid(x, y)
    return (65535 * np.exp(- ((x ** 2) + (y ** 2)) / (2 * sigma**2) )).astype("uint16")


class TestReadWrite(unittest.TestCase):
    """Write a dataset with multiples views, and load it back. Compare the loaded dataset vs expetations.
    """
    def setUp(self) -> None:
        self.test_dir = "./test_files/"
        self.fname = self.test_dir + "test_ex1_t2_ch2_illum2_angle2.h5"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.NZ, self.NY, self.NX = 4, 65, 65 # XY dims must be odd to get nominal 65535 peak value.
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 2, 3, 4, 5, 4
        
        self.stack = np.empty((self.NZ, self.NY, self.NX), "uint16")
        for z in range(self.NZ):
            self.stack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ)

        bdv_writer = npy2bdv.BdvWriter(self.fname,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES, overwrite=True)
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
        reader = npy2bdv.BdvEditor(self.fname)
        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            view = reader.read_view(time=t,
                                                    channel=i_ch,
                                                    illumination=i_illum,
                                                    tile=i_tile,
                                                    angle=i_angle)
                            self.assertTrue(view.min() >= 0, f"Min() value incorrect: {view.min()}")
                            self.assertTrue(view.max() == 65535, f"Max() value incorrect: {view.max()}")
                            self.assertTrue((view == self.stack).all(), "Written stack differs from the loaded stack.")
        reader.close()

    def test_view_properties(self):
        """"BdvReader(): does the meta-info in XML file have expected values?"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        reader = npy2bdv.BdvEditor(self.fname)
        for i_ch in range(self.N_CH):
            for i_illum in range(self.N_ILL):
                for i_tile in range(self.N_TILES):
                    for i_angle in range(self.N_ANGLES):
                        vox_size = reader.get_view_property("voxel_size",
                                                            channel=i_ch,
                                                            illumination=i_illum,
                                                            tile=i_tile,
                                                            angle=i_angle)
                        view_shape = reader.get_view_property("view_shape",
                                                              channel=i_ch,
                                                              illumination=i_illum,
                                                              tile=i_tile,
                                                              angle=i_angle)
                        self.assertEqual(vox_size, (1, 1, 4), f"Voxel size is incorrect: {vox_size}.")
                        self.assertEqual(view_shape, self.stack.shape[::-1], f"View shape incorrect: {view_shape}.")
        reader.close()

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

    @unittest.skip
    def test_cropping(self):
        """"BdvEditor(): crop a view and check if new H5 view size matches the XML view size. Todo"""

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()

