import npy2bdv
import os
import shutil
import unittest
import numpy as np


def generate_test_image(dim_yx, iz, nz):
    """Gaussian blob spanning the range of uint16 type.
    XY dims must be odd to get 65535 peak value."""
    x = np.linspace(-3, 3, dim_yx[1])
    y = np.linspace(-3, 3, dim_yx[0])
    sigma = 1.0 - abs(iz - nz/2) / nz
    x, y = np.meshgrid(x, y)
    return 65535 * np.exp(- ((x ** 2) + (y ** 2)) / (2 * sigma**2) )


class TestRange(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = "./test_files/"
        self.fname = self.test_dir + "test_ex1_t2_ch2_illum2_angle2.h5"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        nz, ny, nx = 16, 129, 129
        stack = np.empty((nz, ny, nx))
        for z in range(nz):
            stack[z, :, :] = generate_test_image((ny, nx), z, nz)

        bdv_writer = npy2bdv.BdvWriter(self.fname, nchannels=2, nilluminations=2, nangles=2, subsamp=((1, 1, 1),))
        for t in range(2):
            for i_ch in range(2):
                for i_illum in range(2):
                    for i_angle in range(2):
                        bdv_writer.append_view(stack, time=t, channel=i_ch, illumination=i_illum, angle=i_angle)
        bdv_writer.write_xml_file(ntimes=2)
        bdv_writer.close()

    def test_range_uint16(self):
        """"Check if the reader imports full uint16 range correctly"""
        assert os.path.exists(self.fname), f'File {self.fname} not found.'
        reader = npy2bdv.BdvReader(self.fname)
        for t in range(2):
            for s in range(8):
                view = reader.read_view(time=t, isetup=s)
                self.assertTrue(view.min() >= 0, f"Min() value incorrect: {view.min()}")
                self.assertTrue(view.max() == 65535, f"Max() value incorrect: {view.max()}")
        reader.close()

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()

