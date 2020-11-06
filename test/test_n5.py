#!/usr/bin/env python -m test.test_n5
import npy2bdv
import os
import shutil
import unittest
import numpy as np
from .sample import generate_test_image


class TestReadWriteN5(unittest.TestCase):
    """Write a dataset with multiples views, and load it back. Compare the loaded dataset vs expetations.
    """

    def setUp(self) -> None:
        self.test_dir = "./test/test_files/"
        self.fname = self.test_dir + "test_t5_ch3_ill2_tiles4_ang2.n5"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.NZ, self.NY, self.NX = 8, 35, 35  # XY dims must be odd to get nominal 65535 peak value.
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 5, 3, 2, 4, 2

        self.stack = np.empty((self.NZ, self.NY, self.NX), "uint16")
        for z in range(self.NZ):
            self.stack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ)

        bdv_writer = npy2bdv.BdvWriter(self.fname,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES, overwrite=True,
                                       compression=None,
                                       format='n5')
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

    def test_n5_writing(self):
        pass

    def tearDown(self) -> None:
        pass
    #    if os.path.exists(self.test_dir):
    #        shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()
