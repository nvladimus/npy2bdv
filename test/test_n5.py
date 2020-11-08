#!/usr/bin/env python -m test.test_n5
import npy2bdv
import os
import shutil
import unittest
import numpy as np
from .sample import generate_test_image


class TestWriteN5(unittest.TestCase):
    """Write a dataset with multiple views and inspect it.
    Compare the written dataset vs expectations.
    """

    def setUp(self) -> None:
        self.test_dir = "./test/test_files/"
        self.fname = self.test_dir + "test_t2_ch3_ill2_tiles4_ang2.n5"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.NZ, self.NY, self.NX = 8, 256, 256
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 2, 3, 2, 4, 2
        self.SUBSAMPLING_ZYX = ((1, 1, 1), (4, 2, 2))
        self.BLOCKDIM_ZYX = ((2, 64, 64), (4, 128, 128))
        self.stack = np.empty((self.NZ, self.NY, self.NX), "uint16")

        for z in range(self.NZ):
            self.stack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ)

        bdv_writer = npy2bdv.BdvWriter(self.fname,
                                       subsamp_zyx=self.SUBSAMPLING_ZYX,
                                       blockdim=self.BLOCKDIM_ZYX,
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
                            bdv_writer.append_view(self.stack,
                                                   time=t,
                                                   channel=i_ch,
                                                   illumination=i_illum,
                                                   tile=i_tile,
                                                   angle=i_angle,
                                                   voxel_size_xyz=(1, 1, 4))
        bdv_writer.write_xml_file(ntimes=self.N_T)
        bdv_writer.close()

    def test_n5_writing(self):
        print(f"INFO: The {self.fname} was manually loaded in Fiji/BigStitcher and checked.")

    def tearDown(self) -> None:
        pass
    #    if os.path.exists(self.test_dir):
    #        shutil.rmtree(self.test_dir)


class TestWriteN5Tiles(unittest.TestCase):
    """Write a dataset with multiple tiles on a grid.
    """

    def setUp(self) -> None:
        self.test_dir = "./test/test_files/"
        self.fname = self.test_dir + "test_t2_ch3_ill2_tiles2x2_ang2.n5"
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.NZ, self.NY, self.NX = 8, 512, 512
        self.N_T, self.N_CH, self.N_ILL, self.N_TILES, self.N_ANGLES = 2, 3, 2, 4, 2
        self.stack = np.empty((self.NZ, self.NY, self.NX), "uint16")

        for z in range(self.NZ):
            self.stack[z, :, :] = generate_test_image((self.NY, self.NX), z, self.NZ)

        bdv_writer = npy2bdv.BdvWriter(self.fname,
                                       nchannels=self.N_CH,
                                       nilluminations=self.N_ILL,
                                       ntiles=self.N_TILES,
                                       nangles=self.N_ANGLES,
                                       overwrite=True,
                                       compression=None,
                                       format='n5')
        # split the samples blob into tiles, and set their absolute coordinates when writing
        n_tiles_y = self.N_TILES // 2
        n_tiles_x = self.N_TILES // n_tiles_y
        tile_w, tile_h = self.NX // n_tiles_x, self.NY // n_tiles_y
        unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0),  # change the 4. value for x_translation (px)
                                (0.0, 1.0, 0.0, 0.0),  # change the 4. value for y_translation (px)
                                (0.0, 0.0, 1.0, 0.0)))  # change the 4. value for z_translation (px)

        for t in range(self.N_T):
            for i_ch in range(self.N_CH):
                for i_illum in range(self.N_ILL):
                    for i_tile in range(self.N_TILES):
                        for i_angle in range(self.N_ANGLES):
                            ix_tile, iy_tile = np.unravel_index(i_tile, (n_tiles_x, n_tiles_y))
                            tile_stack = self.stack[:,
                                                    iy_tile * tile_h:(iy_tile + 1) * tile_h,
                                                    ix_tile * tile_w:(ix_tile + 1) * tile_w]
                            affine_matrix = unit_matrix.copy()
                            affine_matrix[0, 3] = ix_tile * tile_w  # x-translation of a tile
                            affine_matrix[1, 3] = iy_tile * tile_h  # y-translation of a tile
                            print(f"Tile shape: {tile_stack.shape}")
                            bdv_writer.append_view(tile_stack,
                                                   time=t,
                                                   channel=i_ch,
                                                   illumination=i_illum,
                                                   tile=i_tile,
                                                   angle=i_angle,
                                                   voxel_size_xyz=(1, 1, 4),
                                                   m_affine=affine_matrix,
                                                   name_affine=f"tile {i_tile} translation")
        bdv_writer.write_xml_file(ntimes=self.N_T)
        bdv_writer.close()

    def test_n5_tiling(self):
        print(f"INFO: The {self.fname} was manually loaded in Fiji/BigStitcher "
              f"and tiles were checked with random coloring pattern [C-key]")

    def tearDown(self) -> None:
        pass
    #    if os.path.exists(self.test_dir):
    #        shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()
