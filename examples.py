import time
import sys
import numpy as np
import npy2bdv

print("Example1: writing 2 time points and 2 channels")
fname = "./ex1_t2_ch2.h5"
bdv_writer = npy2bdv.BdvWriter(fname, nchannels=2, subsamp=((1, 1, 1),))
rand_stack = np.random.randint(0, 100, size=(41, 1024, 2048), dtype='int16')
bdv_writer.append_view(rand_stack, time=0, channel=0)
bdv_writer.append_view(rand_stack, time=0, channel=1)
bdv_writer.append_view(rand_stack, time=1, channel=0)
bdv_writer.append_view(rand_stack, time=1, channel=1)
bdv_writer.write_xml_file(ntimes=2)
bdv_writer.close()
print("Random-generated data is written into " + fname + "\n")

print("Example2: speed test for 20 time points and 2 channels. File size is 7 GB!")
fname = "./ex2_t20_chan2.h5"
bdv_writer = npy2bdv.BdvWriter(fname, nchannels=2, subsamp=((1, 1, 1),))
ntimes = 20
nchannels = 2
start_time_total = time.time()
i_stacks = 0
time_list = []
for ichannel in range(nchannels):
    for itime in range(ntimes):
        start_time = time.time()
        bdv_writer.append_view(rand_stack, time=itime, channel=ichannel)
        time_interval = time.time() - start_time
        time_list.append(time_interval)
        i_stacks += 1.0

bdv_writer.write_xml_file(ntimes=ntimes)
bdv_writer.close()
time_per_stack = (time.time() - start_time_total) / i_stacks
print("H5 mean writing time per stack: {:1.3f}".format(time_per_stack) + " sec.")
print("H5 mean writing speed: " + str(int(sys.getsizeof(rand_stack) / time_per_stack / 1e6)) + " MB/s")
print("Random-generated data is written into " + fname + "\n")

print("Example3: writing 1 time point and 1 channel with 10-px un-shear transformation along X axis.\n" +
      "With non-isotropic voxel size calibration.")
fname = "./ex3_t1_ch1_unshear.h5"
bdv_writer = npy2bdv.BdvWriter(fname, nchannels=1, subsamp=((1, 1, 1),))
shear_x_px = 10
affine_matrix = np.array(((1.0, 0.0, -shear_x_px, 0.0),
                          (0.0, 1.0, 0.0, 0.0),
                          (0.0, 0.0, 1.0, 0.0)))
bdv_writer.append_view(rand_stack, time=0, channel=0,
                       m_affine=affine_matrix,
                       name_affine="unshearing transformation",
                       calibration=(1, 1, 1))

bdv_writer.write_xml_file(ntimes=1)
bdv_writer.close()
print("(Un)sheared stack is written into " + fname + "\n")

print("Example4: writing 1 time point and 1 channel with voxel size, exposure, camera and microscope properties")
fname = "./ex4_t1_ch1_cam_props.h5"
bdv_writer = npy2bdv.BdvWriter(fname, nchannels=1, subsamp=((1, 1, 1),))
bdv_writer.append_view(rand_stack, time=0, channel=0,
                       voxel_size_xyz=(1, 1, 5), voxel_units='um',
                       exposure_time=10, exposure_units='ms')
bdv_writer.write_xml_file(ntimes=1, camera_name="Hamamatsu OrcaFlash100",
                          microscope_name='Superscope',
                          user_name='nvladimus')
bdv_writer.close()
print("Stack is written into " + fname + "\n")

print("Example5: 1 time point and 1 channel with 3-level subsampling and compression")
fname = "./ex5_t1_ch1_level3_gzip.h5"
bdv_writer = npy2bdv.BdvWriter(fname, nchannels=1,
                               subsamp=((1, 1, 1), (2, 4, 4), (4, 16, 16)),
                               blockdim=((64, 64, 64),),
                               compression='gzip')
bdv_writer.append_view(rand_stack, time=0, channel=0)
bdv_writer.write_xml_file(ntimes=1)
bdv_writer.close()
print("Stack with subsampling is written into " + fname + "\n")
