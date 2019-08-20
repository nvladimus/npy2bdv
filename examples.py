import time
import sys
import numpy as np
import npy2bdv

print("Example1: writing 2 time points and 2 channels")
fname = "./timepts2_channels2.h5"
bdv_writer = npy2bdv.BdvWriter(fname, nsetups=2, subsamp=((1, 1, 1),))
bdv_writer.write_setups_header()
rand_stack = np.random.randint(0, 100, size=(41, 1024, 2048), dtype='int16')
bdv_writer.append_view(rand_stack, itime=0, isetup=0)
bdv_writer.append_view(rand_stack, itime=0, isetup=1)
bdv_writer.append_view(rand_stack, itime=1, isetup=0)
bdv_writer.append_view(rand_stack, itime=1, isetup=1)
bdv_writer.write_xml_file(ntimes=2, nchannels=2)
bdv_writer.close()
print("random-generated data written into " + fname + "\n")

print("Example2: speed test for 20 time points and 2 channels. File size is 7 GB!")
fname = "./timeser20_chan2.h5"
bdv_writer = npy2bdv.BdvWriter(fname, nsetups=2, subsamp=((1, 1, 1),))
ntimes = 20
nchannels = 2
start_time_total = time.time()
bdv_writer.write_setups_header()
i_stacks = 0
time_list = []
for isetup in range(nchannels):
    for itime in range(ntimes):
        start_time = time.time()
        bdv_writer.append_view(rand_stack, itime=itime, isetup=isetup)
        time_interval = time.time() - start_time
        time_list.append(time_interval)
        i_stacks += 1.0

bdv_writer.write_xml_file(ntimes=ntimes, nchannels=2)
bdv_writer.close()
time_per_stack = (time.time() - start_time_total)/i_stacks
print("H5 mean writing time per stack: {:1.3f}".format(time_per_stack) + " sec.")
print("H5 mean writing speed: " + str(int(sys.getsizeof(rand_stack)/time_per_stack/1e6)) + " MB/s")
print("random-generated data written into " + fname + "\n")
