import npy2bdv
import os

examples_dir = "./example_files/"
assert os.path.exists(examples_dir), 'Please run the Example 1 to generate the dataset.'


def check_range():
    """"Check if the reader imports full uint16 range correctly"""
    fname = examples_dir + "ex1_t2_ch2_illum2_angle2.h5"
    assert os.path.exists(fname), f'File {fname} not found.'
    reader = npy2bdv.BdvReader(fname)
    view0 = reader.read_view(time=0, isetup=0)
    print(f"Array min, max, mean: {view0.min()}, {view0.max()}, {int(view0.mean())}")
    assert view0.min() >= 0, "Min() value incorrect: {view0.min()}"
    assert view0.max() <= 65535, "Max() value incorrect: {view0.max()}"
    reader.close()

check_range()

