from setuptools import setup, find_packages

setup(
        name='npy2bdv',
        version='0.1',
        description='A minimalistic library for writing image stacks (numpy 3d-arrays) into HDF5 files in Fiji BigDataViewer/BigStitcher format.',
        url='https://github.com/nvladimus/npy2bdv',
        author='Nikita Vladimirov',
        install_requires=[
            'h5py',
            'numpy',
            'scikit-image',
        ],
        packages=[
                'npy2bdv',
        ]
)
