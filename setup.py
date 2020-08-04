from setuptools import setup, find_packages

setup(
        name='npy2bdv',
        version='2020.06.25',
        description='Package for writing/reading 3d numpy arrays to/from HDF5 files (Fiji/BigDataViewer/BigStitcher format).',
        url='https://github.com/nvladimus/npy2bdv',
        author='Nikita Vladimirov',
        author_email="nvladimus@gmail.com",
        install_requires=[
            'h5py',
            'numpy',
            'scikit-image'
        ],
        packages=find_packages()
)