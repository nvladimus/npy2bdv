from setuptools import setup, find_packages

setup(
        name='npy2bdv',
        version='2019.12.23',
        description='Package for writing 3d numpy arrays into HDF5 files in Fiji/BigDataViewer/BigStitcher format.',
        url='https://github.com/nvladimus/npy2bdv',
        author='Nikita Vladimirov',
        author_email="nvladimus@gmail.com",
        install_requires=[
            'h5py',
            'numpy',
            'scikit-image',
        ],
        packages=find_packages()
)
