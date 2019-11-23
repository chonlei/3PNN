from setuptools import setup, find_packages

setup(
    name='cochlea-efi',
    version='0.0.1',
    description='Modelling choclea EFI',
    license='BSD 3-clause license',
    maintainer='Chon Lok Lei',
    maintainer_email='chon.lei@cs.ox.ac.uk',
    packages=find_packages(include=('method')),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'tensorflow',
    ],
    extras_require={
        'jupyter': ['jupyter'],
    },
)

