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
        'seaborn',
        'scikit-learn',
        'joblib',
        'tensorflow',
        'pints @ git+https://github.com/pints-team/pints@6e30367e07c7be0888c8e051d0e83a8cbebdb2cb#egg=pints',
    ],
    extras_require={
        'jupyter': ['jupyter'],
    },
)

