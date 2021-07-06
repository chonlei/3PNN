from setuptools import setup, find_packages

setup(
    name='3PNN',
    version='1.0.0',
    description='Modelling 3D printed cochlea EFI with neural network.',
    license='BSD 3-clause license',
    maintainer='Chon Lok Lei',
    maintainer_email='chonloklei@gmail.com',
    packages=find_packages(include=('method')),
    install_requires=[
        'tensorflow==2.1.0',
        'pints @ git+https://github.com/pints-team/pints@6e30367e07c7be0888c8e051d0e83a8cbebdb2cb#egg=pints',
        'SALib==1.4.0.1',
        'scikit-learn==0.24.0',
        'seaborn==0.11.1',
        'joblib==1.0.0',
    ],
)

