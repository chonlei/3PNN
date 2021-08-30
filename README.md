# 3PNN

Code and source data used in the article "3D printed biomimetic cochleae and machine learning co-modelling provides clinical informatics for cochlear implant patients" (Lei et al. _Nature Communications_, 2021).

This repository aims to model the relationship between electric field imaging (EFI) profiles and the electro-anatomical features of human cochleae using a multilayer perceptron (MLP) artificial neural network (NN) model.
The NN model was developed by training data generated from 3D printed biomimetic cochleae.
It learned the mapping from the inputs (the 5 model descriptors of the biomimetic cochleae, the stimulating electrode positions, and the recording electrode positions) to the EFI profiles (the output).
Predictions of EFI profiles and the model descriptors (input parameters) can be made using this model.

### Requirements

The code requires Python (3.6+ and was tested with 3.7.6) and the following dependencies:
[tensorflow](https://www.tensorflow.org/install),
[PINTS](https://github.com/pints-team/pints#installing-pints),
[scikit-learn](https://scikit-learn.org/stable/install.html),
[SALib](https://salib.readthedocs.io/en/latest/getting-started.html#installing-salib).
Installing Tensorflow in Windows may require [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).
It also needs [seaborn](https://seaborn.pydata.org/installing.html) for plotting.

To install the code and the above dependencies, navigate to the path where you downloaded this repository and run:
```
$ pip install --upgrade pip
$ pip install .
```


## Structure of the repository

### NN model
- [`fit-nn.py`](./fit-nn.py): Run fitting of a NN model, with argument `[str:input_file_ids.txt]`
                              containing a list of file IDs as training data. 
- [`predict-nn.py`](./predict-nn.py): Run (forward) prediction using the trained NN model (from `fit-nn.py`),
                                      with arguments `[str:nn_name]` (by default, the file name of the `[str:input_file_ids.txt]`)
                                      and `[str:predict_ids.txt]` (a list of file IDs for prediction;
                                      their input parameters are stored in the input folder).
- [`invabc-nn.py`](./invabc-nn.py): Run (inverse) prediction using the trained NN model (from `fit-nn.py`),
                                    with arguments `[str:nn_name]` (by default, the file name of the `[str:input_file_ids.txt]`)
                                    and `[str:predict_ids.txt]` (a list of file IDs for prediction;
                                    their EFI data are stored in the data folder).
                                    It also requires `fix_param.py` to specify the parameters that are not fixed.
- [`fix_param.py`](./fix_param.py): Specify the parameters that are fixed or not fixed. Put `None` if it is not fixed.
- [`sensitivity-nn.py`](./sensitivity-nn.py): Run sensitivity analysis for the trained NN model (from `fit-nn.py`),
                                              with arguments `[str:nn_name]` (by default, the file name of the `[str:input_file_ids.txt]`).


### Data
- [`data`](./data): Contains raw EFI measurements in `.txt` format with tab delimiter.
                    Rows are the readout/recording at each electrode ordered from 1 to 16;
                    columns are the stimulation electrode numbers (again from 1 to 16).
                    Each file name is the file ID, the ID of the measurement.
  - `available-electrodes.csv`: Specify the available electrodes of each raw EFI measurement.
- [`input`](./input): Contains the input/printing parameters for re-creating the cochlea model,
                      of that the file ID should match those in data.
                      Rows from 1 to 5 are basal lumen diameter, infill density, taper ratio, cochlear width and cochlear height.


### Methods
- [`method`](./method): A module containing useful methods, functions, and helper classes for this project; for further details, see below.
  - `feature.py`: Contains functions extracting features of the EFI measurements.
  - `io.py`: I/O helper classes, for read and write predefined file format.
  - `nn.py`: Contains neural network functions for regression.
  - `plot.py`: Contains simple plotting functions.
  - `transform.py`: Contains classes for parameter transformation.


### Results
- [`out-nn`](./out-nn): Contains NN model fitting and prediction results (from `fit-nn.py`, `predict-nn.py`, `invabc-nn.py`, and `sensitivity-nn.py`).


## Acknowledging this work

If you publish any work based on the contents of this repository please cite ([CITATION file](./CITATION)):

Lei, I. M. et al. (2021). 3D printed biomimetic cochleae and machine learning co-modelling provides clinical informatics for cochlear implant patients. _Nature Communications_.
