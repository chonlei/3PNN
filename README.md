# Cochlea EFI

This project aims to extract the features of cochlea electric field imaging (EFI) measurements.
It also aims to study the relationship between the extracted EFI features and the material properties that underlies each cochlea model.

### Requirements

The code requires Python (2.7 or 3.6+) and one dependencies:
[PINTS](https://github.com/pints-team/pints#installing-pints). [Need updating]

To install, navigate to the path where you downloaded this repo and run:
```
$ pip install --upgrade pip
$ pip install .
```


## Structure of the repo

### Main

#### Basic
- `analyse-features.py`: Analyse extracted features from `get-features.py` for each input parameter with argument `[int:analyse_index]`.
- `get-features.py`: Extract predefined EFI features of a specified measurement, by argument `[str:file_id]`.
- `get-features-all.sh`: A bash script to run `get-features.py` over a list of measurements.

#### Resistance model
- `fit-model.py`: Run fitting of the cochlea EFI model in `model.py` with argument `[str:file_id]`, to specify the ID of the measurement to fit to.

#### GP model
This Gaussian process (GP) model assumes each stimulus measurement can be fitted indepedently.

- `fit-gp.py`: Run fitting of a GP model, with argument `[str:input_file]` containing a list of `file_id` as training data (both EFI data and input parameters).
- `predict-gp.py`: Run (forward) prediction using the fitted GP model (from `fit-gp.py`), with arguments `[str:gp_name]` (by default, the file name of the `[str:input_file]`) and `[str:input_file(predict)]` (a list of `file_id`, input parameters, for prediction).
- `inv-gp.py`: Run inverse problem using the fitted GP model, with arguments `[str:gp_name]` and `[str:input_file(inv)]` (a list of `file_id`, EFI data, for inverse problem, i.e. predicting the input parameters). It also requires `fix_param.py` to specify the parameters that are not fitted.

#### GP full model
This Gaussian process (GP) model fits all stimulus measurements together (so it's much slower).

- `fit-gp-full.py`: Run fitting of a GP model, with argument `[str:input_file]` containing a list of `file_id` as training data (both EFI data and input parameters).
- `predict-gp-full.py`: Run (forward) prediction using the fitted GP model (from `fit-gp-full.py`), with arguments `[str:gp_name]` (by default, the file name of the `[str:input_file]`) and `[str:input_file(predict)]` (a list of `file_id`, input parameters, for prediction).

#### NN model
This Neural Network (NN) model assumes each stimulus measurement can be fitted indepedently.

- `fit-nn.py` Run fitting of a NN model, with argument `[str:input_file]` containing a list of `file_id` as training data (both EFI data and input parameters).
- `predict-nn.py` Run (forward) prediction using the fitted NN model (from `fit-nn.py`), with arguments `[str:nn_name]` (by default, the file name of the `[str:input_file]`) and `[str:input_file(predict)]` (a list of `file_id`, input parameters, for prediction).

### Results
- `fig`: Contains all figures (from `analyse-features.py`).
- `out-features`: Contains extracted EFI features (from `get-features.py`).
- `out-model`: Contains resistance model fitting result output (from `fit-model.py`).
- `out-gp`: Contains GP model fitting and prediction results (from `fit-gp.py`, `predict-gp.py`, `inv-gp.py`).
- `out-gp-full`: Contains GP model fitting and prediction results (from `fit-gp-full.py`, `predict-gp-full.py`).
- `out-nn`: Contains NN model fitting and prediction results (from `fit-nn.py`, `predict-nn.py`).

### Data
- `data`: Contains raw EFI measurement in `.txt` format with tab delimiter.
Rows are the readout/recording at each electrode ordered from 1 to 16; columns are the stimulation electrode numbers (again from 1 to 16).
Each file name is the `file_id`, the ID of the measurement.
- `input`: Contains the input/printing parameters for re-creating the cochlea model, of that the file name (`file_id`) should match those in `data`.

### Methods
- `method`: A module containing useful methods, functions, and helper classes for this project;
for further details, see [here](./method/README.md).

### Others
#### Simple utilities
- `get-id.py`: Run with argument `[arr:input_param]`, the input parameter values, print the ID of the measurement(s) that matches the inputs on console.

#### Simple plots
- `quick-plot.py`: Run simple plot for the fitted results (same in those at the end of `fit-model.py`).

#### Simple tests
- `test-features.py`: Run simple plot and check feature extraction;
takes argument `[str:file_id]`, the ID of the measurement.
- `test-model.py`: Run simple plot and simulation with the cochlea EFI model in `model.py`.


## Acknowledging this work

If you publish any work based on the contents of this repository please cite:

[Placeholder]

