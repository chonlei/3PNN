# Cochlea EFI

This project aims to extract the features of cochlea electric field imaging (EFI) measurements.
It also aims to study the relationship between the extracted EFI features and the material properties that underlies each cochlea model.

### Requirements

The code requires Python (2.7 or 3.6+) and one dependencies:
[PINTS](https://github.com/pints-team/pints#installing-pints).


## Structure of the repo

### Main
- `fit-model.py`: Run fitting of the cochlea EFI model in `model.py` with argument `[str:file_id]`, to specify the ID of the measurement to fit to.

### Data
- `data`: Contains raw EFI measurement in `.txt` format with tab delimiter.
Rows are the readout/recording at each electrode ordered from 1 to 16; columns are the stimulation electrode numbers (again from 1 to 16).
Each file name is the `file_id`, the ID of the measurement.
- `input`: Contains the input/printing parameters for re-creating the cochlea model, of that the file name (`file_id`) should match those in `data`.

### Methods
- `read.py`: Read raw data files.
- `plot.py`: Contains simple plotting functions.
- `feature.py`: Contains functions extracting features of the EFI measurements.
- `model.py`: Contains a cochlea EFI model and likelihood functions for fitting.

### Simple tests
- `test-features.py`: Run simple plot and check feature extraction;
takes argument `[str:file_id]`, the ID of the measurement.
- `test-model.py`: Run simple plot and simulation with the cochlea EFI model in `model.py`.


## Acknowledging this work

If you publish any work based on the contents of this repository please cite:

[Placeholder]

