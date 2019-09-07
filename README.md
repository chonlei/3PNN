# Cochlea EFI

This project aims to extract the features of cochlea electric field imaging (EFI) measurements.
It also aims to study the relationship between the extracted EFI features and the material properties that underlies each cochlea model.

## Structure of the code
- `data`: Contains raw EFI measurement in `.txt` format with tab delimiter.
Rows are the readout/recording at each electrode ordered from 1 to 16; columns are the stimulation electrode numbers (again from 1 to 16).
Each file name is the 'file\_id', the ID of the measurement.
- `input`: Contains the input/printing parameters for re-creating the cochlea model, of that the file name ('file\_id') should match those in `data`.

## Methods
- `read.py`: Read raw data files.
- `plot.py`: Contains simple plotting functions.
- `feature.py`: Contains functions extracting features of the EFI measurements.

## Run test
- `test-features.py`: Run simple plot and check feature extraction;
takes argument `[str:file\_id]`, the ID of the measurement.

# Cite this work
[Placeholder]
