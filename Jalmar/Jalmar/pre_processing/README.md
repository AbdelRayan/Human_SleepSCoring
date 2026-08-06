# Jalmar Pre-Processing Module

This folder converts raw EDF sleep recordings into per-subject MAT files used
by the HDF5 feature pipeline.

## Quickstart

```powershell
pip install -r Jalmar/requirements.txt
python Jalmar/pre_processing/pre_processing.py
python Jalmar/pre_processing/validate_preprocessing.py "C:/path/to/output"
```

## Scripts

- `pre_processing.py`: main conversion script (EDF + annotations to MAT)
- `shared_processing_functions.py`: helper functions for annotation, cropping,
  stage extraction, and MAT writing
- `validate_preprocessing.py`: quality checks on generated MAT files

## Inputs

Required inputs for `pre_processing.py`:

1. EDF recordings directory
2. EDF annotation directory
3. Output directory for MAT files

Expected channels:

- `Fpz-Cz`
- `Pz-Oz`
- `horizontal` (converted to EOG)
- `submental` (converted to EMG)

## Outputs

For each subject, the script writes MAT files in a flat directory:

- `<subject>_Fpz-Cz.mat`
- `<subject>_Pz-Oz.mat`
- `<subject>_EMG.mat`
- `<subject>_EOG.mat`
- `<subject>_states.mat`

These files are consumed by `Jalmar/hdf5/main.py`.

## Processing Steps (Chronological)

1. Discover EDF files
2. Extract subject ID from filename
3. Skip subject if already processed in output folder
4. Load selected channels from EDF
5. Find and attach annotation file
6. Crop around sleep period with configured wake-time margin
7. Generate sample-level sleep stage vector and save states MAT
8. Filter EEG channels and process EOG/EMG channels
9. Save per-channel MAT files

## Run

Default run (after editing paths inside script if needed):

```powershell
python Jalmar/pre_processing/pre_processing.py
```

## Important Parameters in pre_processing.py

- `wake_time`: seconds of wake retained before first sleep and after last sleep
- `sleep_edf_stage_id`: mapping from annotation text to stage integers
- `eog_bandpass`: filter range for horizontal channel before saving as EOG
- `channels`: channel list to extract from EDF

## Validation

Run validation on the output MAT folder:

```powershell
python Jalmar/pre_processing/validate_preprocessing.py "C:/path/to/output"
```

Validation checks include:

- missing required channels per subject
- all-NaN/all-zero arrays
- near-zero variance signals
- high NaN ratios
- extreme values

## Notes

- Output is flat-file based by default (all subject MAT files in one folder).
- Subject skip logic allows resume-style processing when rerunning the script.
