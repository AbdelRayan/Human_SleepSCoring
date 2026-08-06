# Jalmar HDF5 Module

This folder converts preprocessed MAT files into a subject-wise HDF5 file with
sleep feature vectors and mapped stage labels.

## Quickstart

```powershell
pip install -r Jalmar/requirements.txt
python Jalmar/hdf5/main.py
```

## Main Files

- `main.py`: pipeline entry point (subject discovery, processing, resume behavior)
- `hdf5_creation.py`: load MAT files, compute features, write HDF5 groups
- `computing_features.py`: spectral, index, and complexity feature computation
- `artifacts_detection.py`: artifact detection and artifact-to-epoch mapping

## Input Requirements

Each subject needs five MAT files (flat or hierarchical directory structure):

- `*Fpz*.mat`
- `*Pz*.mat`
- `*EMG*.mat`
- `*EOG*.mat`
- `*states*.mat`

These are produced by `Jalmar/pre_processing/pre_processing.py`.

## Output HDF5 Structure

One group per subject containing:

- `features`: `(n_epochs, n_features)` float32
- `scores`: `(n_epochs,)` uint8
- attributes describing feature and score semantics

## Features

The current pipeline writes 15 features per epoch:

1. Index_W
2. Index_R
3. Index_N
4. Index_1
5. Index_2
6. Index_3
7. Index_4
8. Delta
9. Theta
10. Aperiodic
11. DFA
12. MSE
13. EOG
14. Index_R_noEOG
15. Index_N_noEOG

## Artifact Handling

- Artifact samples are detected per signal channel using threshold rules.
- Sample-level artifact indices are mapped to epoch indices.
- Any affected epoch is relabeled as score `5` (Movement/Artifact).
- Feature values are still computed; artifact status is carried in `scores`.

## Sleep Stage Encoding

- `0`: Awake
- `1`: N1
- `2`: N2
- `3`: N3
- `4`: REM
- `5`: Movement/Artifact

## Running the Pipeline

From workspace root:

```powershell
python Jalmar/hdf5/main.py
```

Common options:

```powershell
python Jalmar/hdf5/main.py --input_dir "C:/path/to/mat" --output "C:/path/to/sleep_features.h5" --epoch_length 10 --mode a --skip_existing
```

## Resume-Safe Behavior

The default config in `main.py` uses append mode and skip-existing behavior.
This allows long runs to continue without recomputing already written subjects.

