# Jalmar Pipeline

Jalmar contains a full EEG sleep processing workflow:

1. Pre-process EDF recordings into per-subject MAT files
2. Compute sleep features and store them in HDF5
3. Run feature statistics and ranking analyses

## mcRBM Versions

- `mcRBM/` is Dilon's original mcRBM architecture.
- `mcRBM_N1_test/` is the latest Jalmar-created variant.
- `mcRBM_N1_test/` can be used for feature sets with any feature count, making analysis of subsets work seamlessly.

## Paper Model Mapping

Models used in the paper "Data-Driven Sleep Staging: Discovering Latent States From Human Recordings" by Jalmar Derikx:

- `mcRBM_v1` = `mcrbm_full_V1`
- `mcRBM_v2` = `mcrbm_full_relaxed_V4`
- `mcRBM_v3` = `mcrbm_full_raw_eog_V1`

The first name is the label used in the paper, and the second name is the corresponding experiment folder name.

## Dependencies

- Full install: `requirements.txt`
- CPU-only install: `requirements-base.txt`
- GPU support is optional and uses CuPy when available.

## Quickstart

```powershell
pip install -r Jalmar/requirements.txt
python Jalmar/pre_processing/pre_processing.py
python Jalmar/hdf5/main.py
python Jalmar/stats/features/features_stats.py --hdf5 "C:/path/to/sleep_features.h5" --output_dir "Jalmar/stats/features/output"
```

## Folder Overview

- `pre_processing/`: EDF + annotations to MAT conversion
- `hdf5/`: MAT to HDF5 conversion with feature computation and artifact labeling
- `stats/features/`: feature statistics, ranking, and visualization outputs
- `requirements.txt`: Python dependencies used by Jalmar scripts

## End-to-End Order

1. Run pre-processing
2. Run HDF5 feature extraction
3. Run feature statistics and ranking

## Environment

Use the workspace venv and install dependencies:

```powershell
pip install -r Jalmar/requirements.txt
```

If you only need the CPU pipeline, install the base set instead:

```powershell
pip install -r Jalmar/requirements-base.txt
```

## Typical Commands

Pre-processing:

```powershell
python Jalmar/pre_processing/pre_processing.py
```

HDF5 feature extraction:

```powershell
python Jalmar/hdf5/main.py
```

Feature statistics:

```powershell
python Jalmar/stats/features/features_stats.py --hdf5 "C:/path/to/sleep_features.h5" --output_dir "Jalmar/stats/features/output"
```

## Notes

- Artifact epochs are encoded as score 5 during HDF5 creation.
- Feature statistics exclude score 5 by default unless `--include_artifacts` is set.
