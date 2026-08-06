"""
HDF5 creation module for sleep stage features.

This module handles conversion of preprocessed .mat files into HDF5 format
with computed features for machine learning applications.
"""

import importlib
import h5py
import numpy as np
import os
from scipy.io import loadmat
from artifacts_detection import remove_artifacts, artifact_epochs


def _get_compute_features(feature_module_name='computing_features'):
    """Resolve the feature computation entry point by module name."""
    module = importlib.import_module(feature_module_name)
    return module.compute_features


def load_mat_file(file_path, variable_name):
    """
    Load a .mat file and extract the specified variable.

    Parameters:
        file_path (str): Path to .mat file.
        variable_name (str): Variable name or substring to search for.

    Returns:
        numpy.ndarray: Loaded data.
    """
    mat_data = loadmat(file_path)
    
    # Find variable by exact name or substring match
    if variable_name in mat_data:
        return mat_data[variable_name]
    
    # Search for substring match
    for key, value in mat_data.items():
        if not key.startswith('__') and variable_name in key:
            return value
    
    raise ValueError(f"Variable '{variable_name}' not found in {file_path}")


def prepare_hdf5_data(
    subject_name, mat_files_dir, fs, epoch_length,
    emg_thresholds=None, eog_thresholds=None, eeg_thresholds=None,
    welch_nperseg=1024, feature_module_name='computing_features', save_raw=False
):
    """
    Prepare data for HDF5 storage from .mat files.

    Parameters:
        subject_name (str): Subject identifier.
        mat_files_dir (str): Directory containing .mat files.
        fs (float): Sampling frequency.
        epoch_length (int): Epoch length in seconds.
        emg_thresholds (list, optional): EMG artifact thresholds [lower, upper].
        eog_thresholds (list, optional): EOG artifact thresholds [lower, upper].
        eeg_thresholds (list, optional): EEG artifact thresholds [lower, upper].
        welch_nperseg (int): Welch window length for spectral analysis.
        feature_module_name (str): Name of the feature module to use.
        save_raw (bool): If True, also return raw features before normalization.

    Returns:
        If save_raw=False:
            tuple: (features, mapped_scores, subject_name)
        If save_raw=True:
            tuple: (features, mapped_scores, raw_features, subject_name)
    """
    # Set default thresholds
    if emg_thresholds is None:
        emg_thresholds = [9, 8]
    if eog_thresholds is None:
        eog_thresholds = [9, 8]
    if eeg_thresholds is None:
        eeg_thresholds = [9, 8]

    # Find .mat files in directory
    mat_files = {
        'fpz': None, 'pz': None, 'emg': None, 'eog': None, 'states': None
    }

    for filename in os.listdir(mat_files_dir):
        if not filename.endswith('.mat'):
            continue
        filepath = os.path.join(mat_files_dir, filename)
        if 'Fpz' in filename or 'fpz' in filename:
            mat_files['fpz'] = filepath
        elif 'Pz' in filename or 'pz' in filename:
            mat_files['pz'] = filepath
        elif 'EMG' in filename or 'emg' in filename:
            mat_files['emg'] = filepath
        elif 'EOG' in filename or 'eog' in filename:
            mat_files['eog'] = filepath
        elif 'states' in filename:
            mat_files['states'] = filepath

    # Check all required files found
    missing_files = [k for k, v in mat_files.items() if v is None]
    if missing_files:
        print(f"Warning: Missing files for {subject_name}: {missing_files}")
        return None

    # Load data
    print(f"Processing {subject_name}...")
    fpz_data = load_mat_file(mat_files['fpz'], 'Fpz')
    pz_data = load_mat_file(mat_files['pz'], 'Pz')
    emg_data = load_mat_file(mat_files['emg'], 'EMG')
    eog_data = load_mat_file(mat_files['eog'], 'EOG')
    states = load_mat_file(mat_files['states'], 'states')

    # Remove artifacts
    fpz_filt, fpz_artifact_idx = remove_artifacts(fpz_data, fs, eeg_thresholds)
    pz_filt, pz_artifact_idx = remove_artifacts(pz_data, fs, eeg_thresholds)
    emg_filt, emg_artifact_idx = remove_artifacts(emg_data, fs, emg_thresholds)
    eog_filt, eog_artifact_idx = remove_artifacts(eog_data, fs, eog_thresholds)

    states = np.ravel(states)

    # Compute features
    compute_features_fn = _get_compute_features(feature_module_name)
    result = compute_features_fn(
        fpz_filt, pz_filt, emg_filt, eog_filt, states, fs, epoch_length,
        welch_nperseg=welch_nperseg, return_raw=save_raw
    )
    
    # Unpack result based on return_raw flag
    if save_raw:
        features, mapped_scores, raw_features = result
    else:
        features, mapped_scores = result
        raw_features = None

    # Mark artifact epochs in scores
    window_length = fs * epoch_length
    fpz_arte_epochs = artifact_epochs(fpz_artifact_idx, int(window_length))
    pz_arte_epochs = artifact_epochs(pz_artifact_idx, int(window_length))
    emg_arte_epochs = artifact_epochs(emg_artifact_idx, int(window_length))
    eog_arte_epochs = artifact_epochs(eog_artifact_idx, int(window_length))

    artifact_indices = np.unique(np.concatenate((
        fpz_arte_epochs, pz_arte_epochs, emg_arte_epochs, eog_arte_epochs
    ))).astype(int)

    # Truncate scores and features to same length
    min_len = min(len(mapped_scores), len(features))
    mapped_scores = mapped_scores[:min_len]
    features = features[:min_len]

    # Mark artifacts (assume value 5 = movement/artifact)
    if len(artifact_indices) > 0:
        valid_indices = artifact_indices[artifact_indices < len(mapped_scores)]
        mapped_scores[valid_indices] = 5

    print(f"Computed {len(features)} epochs for {subject_name}")
    if save_raw:
        # Truncate raw_features to same length as features
        raw_features = raw_features[:min_len]
        return features, mapped_scores, raw_features, subject_name
    else:
        return features, mapped_scores, subject_name


def create_hdf5_file(output_path, mode='w'):
    """
    Create a new HDF5 file for storing sleep features.

    Parameters:
        output_path (str): Path to output HDF5 file.
        mode (str): File mode ('w' for write, 'a' for append).

    Returns:
        h5py.File: Open HDF5 file object.
    """
    return h5py.File(output_path, mode)


def add_subject_to_hdf5(hdf5_file, features, mapped_scores, subject_name):
    """
    Add subject data to HDF5 file.

    Parameters:
        hdf5_file (h5py.File): Open HDF5 file.
        features (numpy.ndarray): Feature array (normalized).
        mapped_scores (numpy.ndarray): Sleep stage scores.
        subject_name (str): Subject identifier.
    """
    group = hdf5_file.create_group(str(subject_name))
    group.attrs['n_features'] = features.shape[1]
    group.attrs['n_epochs'] = features.shape[0]
    group.attrs['description_features'] = (
        'Index_W, Index_R, Index_N, Index_1, Index_2, Index_3, Index_4, '
        'Delta, Theta, Aperiodic, DFA, MSE, EOG, Index_R_noEOG, Index_N_noEOG'
    )
    group.attrs['description_scores'] = (
        'Sleep stages: 0=Awake, 1=N1, 2=N2, 3=N3, 4=REM, 5=Movement/Artifact'
    )
    group.create_dataset('features', data=features, dtype='float32')
    group.create_dataset('scores', data=mapped_scores, dtype='uint8')


def add_subject_to_hdf5_raw(hdf5_file, raw_features, mapped_scores, subject_name):
    """
    Add raw subject data to HDF5 file.

    Parameters:
        hdf5_file (h5py.File): Open HDF5 file.
        raw_features (numpy.ndarray): Raw feature array (before normalization).
        mapped_scores (numpy.ndarray): Sleep stage scores.
        subject_name (str): Subject identifier.
    """
    group = hdf5_file.create_group(str(subject_name))
    group.attrs['n_features'] = raw_features.shape[1]
    group.attrs['n_epochs'] = raw_features.shape[0]
    group.attrs['description_features'] = (
        'Index_W, Index_R, Index_N, Index_1, Index_2, Index_3, Index_4, '
        'Delta, Theta, Aperiodic, DFA, MSE, EOG, Index_R_noEOG, Index_N_noEOG'
    )
    group.attrs['description_scores'] = (
        'Sleep stages: 0=Awake, 1=N1, 2=N2, 3=N3, 4=REM, 5=Movement/Artifact'
    )
    group.attrs['note'] = 'Raw features before Wei percentile normalization and smoothing'
    group.create_dataset('features', data=raw_features, dtype='float32')
    group.create_dataset('scores', data=mapped_scores, dtype='uint8')


def process_subjects(
    subjects_list, data_dir, output_hdf5_path, fs, epoch_length,
    emg_thresholds=None, eog_thresholds=None, eeg_thresholds=None,
    welch_nperseg=1024, feature_module_name='computing_features', save_raw=False
):
    """
    Process multiple subjects and save to HDF5.

    Parameters:
        subjects_list (list): List of subject names or directories.
        data_dir (str): Root directory containing subject data.
        output_hdf5_path (str): Path to output HDF5 file.
        fs (float): Sampling frequency.
        epoch_length (int): Epoch length in seconds.
        emg_thresholds (list, optional): EMG artifact thresholds.
        eog_thresholds (list, optional): EOG artifact thresholds.
        eeg_thresholds (list, optional): EEG artifact thresholds.
        welch_nperseg (int): Welch window length for spectral analysis.
        feature_module_name (str): Name of the feature module to use.
        save_raw (bool): If True, also save raw features to separate HDF5 file.
    """
    # Generate raw features file path if needed
    raw_hdf5_path = None
    if save_raw:
        base, ext = os.path.splitext(output_hdf5_path)
        raw_hdf5_path = base + '_raw' + ext
    
    with create_hdf5_file(output_hdf5_path, mode='w') as hdf5_file:
        raw_hdf5_file = create_hdf5_file(raw_hdf5_path, mode='w') if save_raw else None
        
        try:
            for subject_name in subjects_list:
                subject_dir = os.path.join(data_dir, subject_name)
                if not os.path.isdir(subject_dir):
                    print(f"Skipping {subject_name}: directory not found")
                    continue

                result = prepare_hdf5_data(
                    subject_name, subject_dir, fs, epoch_length,
                    emg_thresholds, eog_thresholds, eeg_thresholds,
                    welch_nperseg=welch_nperseg,
                    feature_module_name=feature_module_name,
                    save_raw=save_raw
                )

                if result is not None:
                    if save_raw:
                        features, scores, raw_features, subj_name = result
                        add_subject_to_hdf5(hdf5_file, features, scores, subj_name)
                        add_subject_to_hdf5_raw(raw_hdf5_file, raw_features, scores, subj_name)
                    else:
                        features, scores, subj_name = result
                        add_subject_to_hdf5(hdf5_file, features, scores, subj_name)
                    print(f"Successfully added {subj_name} to HDF5")
                else:
                    print(f"Failed to process {subject_name}")
        finally:
            if raw_hdf5_file is not None:
                raw_hdf5_file.close()


def process_single_subject(
    subject_name, mat_files_dir, output_hdf5_path, fs=100, epoch_length=30,
    emg_thresholds=None, eog_thresholds=None, eeg_thresholds=None,
    welch_nperseg=1024, feature_module_name='computing_features', save_raw=False
):
    """
    Process a single subject and save to HDF5.

    Parameters:
        subject_name (str): Subject identifier.
        mat_files_dir (str): Directory containing .mat files for the subject.
        output_hdf5_path (str): Path to output HDF5 file.
        fs (float): Sampling frequency.
        epoch_length (int): Epoch length in seconds.
        emg_thresholds (list, optional): EMG artifact thresholds.
        eog_thresholds (list, optional): EOG artifact thresholds.
        eeg_thresholds (list, optional): EEG artifact thresholds.
        welch_nperseg (int): Welch window length for spectral analysis.
        feature_module_name (str): Name of the feature module to use.
        save_raw (bool): If True, also save raw features to separate HDF5 file.
    """
    result = prepare_hdf5_data(
        subject_name, mat_files_dir, fs, epoch_length,
        emg_thresholds, eog_thresholds, eeg_thresholds,
        welch_nperseg=welch_nperseg,
        feature_module_name=feature_module_name,
        save_raw=save_raw
    )

    if result is None:
        print(f"Failed to prepare data for {subject_name}")
        return

    if save_raw:
        features, scores, raw_features, subj_name = result
    else:
        features, scores, subj_name = result
        raw_features = None

    # Check if file exists
    mode = 'a' if os.path.exists(output_hdf5_path) else 'w'
    with create_hdf5_file(output_hdf5_path, mode=mode) as hdf5_file:
        # Check if subject already exists
        if str(subj_name) in hdf5_file:
            del hdf5_file[str(subj_name)]
        add_subject_to_hdf5(hdf5_file, features, scores, subj_name)
        print(f"Successfully saved {subj_name} to {output_hdf5_path}")
    
    # Save raw features to separate file if requested
    if save_raw and raw_features is not None:
        base, ext = os.path.splitext(output_hdf5_path)
        raw_hdf5_path = base + '_raw' + ext
        mode = 'a' if os.path.exists(raw_hdf5_path) else 'w'
        with create_hdf5_file(raw_hdf5_path, mode=mode) as raw_hdf5_file:
            if str(subj_name) in raw_hdf5_file:
                del raw_hdf5_file[str(subj_name)]
            add_subject_to_hdf5_raw(raw_hdf5_file, raw_features, scores, subj_name)
            print(f"Successfully saved raw features for {subj_name} to {raw_hdf5_path}")
