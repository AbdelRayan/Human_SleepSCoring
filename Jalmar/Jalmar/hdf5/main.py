"""
Main script for HDF5 feature extraction pipeline.

This script orchestrates the complete workflow:
1. Discovers preprocessed .mat files for subjects
2. Automatically detects directory structure (flat or hierarchical)
3. Computes features from raw signals
4. Detects and marks artifacts
5. Stores everything in HDF5 format

Supports both directory structures:
- Flat: All .mat files in a single directory (named by subject, e.g., SC4001_Fpz-Cz.mat)
- Hierarchical: Organized in subdirectories per subject (SC4001/SC4001_Fpz-Cz.mat)

Usage:
    python main.py --input_dir /path/to/preprocessed_data --output /path/to/output.h5
    
Or edit the configuration section below and run:
    python main.py
"""

import os
import argparse
import importlib
import sys
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

from hdf5_creation import process_single_subject, create_hdf5_file, add_subject_to_hdf5, add_subject_to_hdf5_raw, prepare_hdf5_data


def _get_compute_features(feature_module_name='computing_features'):
    """Resolve the feature computation entry point by module name."""
    module = importlib.import_module(feature_module_name)
    return module.compute_features


# Configure logging
def setup_logging(log_file=None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION SECTION - Edit these parameters
# ============================================================================

CONFIG = {
    # Input/Output paths
    # input_directory can be either:
    # - Flat structure: single folder with all .mat files (auto-detected)
    # - Hierarchical: folder with subject subdirectories containing .mat files
    'input_directory': r'C:\\Users\\jalma\\OneDrive - HAN\stage_donders\\output',
    'output_hdf5': r'C:\\Users\\jalma\\OneDrive - HAN\stage_donders\\features\\sleep_features.h5',
    'log_file': r'C:\\Users\\jalma\\OneDrive - HAN\stage_donders\\features\\hdf5_processing.log',
    
    # Processing parameters
    'sampling_frequency': 100,      # Hz
    # 'epoch_length': 30,             # seconds
    'epoch_length': 15,             # seconds
    # 'epoch_length': 10,             # seconds
    'welch_nperseg': 1024,          # max Welch segment length for aperiodic PSD
    'mode': 'a',                    # 'w' = overwrite, 'a' = append
    # 'feature_version': 'standard',  # 'standard' or 'relaxed'
    'feature_version': 'relaxed',  # 'standard' or 'relaxed'
    
    # Artifact detection thresholds (in standard deviations)
    'emg_thresholds': [9, 8],       # [lower, upper]
    'eog_thresholds': [9, 8],       # [lower, upper]
    'eeg_thresholds': [9, 8],       # [lower, upper]
    
    # Subject selection
    'subjects': None,               # None = process all, or list specific subjects
    # 'subjects': ['SC4001', 'SC4002', 'SC4011', 'SC4021'],  # Example
    
    # Processing options
    'skip_existing': True,          # Skip subjects already in HDF5 (resume-safe default)
    'save_raw': True,               # Save raw features to a separate _raw HDF5 file
    'verbose': True,
}

# ============================================================================


def detect_directory_structure(directory):
    """
    Detect if directory uses hierarchical (subdirectories) or flat structure.
    
    Parameters:
        directory (str): Path to directory to analyze.
    
    Returns:
        str: 'hierarchical' or 'flat'
    """
    mat_files_in_root = [f for f in os.listdir(directory) if f.endswith('.mat')]
    subdirs_with_mat = 0
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            mat_in_dir = [f for f in os.listdir(item_path) if f.endswith('.mat')]
            if mat_in_dir:
                subdirs_with_mat += 1
    
    if len(mat_files_in_root) > 0:
        return 'flat'
    elif subdirs_with_mat > 0:
        return 'hierarchical'
    else:
        return 'unknown'


def extract_subject_name_from_filename(filename):
    """
    Extract subject name from .mat filename.
    
    Handles patterns like: SC4001_Fpz-Cz.mat, subject005_EMG.mat, etc.
    
    Parameters:
        filename (str): .mat filename.
    
    Returns:
        str: Extracted subject name.
    """
    # Remove .mat extension
    name = filename.replace('.mat', '')
    
    # Split by underscore and take the first part
    parts = name.split('_')
    subject = parts[0]
    
    return subject


def find_subjects_hierarchical(directory):
    """
    Discover subject directories in hierarchical structure.
    
    Parameters:
        directory (str): Path to directory containing subject subdirectories.
    
    Returns:
        list: List of subject directory names.
    """
    subjects = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # Check if directory contains .mat files
            mat_files = [f for f in os.listdir(item_path) if f.endswith('.mat')]
            if mat_files:
                subjects.append(item)
    
    return sorted(subjects)


def find_subjects_flat(directory):
    """
    Discover subjects from .mat files in flat directory structure.
    
    Groups files by extracting subject name from filenames.
    
    Parameters:
        directory (str): Path to directory containing all .mat files.
    
    Returns:
        dict: Mapping of subject names to their .mat files.
    """
    subjects_files = {}
    
    for filename in os.listdir(directory):
        if not filename.endswith('.mat'):
            continue
        
        subject = extract_subject_name_from_filename(filename)
        if subject not in subjects_files:
            subjects_files[subject] = []
        subjects_files[subject].append(filename)
    
    return subjects_files


def find_subjects(directory):
    """
    Discover subjects in input directory (handles both hierarchical and flat structures).
    
    Parameters:
        directory (str): Path to directory containing subject data.
    
    Returns:
        tuple: (structure_type, subjects_list_or_dict)
            - 'hierarchical': list of subdirectory names
            - 'flat': dict of {subject_name: [filenames]}
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    structure = detect_directory_structure(directory)
    
    if structure == 'flat':
        subjects = find_subjects_flat(directory)
        return 'flat', subjects
    else:
        subjects = find_subjects_hierarchical(directory)
        return 'hierarchical', subjects


def validate_subject_data(subject_data):
    """
    Check if a subject has all required .mat files.
    
    Parameters:
        subject_data: Either a directory path (str) or list of filenames (list)
    
    Returns:
        tuple: (is_valid, missing_files_list)
    """
    required_keywords = ['Fpz', 'Pz', 'EMG', 'EOG', 'states']
    found = {key: False for key in required_keywords}
    
    if isinstance(subject_data, str):
        # Directory path
        if not os.path.isdir(subject_data):
            return False, required_keywords
        filenames = os.listdir(subject_data)
    else:
        # List of filenames
        filenames = subject_data
    
    for filename in filenames:
        if filename.endswith('.mat'):
            for key in required_keywords:
                if key in filename:
                    found[key] = True
    
    missing = [k for k, v in found.items() if not v]
    is_valid = len(missing) == 0
    
    return is_valid, missing


def prepare_hdf5_data_flat(
    subject_name, input_dir, subject_files, fs, epoch_length,
    emg_thresholds=None, eog_thresholds=None, eeg_thresholds=None,
    welch_nperseg=1024, feature_module_name='computing_features', save_raw=False
):
    """
    Prepare data for HDF5 from flat directory structure.
    
    This function loads .mat files for a subject directly from a flat directory
    (all files in one folder, not organized by subject subdirectories).
    
    Parameters:
        subject_name (str): Subject identifier.
        input_dir (str): Root directory containing all .mat files.
        subject_files (list): List of filenames for this subject.
        fs (float): Sampling frequency.
        epoch_length (int): Epoch length in seconds.
        emg_thresholds (list, optional): EMG artifact thresholds.
        eog_thresholds (list, optional): EOG artifact thresholds.
        eeg_thresholds (list, optional): EEG artifact thresholds.
        welch_nperseg (int): Welch window length for spectral analysis.
        feature_module_name (str): Name of the feature module to use.
        save_raw (bool): If True, also return raw features before normalization.
    
    Returns:
        If save_raw=False:
            tuple: (features, mapped_scores, subject_name) or None if failed.
        If save_raw=True:
            tuple: (features, mapped_scores, raw_features, subject_name) or None if failed.
    """
    from scipy.io import loadmat
    from artifacts_detection import remove_artifacts, artifact_epochs
    
    # Set default thresholds
    if emg_thresholds is None:
        emg_thresholds = [9, 8]
    if eog_thresholds is None:
        eog_thresholds = [9, 8]
    if eeg_thresholds is None:
        eeg_thresholds = [9, 8]
    
    # Find the required files
    mat_files = {
        'fpz': None, 'pz': None, 'emg': None, 'eog': None, 'states': None
    }
    
    for filename in subject_files:
        if 'Fpz' in filename or 'fpz' in filename:
            mat_files['fpz'] = os.path.join(input_dir, filename)
        elif 'Pz' in filename or 'pz' in filename:
            mat_files['pz'] = os.path.join(input_dir, filename)
        elif 'EMG' in filename or 'emg' in filename:
            mat_files['emg'] = os.path.join(input_dir, filename)
        elif 'EOG' in filename or 'eog' in filename:
            mat_files['eog'] = os.path.join(input_dir, filename)
        elif 'states' in filename:
            mat_files['states'] = os.path.join(input_dir, filename)
    
    # Check all required files found
    missing_files = [k for k, v in mat_files.items() if v is None]
    if missing_files:
        return None
    
    # Load data
    fpz_data = loadmat(mat_files['fpz'])
    pz_data = loadmat(mat_files['pz'])
    emg_data = loadmat(mat_files['emg'])
    eog_data = loadmat(mat_files['eog'])
    states = loadmat(mat_files['states'])
    
    # Extract actual arrays from loaded dicts
    # The loadmat returns dicts with variable names as keys
    fpz_data = next(v for k, v in fpz_data.items() if not k.startswith('__') and ('Fpz' in k or 'fpz' in k))
    pz_data = next(v for k, v in pz_data.items() if not k.startswith('__') and ('Pz' in k or 'pz' in k))
    emg_data = next(v for k, v in emg_data.items() if not k.startswith('__') and ('EMG' in k or 'emg' in k))
    eog_data = next(v for k, v in eog_data.items() if not k.startswith('__') and ('EOG' in k or 'eog' in k))
    states = next(v for k, v in states.items() if not k.startswith('__') and 'state' in k.lower())
    
    # Validate signal quality
    fpz_std = np.nanstd(fpz_data)
    pz_std = np.nanstd(pz_data)
    emg_std = np.nanstd(emg_data)
    eog_std = np.nanstd(eog_data)
    
    if fpz_std == 0 or fpz_std < 1e-6 or np.isnan(fpz_std):
        print(f"Warning: Fpz signal has zero or very low variance ({fpz_std:.2e}) for {subject_name}")
    if pz_std == 0 or pz_std < 1e-6 or np.isnan(pz_std):
        print(f"Warning: Pz signal has zero or very low variance ({pz_std:.2e}) for {subject_name}")
    if emg_std == 0 or emg_std < 1e-6 or np.isnan(emg_std):
        print(f"Warning: EMG signal has zero or very low variance ({emg_std:.2e}) for {subject_name}")
    if eog_std == 0 or eog_std < 1e-6 or np.isnan(eog_std):
        print(f"Warning: EOG signal has zero or very low variance ({eog_std:.2e}) for {subject_name}")
    
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
    if save_raw:
        raw_features = raw_features[:min_len]
    
    # Mark artifacts (assume value 5 = movement/artifact)
    if len(artifact_indices) > 0:
        valid_indices = artifact_indices[artifact_indices < len(mapped_scores)]
        mapped_scores[valid_indices] = 5
    
    if save_raw:
        return features, mapped_scores, raw_features, subject_name
    else:
        return features, mapped_scores, subject_name


def process_hdf5_pipeline(config, logger):
    """
    Main pipeline: discover subjects, process them, and create HDF5.
    
    Handles both hierarchical (subdirectories) and flat (single directory) structures.
    
    Parameters:
        config (dict): Configuration dictionary.
        logger: Logger instance.
    """
    logger.info("=" * 70)
    logger.info("Starting HDF5 Feature Extraction Pipeline")
    logger.info("=" * 70)
    
    # Validate input directory
    input_dir = config['input_directory']
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    # Detect directory structure and find subjects
    structure_type, subjects_data = find_subjects(input_dir)
    logger.info(f"Detected directory structure: {structure_type}")
    logger.info(f"Feature version: {config['feature_version']}")

    feature_module_name = 'computing_features_relaxed' if config['feature_version'] == 'relaxed' else 'computing_features'
    logger.info(f"Feature module: {feature_module_name}")
    
    if structure_type == 'flat':
        all_subjects = list(subjects_data.keys())
        logger.info(f"Found {len(all_subjects)} subjects from .mat files in {input_dir}")
    else:
        all_subjects = subjects_data
        logger.info(f"Found {len(all_subjects)} subject directories in {input_dir}")
    
    # Select subjects to process
    if config['subjects'] is not None:
        subjects_to_process = [s for s in config['subjects'] if s in all_subjects]
        skipped_subjects = [s for s in config['subjects'] if s not in all_subjects]
        if skipped_subjects:
            logger.warning(f"Requested subjects not found: {skipped_subjects}")
    else:
        subjects_to_process = all_subjects
    
    logger.info(f"Processing {len(subjects_to_process)} subjects...")
    save_raw = config.get('save_raw', False)
    
    # Create HDF5 file
    output_path = config['output_hdf5']
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Check for existing file
    if config['mode'] == 'a' and os.path.exists(output_path):
        logger.info(f"Appending to existing HDF5: {output_path}")
    elif config['mode'] == 'w':
        logger.info(f"Creating new HDF5 file: {output_path}")
    
    # Generate raw features file path if needed
    raw_hdf5_path = None
    if save_raw:
        base, ext = os.path.splitext(output_path)
        raw_hdf5_path = base + '_raw' + ext
        logger.info(f"Raw features will be saved to: {raw_hdf5_path}")
    
    # Process each subject
    successful = 0
    failed = 0
    skipped = 0
    
    with create_hdf5_file(output_path, mode=config['mode']) as hdf5_file:
        raw_hdf5_file = create_hdf5_file(raw_hdf5_path, mode=config['mode']) if save_raw else None
        
        try:
            for i, subject_name in enumerate(subjects_to_process, 1):
                # Determine subject data location based on structure
                if structure_type == 'flat':
                    subject_files = subjects_data[subject_name]
                    subject_dir = input_dir
                else:
                    subject_dir = os.path.join(input_dir, subject_name)
                    subject_files = subject_dir
                
                # Validate subject data
                is_valid, missing = validate_subject_data(subject_files)
                if not is_valid:
                    logger.warning(
                        f"[{i}/{len(subjects_to_process)}] {subject_name}: "
                        f"Missing files: {missing}"
                    )
                    failed += 1
                    continue
                
                # Check if should skip existing
                if config['skip_existing'] and str(subject_name) in hdf5_file:
                    logger.info(f"[{i}/{len(subjects_to_process)}] {subject_name}: Already in HDF5, skipping")
                    skipped += 1
                    continue
                
                # Process subject
                logger.info(f"[{i}/{len(subjects_to_process)}] Processing {subject_name}...")
                try:
                    # For flat structure, we pass the input_dir but with a modified prepare_hdf5_data
                    if structure_type == 'flat':
                        # Need to use a temporary directory or modified approach
                        result = prepare_hdf5_data_flat(
                            subject_name,
                            input_dir,
                            subject_files,
                            config['sampling_frequency'],
                            config['epoch_length'],
                            config['emg_thresholds'],
                            config['eog_thresholds'],
                            config['eeg_thresholds'],
                            config['welch_nperseg'],
                            feature_module_name=feature_module_name,
                            save_raw=save_raw
                        )
                    else:
                        result = prepare_hdf5_data(
                            subject_name,
                            subject_dir,
                            config['sampling_frequency'],
                            config['epoch_length'],
                            config['emg_thresholds'],
                            config['eog_thresholds'],
                            config['eeg_thresholds'],
                            config['welch_nperseg'],
                            feature_module_name=feature_module_name,
                            save_raw=save_raw
                        )
                    
                    if result is not None:
                        if save_raw:
                            features, scores, raw_features, subj_name = result
                        else:
                            features, scores, subj_name = result
                            raw_features = None
                        
                        # Remove if already exists (when appending)
                        if str(subj_name) in hdf5_file:
                            del hdf5_file[str(subj_name)]
                        if save_raw and str(subj_name) in raw_hdf5_file:
                            del raw_hdf5_file[str(subj_name)]
                        
                        # Add to HDF5
                        add_subject_to_hdf5(hdf5_file, features, scores, subj_name)
                        if save_raw and raw_features is not None:
                            add_subject_to_hdf5_raw(raw_hdf5_file, raw_features, scores, subj_name)
                        # Ensure each completed subject is safely written for resume after interruption.
                        hdf5_file.flush()
                        if raw_hdf5_file is not None:
                            raw_hdf5_file.flush()
                        logger.info(f"  [OK] Successfully added to HDF5")
                        successful += 1
                    else:
                        logger.error(f"  [FAIL] Failed to prepare data")
                        failed += 1
                
                except Exception as e:
                    logger.error(f"  [FAIL] Error processing {subject_name}: {str(e)}")
                    if config['verbose']:
                        import traceback
                        logger.error(traceback.format_exc())
                    failed += 1
        finally:
            if raw_hdf5_file is not None:
                raw_hdf5_file.close()
                logger.info(f"Closed raw features file: {raw_hdf5_path}")
    
    # Summary
    logger.info("=" * 70)
    logger.info("Pipeline Summary")
    logger.info("=" * 70)
    logger.info(f"Total subjects found: {len(subjects_to_process)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Output file: {output_path}")
    if save_raw:
        logger.info(f"Raw features file: {raw_hdf5_path}")
    
    return successful > 0


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="HDF5 Feature Extraction Pipeline for Sleep EEG Data"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to directory containing subject subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output HDF5 file'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Specific subjects to process (space-separated)'
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=100,
        help='Sampling frequency (Hz)'
    )
    parser.add_argument(
        '--epoch_length',
        type=int,
        default=None,
        help='Epoch length (seconds)'
    )
    parser.add_argument(
        '--welch_nperseg',
        type=int,
        default=None,
        help='Maximum Welch nperseg for aperiodic PSD (capped by epoch samples)'
    )
    parser.add_argument(
        '--feature_version',
        choices=['standard', 'relaxed'],
        default=None,
        help='Choose which computing_features module to use'
    )
    parser.add_argument(
        '--mode',
        choices=['w', 'a'],
        default=None,
        help="File mode: 'w' to overwrite, 'a' to append (default from CONFIG)"
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip subjects already in HDF5 file'
    )
    parser.add_argument(
        '--config_only',
        action='store_true',
        help='Print configuration and exit (for debugging)'
    )
    parser.add_argument(
        '--save_raw',
        action='store_true',
        help='Save raw features (before normalization) in addition to processed features'
    )
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    if args.input_dir:
        CONFIG['input_directory'] = args.input_dir
    if args.output:
        CONFIG['output_hdf5'] = args.output
    if args.subjects:
        CONFIG['subjects'] = args.subjects
    CONFIG['sampling_frequency'] = args.fs
    if args.epoch_length is not None:
        CONFIG['epoch_length'] = args.epoch_length
    if args.welch_nperseg is not None:
        CONFIG['welch_nperseg'] = args.welch_nperseg
    if args.feature_version:
        CONFIG['feature_version'] = args.feature_version
    if args.mode:
        CONFIG['mode'] = args.mode
    if args.skip_existing:
        CONFIG['skip_existing'] = True
    if args.save_raw:
        CONFIG['save_raw'] = True
    
    # Setup logging
    logger = setup_logging(CONFIG['log_file'])
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Input directory: {CONFIG['input_directory']}")
    logger.info(f"  Output HDF5: {CONFIG['output_hdf5']}")
    logger.info(f"  Sampling frequency: {CONFIG['sampling_frequency']} Hz")
    logger.info(f"  Epoch length: {CONFIG['epoch_length']} s")
    logger.info(f"  Welch nperseg: {CONFIG['welch_nperseg']}")
    logger.info(f"  File mode: {CONFIG['mode']}")
    logger.info(f"  Skip existing: {CONFIG['skip_existing']}")
    logger.info(f"  Save raw features: {CONFIG.get('save_raw', False)}")
    logger.info(f"  Feature version: {CONFIG['feature_version']}")
    
    if args.config_only:
        logger.info("(Config-only mode, exiting)")
        return
    
    # Run pipeline
    success = process_hdf5_pipeline(CONFIG, logger)
    
    if success:
        logger.info("\n[SUCCESS] Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n[FAILED] Pipeline failed or no subjects processed")
        sys.exit(1)


if __name__ == '__main__':
    main()
