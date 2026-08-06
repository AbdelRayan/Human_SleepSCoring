"""
Validation script for pre_processing module.

This script checks for issues in preprocessed data that could cause
problems in the HDF5 module (like zero-variance signals).
"""

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path


def validate_mat_file(mat_path, subject_name):
    """
    Validate a single .mat file for data quality.
    
    Parameters:
        mat_path (str): Path to .mat file.
        subject_name (str): Subject identifier.
    
    Returns:
        dict: Validation results.
    """
    results = {
        'file': os.path.basename(mat_path),
        'subject': subject_name,
        'is_valid': True,
        'issues': []
    }
    
    try:
        data = loadmat(mat_path)
    except Exception as e:
        results['is_valid'] = False
        results['issues'].append(f"Cannot load file: {str(e)}")
        return results
    
    # Find the data variable (skip special keys)
    data_var = None
    for key, value in data.items():
        if not key.startswith('__'):
            data_var = value
            break
    
    if data_var is None:
        results['is_valid'] = False
        results['issues'].append("No data variable found in .mat file")
        return results
    
    # Flatten to 1D for analysis
    data_flat = np.ravel(data_var)
    
    # Check 1: All zeros or NaNs
    if np.all(np.isnan(data_flat)):
        results['is_valid'] = False
        results['issues'].append("Data contains all NaNs")
    
    if np.all(data_flat == 0):
        results['is_valid'] = False
        results['issues'].append("Data contains all zeros")
    
    # Check 2: Constant values (zero variance)
    valid_data = data_flat[~np.isnan(data_flat)]
    if len(valid_data) > 0:
        std = np.std(valid_data)
        if std < 1e-10:
            results['is_valid'] = False
            results['issues'].append(f"Zero or very low variance: std={std:.2e}")
        elif std < 1e-6:
            results['issues'].append(f"Warning: Very low variance: std={std:.2e}")
    
    # Check 3: Excessive NaNs
    nan_ratio = np.sum(np.isnan(data_flat)) / len(data_flat)
    if nan_ratio > 0.1:
        results['issues'].append(f"Warning: {nan_ratio*100:.1f}% NaNs in data")
    
    # Check 4: Extreme values
    if len(valid_data) > 0:
        max_val = np.nanmax(np.abs(valid_data))
        if max_val > 1e6:
            results['issues'].append(f"Warning: Extreme values detected: max={max_val:.2e}")
    
    # Check 5: Data shape consistency
    results['shape'] = data_var.shape
    results['samples'] = len(valid_data)
    results['nans'] = np.sum(np.isnan(data_flat))
    results['std'] = np.std(valid_data) if len(valid_data) > 0 else np.nan
    results['min'] = np.nanmin(data_flat) if len(valid_data) > 0 else np.nan
    results['max'] = np.nanmax(data_flat) if len(valid_data) > 0 else np.nan
    
    return results


def validate_subject_files(subject_dir, subject_name):
    """
    Validate all .mat files for a subject.
    
    Parameters:
        subject_dir (str): Directory containing .mat files.
        subject_name (str): Subject identifier.
    
    Returns:
        dict: Validation results grouped by channel.
    """
    results = {
        'subject': subject_name,
        'directory': subject_dir,
        'channels': {},
        'all_valid': True
    }
    
    if not os.path.isdir(subject_dir):
        results['all_valid'] = False
        results['error'] = f"Directory not found: {subject_dir}"
        return results
    
    mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]
    
    if not mat_files:
        results['all_valid'] = False
        results['error'] = "No .mat files found"
        return results
    
    required_channels = ['Fpz', 'Pz', 'EMG', 'EOG', 'states']
    found_channels = set()
    
    for mat_file in mat_files:
        mat_path = os.path.join(subject_dir, mat_file)
        file_result = validate_mat_file(mat_path, subject_name)
        
        # Determine channel
        channel = None
        for ch in required_channels:
            if ch in mat_file:
                channel = ch
                found_channels.add(ch)
                break
        
        if channel:
            results['channels'][channel] = file_result
            if not file_result['is_valid']:
                results['all_valid'] = False
    
    # Check for missing channels
    missing = set(required_channels) - found_channels
    if missing:
        results['all_valid'] = False
        results['missing_channels'] = list(missing)
    
    return results


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


def validate_flat_directory(input_dir):
    """
    Validate flat directory structure with mixed files.
    
    Parameters:
        input_dir (str): Directory containing all .mat files.
    
    Returns:
        dict: Validation results by subject.
    """
    
    results = {
        'directory': input_dir,
        'structure': 'flat',
        'subjects': {},
        'summary': {}
    }
    
    if not os.path.isdir(input_dir):
        results['error'] = f"Directory not found: {input_dir}"
        return results
    
    # Group files by subject
    subjects_files = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            subject = extract_subject_name_from_filename(filename)
            if subject not in subjects_files:
                subjects_files[subject] = []
            subjects_files[subject].append(filename)
    
    # Validate each subject's files
    for subject_name in sorted(subjects_files.keys()):
        subject_result = {
            'files': subjects_files[subject_name],
            'channels': {},
            'all_valid': True
        }
        
        required_channels = {'Fpz': False, 'Pz': False, 'EMG': False, 'EOG': False, 'states': False}
        
        for mat_file in subjects_files[subject_name]:
            mat_path = os.path.join(input_dir, mat_file)
            file_result = validate_mat_file(mat_path, subject_name)
            
            # Categorize by channel
            for ch in required_channels.keys():
                if ch in mat_file:
                    required_channels[ch] = True
                    subject_result['channels'][ch] = file_result
                    if not file_result['is_valid']:
                        subject_result['all_valid'] = False
        
        subject_result['channels_found'] = {k: v for k, v in required_channels.items() if v}
        subject_result['channels_missing'] = [k for k, v in required_channels.items() if not v]
        
        if subject_result['channels_missing']:
            subject_result['all_valid'] = False
        
        results['subjects'][subject_name] = subject_result
    
    # Summary statistics
    total_subjects = len(subjects_files)
    valid_subjects = sum(1 for s in results['subjects'].values() if s['all_valid'])
    
    results['summary']['total_subjects'] = total_subjects
    results['summary']['valid_subjects'] = valid_subjects
    results['summary']['invalid_subjects'] = total_subjects - valid_subjects
    
    return results


def print_validation_report(results, verbose=True):
    """
    Print validation results in a readable format.
    
    Parameters:
        results (dict): Validation results from validate_*_directory().
        verbose (bool): Print detailed information about each issue.
    """
    print("\n" + "=" * 70)
    print("PRE-PROCESSING VALIDATION REPORT")
    print("=" * 70)
    
    if 'structure' in results and results['structure'] == 'flat':
        print(f"\nDirectory: {results['directory']}")
        print(f"Structure: Flat (all files in one directory)")
        print(f"\nSummary:")
        print(f"  Total subjects: {results['summary']['total_subjects']}")
        print(f"  Valid subjects: {results['summary']['valid_subjects']}")
        print(f"  Invalid subjects: {results['summary']['invalid_subjects']}")
        
        if verbose and results['summary']['invalid_subjects'] > 0:
            print("\nInvalid subjects:")
            for subject_name, subject_result in results['subjects'].items():
                if not subject_result['all_valid']:
                    print(f"\n  {subject_name}:")
                    if subject_result['channels_missing']:
                        print(f"    Missing channels: {subject_result['channels_missing']}")
                    for ch, ch_result in subject_result['channels'].items():
                        if not ch_result['is_valid']:
                            print(f"    {ch}:")
                            for issue in ch_result['issues']:
                                print(f"      - {issue}")
    else:
        # Hierarchical structure
        print(f"\nDirectory: {results['directory']}")
        print(f"Subject: {results['subject']}")
        print(f"Overall result: {'[PASS]' if results['all_valid'] else '[FAIL]'}")
        
        if 'missing_channels' in results:
            print(f"Missing channels: {results['missing_channels']}")
        
        if verbose:
            print("\nChannel-by-channel results:")
            for channel, ch_result in results['channels'].items():
                status = "[OK]" if ch_result['is_valid'] else "[FAIL]"
                print(f"  {channel}: {status}")
                if 'std' in ch_result:
                    print(f"    Shape: {ch_result['shape']}, Samples: {ch_result['samples']}")
                    print(f"    Std: {ch_result['std']:.2e}, Range: [{ch_result['min']:.2e}, {ch_result['max']:.2e}]")
                if ch_result['issues']:
                    for issue in ch_result['issues']:
                        print(f"    - {issue}")
    
    print("\n" + "=" * 70)


def main(input_dir=None, verbose=True):
    """
    Main validation function.
    
    Parameters:
        input_dir (str, optional): Directory to validate. If None, uses current dir.
        verbose (bool): Print detailed output.
    """
    if input_dir is None:
        input_dir = r'C:\Users\jalma\OneDrive - HAN\stage_donders\output'
    
    print(f"Validating preprocessed data in: {input_dir}")
    
    # Validate
    results = validate_flat_directory(input_dir)
    
    # Print report
    print_validation_report(results, verbose=verbose)
    
    return results


if __name__ == '__main__':
    import sys
    
    input_dir = r'C:\Users\jalma\OneDrive - HAN\stage_donders\output'
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    
    results = main(input_dir, verbose=True)
