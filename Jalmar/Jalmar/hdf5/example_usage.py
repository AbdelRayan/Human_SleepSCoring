"""
Example script for HDF5 feature extraction.

This script demonstrates how to use the Jalmar HDF5 module to process
preprocessed sleep data and create HDF5 files for machine learning.
"""

import os
from hdf5_creation import process_single_subject, process_subjects


def example_single_subject():
    """Example: Process a single subject."""
    # Configuration
    subject_name = 'SC4001'
    mat_files_dir = r'C:\Users\jalma\OneDrive - HAN\stage_donders\output\SC4001'
    output_hdf5 = r'C:\Users\jalma\OneDrive - HAN\stage_donders\output\sleep_features.h5'
    
    # Processing parameters
    fs = 100  # Sampling frequency (Hz)
    epoch_length = 30  # Epoch length (seconds)
    
    # Process the subject
    process_single_subject(
        subject_name=subject_name,
        mat_files_dir=mat_files_dir,
        output_hdf5_path=output_hdf5,
        fs=fs,
        epoch_length=epoch_length,
        # Optional: customize artifact detection thresholds
        # emg_thresholds=[9, 8],
        # eog_thresholds=[9, 8],
        # eeg_thresholds=[9, 8]
    )
    
    print(f"Successfully created {output_hdf5}")


def example_batch_processing():
    """Example: Process multiple subjects."""
    # Configuration
    output_dir = r'C:\Users\jalma\OneDrive - HAN\stage_donders\output'
    output_hdf5 = os.path.join(output_dir, 'sleep_features_batch.h5')
    
    # List of subjects to process
    subjects = [
        'SC4001', 'SC4002', 'SC4011', 'SC4021',
        'SC4031', 'SC4051', 'SC4061', 'SC4071'
    ]
    
    # Processing parameters
    fs = 100
    epoch_length = 30
    
    # Process all subjects
    process_subjects(
        subjects_list=subjects,
        data_dir=output_dir,
        output_hdf5_path=output_hdf5,
        fs=fs,
        epoch_length=epoch_length
    )
    
    print(f"Successfully created {output_hdf5} with {len(subjects)} subjects")


def example_read_hdf5():
    """Example: Read data from HDF5 file."""
    import h5py
    import numpy as np
    
    hdf5_path = r'C:\Users\jalma\OneDrive - HAN\stage_donders\output\sleep_features.h5'
    
    with h5py.File(hdf5_path, 'r') as f:
        # List all subjects
        subjects = list(f.keys())
        print(f"Found {len(subjects)} subjects: {subjects}")
        
        # Process each subject
        for subject_name in subjects:
            subject_group = f[subject_name]
            
            # Read features and scores
            features = subject_group['features'][:]
            scores = subject_group['scores'][:]
            
            # Print information
            print(f"\n{subject_name}:")
            print(f"  Features shape: {features.shape}")
            print(f"  Scores shape: {scores.shape}")
            print(f"  Features description: {subject_group.attrs['description_features']}")
            
            # Statistics
            n_epochs_per_stage = np.bincount(scores)
            stage_names = ['Awake', 'N1', 'N2', 'N3', 'REM', 'Movement/Artifact']
            print("  Epochs per stage:")
            for stage, count in enumerate(n_epochs_per_stage):
                if stage < len(stage_names):
                    print(f"    {stage_names[stage]}: {count}")


if __name__ == '__main__':
    # Run examples
    print("=" * 60)
    print("Example 1: Single Subject Processing")
    print("=" * 60)
    # example_single_subject()
    
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    # example_batch_processing()
    
    print("\n" + "=" * 60)
    print("Example 3: Reading HDF5 File")
    print("=" * 60)
    # example_read_hdf5()
    
    print("\nTo run examples, uncomment the function calls above.")
