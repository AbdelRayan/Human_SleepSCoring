#!/usr/bin/env python3
"""
Optional example workflow for Jalmar mcRBM.

This is a demo-only script. The actual Jalmar workflow is:
pre_processing -> hdf5 -> stats/features -> mcRBM.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add scripts to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from mcrbm import mcRBM
from infer_states import mcRBMInference
from data_preproc import DataPreproc
import h5py


def load_hdf5_features(hdf5_path, output_file='mcrbm_input.npz', max_samples=None):
    """
    Load features from HDF5 file (output of Jalmar/hdf5 pipeline).
    
    Args:
        hdf5_path: Path to HDF5 file
        output_file: Where to save .npz for mcRBM
        max_samples: Limit number of samples (for testing)
        
    Returns:
        Path to created .npz file
    """
    print(f"Loading features from: {hdf5_path}")
    
    features_list = []
    labels_list = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # List available groups
        print(f"Available groups: {list(f.keys())}")
        
        for subject_id in f.keys():
            subject_group = f[subject_id]
            
            if 'features' in subject_group:
                features = subject_group['features'][:]
                if 'hypnogram' in subject_group:
                    labels = subject_group['hypnogram'][:]
                else:
                    labels = np.zeros(features.shape[0])
                
                print(f"  {subject_id}: {features.shape}")
                
                features_list.append(features)
                labels_list.append(labels)
    
    # Concatenate all subjects
    X = np.vstack(features_list).astype(np.float32)
    y = np.concatenate(labels_list).astype(np.float32)
    
    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]
    
    print(f"\nCombined dataset: {X.shape}")
    
    # Save as .npz
    np.savez(output_file, d=X, epochsLinked=y, epochTime=np.zeros((X.shape[0], 1)))
    print(f"Saved to: {output_file}")
    
    return output_file


def example_full_workflow():
    """
    Complete workflow example: load data, train, infer, analyze.
    """
    print("=" * 60)
    print("mcRBM Full Workflow Example")
    print("=" * 60)
    
    # Step 0: Load data from HDF5 (if available)
    hdf5_path = 'C:/Users/jalma/OneDrive - HAN/stage_donders/features/sleep_features.h5'
    if os.path.exists(hdf5_path):
        print("\n[Step 0] Loading data from HDF5...")
        data_file = load_hdf5_features(hdf5_path, output_file='mcrbm_input.npz', max_samples=50000)
    else:
        print("\n[Step 0] Using example synthetic data...")
        # Create synthetic data
        np.random.seed(42)
        n_samples = 10000
        n_features = 13  # Typical feature count
        
        # Create realistic-ish feature distributions
        X = np.random.randn(n_samples, n_features).astype(np.float32) * 0.3 + 0.5
        X = np.clip(X, 0.01, 1.0)
        y = np.random.randint(0, 6, n_samples).astype(np.float32)  # 0-5 sleep stages
        
        data_file = 'mcrbm_input.npz'
        np.savez(data_file, d=X, epochsLinked=y, epochTime=np.zeros((n_samples, 1)))
        print(f"Created synthetic data: {X.shape}")
    
    # Step 1: Preprocess data
    print("\n[Step 1] Preprocessing data...")
    dpp = DataPreproc()
    data = np.load(data_file)
    d = data['d']
    obsKeys = data['epochsLinked']
    
    # Optional preprocessing
    d_proc, obsKeys, stats = dpp.preprocAndScaleData(
        d, obsKeys,
        logFlag=True,
        meanSubtractionFlag=True,
        scalingFlag=True,
        scaling='standard',
        pcaFlag=False,
        whitenFlag=False,
        rescalingFlag=False,
        rescaling='standard',
        minmaxFile='minmax_stats.pkl',
        saveDir='./experiments/example'
    )
    
    # Save processed data
    np.savez('mcrbm_processed.npz', d=d_proc, epochsLinked=obsKeys, epochTime=np.zeros((d_proc.shape[0], 1)))
    print(f"Processed data: {d_proc.shape}")
    
    # Step 2: Update config and train
    print("\n[Step 2] Training mcRBM...")
    print("Note: Set dsetDir and dSetName in exp_config.ini before running train_mcrbm.py")
    print("Or use: python train_mcrbm.py --config-dir ./configuration_files")
    
    # Example training call (uncomment to run):
    # os.chdir('./scripts')
    # model = mcRBM(
    #     refDir='../configuration_files',
    #     expConfigFilename='exp_config.ini',
    #     modelConfigFilename='model_config.ini'
    # )
    # model.train()
    
    # Step 3: Inference (after training)
    print("\n[Step 3] Inference and Analysis...")
    print("After training completes, run:")
    print("  python infer_mcrbm.py \\")
    print("    --model-dir ./experiments/example/weights \\")
    print("    --exp-dir ./experiments/example \\")
    print("    --analyze")
    
    # Example inference call (uncomment after training):
    # os.chdir('./scripts')
    # inference = mcRBMInference(
    #     modelDir='../experiments/example/weights',
    #     expDir='../experiments/example',
    #     modelFile='ws_final.mat'
    # )
    # inference.loadData('./', 'states.mat')
    # states = inference.computeStates()
    # analysis = inference.analyzeStates()
    
    print("\n" + "=" * 60)
    print("Workflow Summary:")
    print("1. Data prepared in: mcrbm_processed.npz")
    print("2. Configuration in: ./configuration_files/")
    print("3. Run training: python train_mcrbm.py")
    print("4. Run inference: python infer_mcrbm.py --analyze")
    print("=" * 60)


if __name__ == '__main__':
    example_full_workflow()
