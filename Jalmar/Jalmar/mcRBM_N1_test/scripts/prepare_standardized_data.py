#!/usr/bin/env python3
"""
Prepare standardized training/test data for mcRBM to prevent model collapse.

Standardization: (X - mean) / std, computed from training set and applied to both.

Usage:
    python prepare_standardized_data.py \
        --train-npz <path> \
        --test-npz <path> \
        --output-dir <dir>
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Standardize NPZ data for mcRBM training')
    parser.add_argument('--train-npz', type=str, required=True, help='Path to training NPZ')
    parser.add_argument('--test-npz', type=str, required=True, help='Path to test NPZ')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading train: {args.train_npz}")
    train_data = np.load(args.train_npz, allow_pickle=True)
    X_train = train_data['d'].astype(np.float64)
    
    print(f"Loading test: {args.test_npz}")
    test_data = np.load(args.test_npz, allow_pickle=True)
    X_test = test_data['d'].astype(np.float64)
    
    print(f"\nBefore standardization:")
    print(f"  Train: shape={X_train.shape}, mean={X_train.mean(axis=0)[:3]}, std={X_train.std(axis=0)[:3]}")
    print(f"  Test:  shape={X_test.shape}, mean={X_test.mean(axis=0)[:3]}, std={X_test.std(axis=0)[:3]}")
    
    # Compute standardization from train set
    train_mean = X_train.mean(axis=0, keepdims=True)
    train_std = X_train.std(axis=0, keepdims=True)
    
    # Prevent division by zero
    train_std[train_std < 1e-8] = 1.0
    
    # Standardize both sets
    X_train_std = (X_train - train_mean) / train_std
    X_test_std = (X_test - train_mean) / train_std
    
    print(f"\nAfter standardization:")
    print(f"  Train: mean={X_train_std.mean(axis=0)[:3]}, std={X_train_std.std(axis=0)[:3]}")
    print(f"  Test:  mean={X_test_std.mean(axis=0)[:3]}, std={X_test_std.std(axis=0)[:3]}")
    
    # Save standardized data
    train_out = output_dir / 'train_standardized.npz'
    test_out = output_dir / 'test_standardized.npz'
    stats_out = output_dir / 'standardization_stats.npz'
    
    np.savez(train_out, d=X_train_std.astype(np.float32))
    np.savez(test_out, d=X_test_std.astype(np.float32))
    np.savez(stats_out, mean=train_mean.flatten(), std=train_std.flatten())
    
    print(f"\nSaved:")
    print(f"  {train_out}")
    print(f"  {test_out}")
    print(f"  {stats_out}")
    print(f"\nStandardization complete! Use train_standardized.npz for mcRBM training.")


if __name__ == '__main__':
    main()
