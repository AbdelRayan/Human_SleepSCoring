#!/usr/bin/env python3
"""
Strip sleep stage labels from dataset.

Removes or clears the 'epochsLinked' field from a .npz file, creating a clean
unsupervised dataset without ground truth labels.

This is useful if you want to ensure the mcRBM learns purely from the data
without any label information leakage.
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Strip sleep stage labels from a .npz dataset.'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input .npz file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output .npz file path (default: input_stripped.npz)'
    )
    parser.add_argument(
        '--keep-epoch-time',
        action='store_true',
        default=True,
        help='Keep epochTime field (default: True)'
    )
    parser.add_argument(
        '--no-keep-epoch-time',
        dest='keep_epoch_time',
        action='store_false',
        help='Remove epochTime field as well'
    )
    
    args = parser.parse_args()
    
    # Load input file
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading: {input_path}")
    data = np.load(input_path)
    
    X = data['d'].astype(np.float32)
    labels = data.get('epochsLinked', None)
    epoch_times = data.get('epochTime', None)
    
    print(f"Data shape: {X.shape}")
    if labels is not None:
        print(f"Original labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels[~np.isnan(labels)])}")
    else:
        print("No labels found in dataset")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + "_stripped.npz")
    
    # Save without labels
    print(f"\nStripping labels...")
    save_dict = {'d': X}
    
    if args.keep_epoch_time and epoch_times is not None:
        save_dict['epochTime'] = epoch_times
        print(f"Keeping epochTime field: shape {epoch_times.shape}")
    else:
        print("Removing epochTime field")
    
    np.savez(output_path, **save_dict)
    
    print(f"\nSaving to: {output_path}")
    
    # Verify output
    verify_data = np.load(output_path)
    print(f"\nOutput file contents:")
    print(f"  Keys: {list(verify_data.keys())}")
    for key in verify_data.keys():
        arr = verify_data[key]
        print(f"  {key}: shape {arr.shape}, dtype {arr.dtype}")
    
    # Summary
    print(f"\n" + "="*60)
    print("LABEL STRIPPING SUMMARY")
    print("="*60)
    print(f"Input file:  {input_path.name}")
    print(f"Output file: {output_path.name}")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Labels removed: {'epochsLinked' in data}")
    print(f"EpochTime kept: {args.keep_epoch_time and 'epochTime' in data}")
    print("="*60)
    print("Done!")


if __name__ == '__main__':
    main()
