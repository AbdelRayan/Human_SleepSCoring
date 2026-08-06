#!/usr/bin/env python3
"""
Split dataset into train and test sets with balance checking.

Loads a .npz file, splits into train/test, and verifies that both sets have
similar distributions of sleep stage labels (if present).

Saves separate .npz files for train and test sets.
"""

import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def check_balance(labels, set_name="Dataset"):
    """
    Check label distribution and return statistics.
    
    Args:
        labels: Array of label values
        set_name: Name of dataset (for printing)
    
    Returns:
        Dictionary with balance statistics
    """
    if labels is None or len(labels) == 0:
        return None
    
    unique_labels = np.unique(labels[~np.isnan(labels)])
    if len(unique_labels) == 0:
        return None
    
    counts = Counter(labels[~np.isnan(labels)])
    total = sum(counts.values())
    
    print(f"\n{set_name} Balance:")
    print(f"  Total samples: {total}")
    print(f"  Unique labels: {sorted(unique_labels)}")
    
    for label in sorted(unique_labels):
        count = counts[label]
        pct = 100.0 * count / total
        print(f"    Label {int(label)}: {count:6d} ({pct:5.1f}%)")
    
    return {
        'total': total,
        'counts': dict(counts),
        'percentages': {k: 100.0 * v / total for k, v in counts.items()}
    }


def check_balance_consistency(train_stats, test_stats, tolerance=5.0):
    """
    Check if train and test sets have consistent label distributions.
    
    Args:
        train_stats: Balance statistics for train set
        test_stats: Balance statistics for test set
        tolerance: Maximum allowed percentage point difference (default: 5%)
    
    Returns:
        Tuple: (is_balanced, imbalance_report)
    """
    if train_stats is None or test_stats is None:
        return True, "No labels to check"
    
    imbalances = {}
    max_diff = 0.0
    
    for label in train_stats['percentages']:
        train_pct = train_stats['percentages'].get(label, 0.0)
        test_pct = test_stats['percentages'].get(label, 0.0)
        diff = abs(train_pct - test_pct)
        imbalances[label] = diff
        max_diff = max(max_diff, diff)
    
    for label in test_stats['percentages']:
        if label not in train_stats['percentages']:
            train_pct = 0.0
            test_pct = test_stats['percentages'][label]
            diff = abs(train_pct - test_pct)
            imbalances[label] = diff
            max_diff = max(max_diff, diff)
    
    is_balanced = max_diff <= tolerance
    
    report = f"Max percentage point difference: {max_diff:.1f}%\n"
    report += f"Tolerance: {tolerance:.1f}%\n"
    report += f"Status: {'BALANCED' if is_balanced else 'IMBALANCED (warning)'}\n"
    report += "\nPer-label differences (percentage points):\n"
    for label in sorted(imbalances.keys()):
        report += f"  Label {int(label)}: {imbalances[label]:.1f}%\n"
    
    return is_balanced, report


def train_test_split_stratified(X, labels=None, test_ratio=0.2, random_seed=42):
    """
    Split data into train/test sets with optional stratification by labels.
    
    Args:
        X: Data array (N_samples × N_features)
        labels: Optional label array for stratification
        test_ratio: Fraction of data for test set (0-1)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple: (X_train, X_test, labels_train, labels_test, train_idx, test_idx)
    """
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    
    if labels is None or np.all(np.isnan(labels)):
        # No labels: simple random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_idx = int(n_samples * (1 - test_ratio))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
    else:
        # Stratified split by label
        unique_labels = np.unique(labels[~np.isnan(labels)])
        train_idx = []
        test_idx = []
        
        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]
            np.random.shuffle(indices)
            split_idx = int(len(indices) * (1 - test_ratio))
            train_idx.extend(indices[:split_idx])
            test_idx.extend(indices[split_idx:])
        
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    labels_train = labels[train_idx] if labels is not None else None
    labels_test = labels[test_idx] if labels is not None else None
    
    return X_train, X_test, labels_train, labels_test, train_idx, test_idx


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train and test sets with balance checking.'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input .npz file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input file directory)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Fraction of data for test set (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=5.0,
        help='Max percentage point difference for balance check (default: 5.0)'
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
    if labels is not None:
        labels = np.array(labels, dtype=np.float32).flatten()
    epoch_times = data.get('epochTime', None)
    
    print(f"Data shape: {X.shape}")
    if labels is not None:
        print(f"Labels shape: {labels.shape}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform split
    print(f"\nPerforming stratified train/test split...")
    print(f"Test ratio: {args.test_ratio:.1%}")
    
    X_train, X_test, labels_train, labels_test, train_idx, test_idx = train_test_split_stratified(
        X, labels=labels, test_ratio=args.test_ratio, random_seed=args.seed
    )
    
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Check balance
    train_stats = check_balance(labels_train, set_name="Train Set")
    test_stats = check_balance(labels_test, set_name="Test Set")
    
    is_balanced, balance_report = check_balance_consistency(
        train_stats, test_stats, tolerance=args.tolerance
    )
    
    print("\n" + "="*60)
    print("BALANCE CHECK REPORT")
    print("="*60)
    print(balance_report)
    print("="*60)
    
    # Save train and test sets
    train_path = output_dir / input_path.stem + "_train.npz"
    test_path = output_dir / input_path.stem + "_test.npz"
    
    print(f"\nSaving train set: {train_path}")
    np.savez(
        train_path,
        d=X_train,
        epochsLinked=labels_train if labels_train is not None else np.array([]),
        epochTime=epoch_times[train_idx] if epoch_times is not None else np.array([])
    )
    
    print(f"Saving test set: {test_path}")
    np.savez(
        test_path,
        d=X_test,
        epochsLinked=labels_test if labels_test is not None else np.array([]),
        epochTime=epoch_times[test_idx] if epoch_times is not None else np.array([])
    )
    
    # Save summary report
    report_path = output_dir / (input_path.stem + "_split_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Train/Test Split Report\n")
        f.write(f"Source: {input_path}\n")
        f.write(f"Test ratio: {args.test_ratio:.1%}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Train set: {train_path.name}\n")
        f.write(f"  Samples: {X_train.shape[0]}\n")
        f.write(f"  Features: {X_train.shape[1]}\n")
        f.write(f"\nTest set: {test_path.name}\n")
        f.write(f"  Samples: {X_test.shape[0]}\n")
        f.write(f"  Features: {X_test.shape[1]}\n")
        f.write(f"\n{balance_report}\n")
    
    print(f"Saving report: {report_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
