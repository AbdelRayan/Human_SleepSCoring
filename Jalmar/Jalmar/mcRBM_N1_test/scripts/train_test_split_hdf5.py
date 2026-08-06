#!/usr/bin/env python3
"""
Split HDF5 or NPZ dataset into train and test sets with balance checking.

Loads a .h5/.hdf5 or .npz file, splits into train/test, and verifies that both 
sets have similar distributions of sleep stage labels (if present).

Saves separate .npz files for train and test sets.
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import Counter


# ============================================================================
# CONFIGURATION SECTION - edit these paths before running
# ============================================================================
# INPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\relaxed\sleep_features_N1_selection_relaxed_artifact_filtered.h5"
# OUTPUT_DIR = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\relaxed\N1_selection_relaxed_artifact_filtered"

# INPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\relaxed\sleep_features_full_selection_relaxed_artifact_filtered.h5"
# OUTPUT_DIR = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\relaxed\full_selection_relaxed_artifact_filtered"

INPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\15s_processing_test\raw_full\full_sleep_features_raw_artifact_filtered.h5"

OUTPUT_DIR = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\15s_processing_test\raw_full\full_artifact_filtered_split"

TEST_RATIO = 0.2
RANDOM_SEED = 42
BALANCE_TOLERANCE = 5.0  # Max percentage point difference allowed
# ============================================================================


def load_data_from_file(file_path):
    """
    Load data from HDF5 or NPZ file.
    
    Args:
        file_path: Path to input file (.h5, .hdf5, or .npz)
    
    Returns:
        Tuple: (X, labels, epoch_times)
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() in ['.h5', '.hdf5']:
        # Load from HDF5
        print(f"Loading HDF5 file: {file_path}")
        X_list = []
        labels_list = []
        epoch_times_list = []
        
        with h5py.File(file_path, 'r') as f:
            # Iterate over subject groups
            for subject_id in sorted(f.keys()):
                subject_group = f[subject_id]
                
                if 'features' in subject_group:
                    features = subject_group['features'][:]
                    X_list.append(features)
                    
                    # Get labels if available
                    if 'scores' in subject_group:
                        labels = np.asarray(subject_group['scores'][:], dtype=np.float32).flatten()
                        labels_list.append(labels)
                    elif 'labels' in subject_group.attrs:
                        label_val = subject_group.attrs['labels']
                        labels = np.full(features.shape[0], label_val, dtype=np.float32)
                        labels_list.append(labels)
                    elif 'label' in subject_group:
                        labels = np.asarray(subject_group['label'][:], dtype=np.float32).flatten()
                        labels_list.append(labels)
                    
                    # Get epoch times if available
                    if 'epochTime' in subject_group:
                        epoch_times = subject_group['epochTime'][:]
                        epoch_times_list.append(epoch_times)
                
                print(f"  {subject_id}: {features.shape}")
        
        X = np.vstack(X_list).astype(np.float32)
        labels = np.hstack(labels_list) if labels_list else None
        epoch_times = np.hstack(epoch_times_list) if epoch_times_list else None
        
    elif file_path.suffix.lower() == '.npz':
        # Load from NPZ
        print(f"Loading NPZ file: {file_path}")
        data = np.load(file_path)
        X = data['d'].astype(np.float32)
        labels = data.get('epochsLinked', None)
        if labels is not None:
            labels = np.array(labels, dtype=np.float32).flatten()
        epoch_times = data.get('epochTime', None)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Data shape: {X.shape}")
    if labels is not None:
        print(f"Labels shape: {labels.shape}")
    if epoch_times is not None:
        print(f"Epoch times shape: {epoch_times.shape}")
    
    return X, labels, epoch_times


def load_hdf5_subject_rows(file_path):
    """Load subject-wise rows from an HDF5 file while preserving source metadata."""
    subject_rows = []

    with h5py.File(file_path, "r") as f:
        for subject_id in sorted(f.keys()):
            subject_group = f[subject_id]
            if "features" not in subject_group:
                continue

            features = np.asarray(subject_group["features"][:], dtype=np.float32)
            row_index = np.arange(features.shape[0], dtype=np.int32)

            if "scores" in subject_group:
                labels = np.asarray(subject_group["scores"][:], dtype=np.float32).flatten()
            elif "label" in subject_group:
                labels = np.asarray(subject_group["label"][:], dtype=np.float32).flatten()
            elif "labels" in subject_group.attrs:
                labels = np.full(features.shape[0], subject_group.attrs["labels"], dtype=np.float32)
            else:
                labels = None

            epoch_times = None
            if "epochTime" in subject_group:
                epoch_times = np.asarray(subject_group["epochTime"][:], dtype=np.float32)

            subject_rows.append(
                {
                    "subject_id": str(subject_id),
                    "features": features,
                    "labels": labels,
                    "epoch_times": epoch_times,
                    "row_index": row_index,
                    "attrs": dict(subject_group.attrs),
                }
            )

    return subject_rows


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


def _split_indices_for_subject(n_samples, labels, test_ratio, random_seed):
    """Split one subject's rows into train/test indices using local label balance."""
    rng = np.random.default_rng(random_seed)

    if labels is None or np.all(np.isnan(labels)):
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        split_idx = int(n_samples * (1 - test_ratio))
        return indices[:split_idx], indices[split_idx:]

    unique_labels = np.unique(labels[~np.isnan(labels)])
    train_idx = []
    test_idx = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        split_idx = int(len(label_indices) * (1 - test_ratio))
        train_idx.extend(label_indices[:split_idx])
        test_idx.extend(label_indices[split_idx:])

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def split_hdf5_subject_rows(subject_rows, test_ratio=0.2, random_seed=42):
    """Split HDF5 subject rows independently so each subject keeps similar label proportions."""
    train_rows = []
    test_rows = []

    for subject_offset, row in enumerate(subject_rows):
        labels = row["labels"]
        n_samples = row["features"].shape[0]
        local_seed = random_seed + subject_offset
        train_local, test_local = _split_indices_for_subject(n_samples, labels, test_ratio, local_seed)

        def _make_row(local_indices):
            local_indices = np.asarray(local_indices, dtype=int)
            if local_indices.size == 0:
                return None
            local_indices = np.sort(local_indices)
            return {
                "subject_id": row["subject_id"],
                "features": row["features"][local_indices],
                "labels": row["labels"][local_indices] if row["labels"] is not None else None,
                "epoch_times": row["epoch_times"][local_indices] if row["epoch_times"] is not None else None,
                "row_index": row["row_index"][local_indices],
                "attrs": row["attrs"],
            }

        train_row = _make_row(train_local)
        test_row = _make_row(test_local)
        if train_row is not None:
            train_rows.append(train_row)
        if test_row is not None:
            test_rows.append(test_row)

    return train_rows, test_rows


def flatten_split_rows(split_rows):
    """Flatten per-subject split rows into arrays for npz export and balance checks."""
    if not split_rows:
        return (
            np.empty((0, 0), dtype=np.float32),
            None,
            None,
            np.array([], dtype="U"),
            np.array([], dtype=np.int32),
        )

    split_rows = sorted(split_rows, key=lambda row: row["subject_id"])

    features = np.vstack([row["features"] for row in split_rows]).astype(np.float32)

    labels_parts = [row["labels"] for row in split_rows if row["labels"] is not None]
    labels = np.hstack(labels_parts).astype(np.float32) if labels_parts else None

    epoch_parts = [row["epoch_times"] for row in split_rows if row["epoch_times"] is not None]
    epoch_times = np.hstack(epoch_parts).astype(np.float32) if epoch_parts else None

    subject_ids = np.asarray(
        [row["subject_id"] for row in split_rows for _ in range(row["features"].shape[0])],
        dtype=str,
    )
    row_indices = np.hstack([row["row_index"] for row in split_rows]).astype(np.int32)

    return features, labels, epoch_times, subject_ids, row_indices


def save_hdf5_split(output_path, split_rows, source_path, split_name, test_ratio, seed, balance_report):
    """Save the split using the original HDF5 subject-group layout."""
    with h5py.File(output_path, "w") as f:
        f.attrs["source_file"] = str(source_path)
        f.attrs["split_name"] = split_name
        f.attrs["test_ratio"] = float(test_ratio)
        f.attrs["random_seed"] = int(seed)
        f.attrs["balance_report"] = balance_report

        for row in split_rows:
            group = f.create_group(row["subject_id"])

            for attr_key, attr_value in row["attrs"].items():
                group.attrs[attr_key] = attr_value

            group.create_dataset("features", data=row["features"], compression="gzip")
            group.create_dataset("row_index", data=row["row_index"], compression="gzip")

            if row["labels"] is not None:
                group.create_dataset("scores", data=row["labels"], compression="gzip")

            if row["epoch_times"] is not None:
                group.create_dataset("epochTime", data=row["epoch_times"], compression="gzip")


def _build_split_rows(subject_rows, sample_indices):
    """Create per-subject HDF5 rows for one split from global sample indices."""
    index_set = set(int(idx) for idx in sample_indices)
    split_rows = []
    cursor = 0

    for row in subject_rows:
        features = row["features"]
        labels = row["labels"]
        epoch_times = row["epoch_times"]
        n_samples = features.shape[0]
        local_indices = [local_idx for local_idx in range(n_samples) if (cursor + local_idx) in index_set]

        if not local_indices:
            cursor += n_samples
            continue

        local_indices = np.asarray(local_indices, dtype=int)
        split_row = {
            "subject_id": row["subject_id"],
            "features": features[local_indices],
            "labels": labels[local_indices] if labels is not None else None,
            "epoch_times": epoch_times[local_indices] if epoch_times is not None else None,
            "row_index": row["row_index"][local_indices],
            "attrs": row["attrs"],
        }
        split_rows.append(split_row)
        cursor += n_samples

    return split_rows


def _build_split_metadata(subject_rows, sample_indices):
    """Create flat subject_id and row_index arrays for one split."""
    index_set = set(int(idx) for idx in sample_indices)
    subject_ids = []
    row_indices = []
    cursor = 0

    for row in subject_rows:
        n_samples = row["features"].shape[0]
        for local_idx in range(n_samples):
            if (cursor + local_idx) in index_set:
                subject_ids.append(row["subject_id"])
                row_indices.append(int(row["row_index"][local_idx]))
        cursor += n_samples

        return np.asarray(subject_ids, dtype=str), np.asarray(row_indices, dtype=np.int32)


def _build_npz_payload(features, labels=None, epoch_times=None, subject_ids=None, row_indices=None):
    """Build a payload for np.savez with optional metadata fields."""
    payload = {"d": features}
    if labels is not None:
        payload["epochsLinked"] = labels
    if epoch_times is not None:
        payload["epochTime"] = epoch_times
    if subject_ids is not None:
        payload["subject_id"] = subject_ids
    if row_indices is not None:
        payload["row_index"] = row_indices
    return payload


def main():
    parser = argparse.ArgumentParser(
        description='Split HDF5/NPZ dataset into train and test sets with balance checking.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=INPUT_HDF5_PATH,
        help=f'Input .h5/.hdf5/.npz file path (default: {INPUT_HDF5_PATH})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=TEST_RATIO,
        help=f'Fraction of data for test set (default: {TEST_RATIO})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=BALANCE_TOLERANCE,
        help=f'Max percentage point difference for balance check (default: {BALANCE_TOLERANCE})'
    )
    
    args = parser.parse_args()
    
    # Load input file
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading: {input_path}")
    is_hdf5_input = input_path.suffix.lower() in {".h5", ".hdf5"}

    hdf5_subject_rows = load_hdf5_subject_rows(input_path) if is_hdf5_input else None
    X, labels, epoch_times = load_data_from_file(input_path)
    
    # Determine output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform split
    print(f"\nPerforming stratified train/test split...")
    print(f"Test ratio: {args.test_ratio:.1%}")

    train_rows = None
    test_rows = None

    if is_hdf5_input and hdf5_subject_rows is not None:
        train_rows, test_rows = split_hdf5_subject_rows(
            hdf5_subject_rows,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )
        X_train, labels_train, epoch_times_train, train_subject_ids, train_row_indices = flatten_split_rows(train_rows)
        X_test, labels_test, epoch_times_test, test_subject_ids, test_row_indices = flatten_split_rows(test_rows)
        train_idx = None
        test_idx = None
    else:
        X_train, X_test, labels_train, labels_test, train_idx, test_idx = train_test_split_stratified(
            X, labels=labels, test_ratio=args.test_ratio, random_seed=args.seed
        )
        epoch_times_train = epoch_times[train_idx] if epoch_times is not None else None
        epoch_times_test = epoch_times[test_idx] if epoch_times is not None else None
        train_subject_ids = None
        train_row_indices = None
        test_subject_ids = None
        test_row_indices = None
    
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
    
    # Save train and test sets as NPZ
    train_path = output_dir / (input_path.stem + "_train.npz")
    test_path = output_dir / (input_path.stem + "_test.npz")
    
    print(f"\nSaving train set: {train_path}")
    np.savez(
        train_path,
        **_build_npz_payload(
            X_train,
            labels=labels_train,
            epoch_times=epoch_times_train,
            subject_ids=train_subject_ids,
            row_indices=train_row_indices,
        ),
    )

    if is_hdf5_input and train_rows is not None:
        train_h5_path = output_dir / (input_path.stem + "_train.h5")
        print(f"Saving train HDF5 set: {train_h5_path}")
        save_hdf5_split(
            train_h5_path,
            train_rows,
            source_path=input_path,
            split_name="train",
            test_ratio=args.test_ratio,
            seed=args.seed,
            balance_report=balance_report,
        )
    
    print(f"Saving test set: {test_path}")
    np.savez(
        test_path,
        **_build_npz_payload(
            X_test,
            labels=labels_test,
            epoch_times=epoch_times_test,
            subject_ids=test_subject_ids,
            row_indices=test_row_indices,
        ),
    )

    if is_hdf5_input and test_rows is not None:
        test_h5_path = output_dir / (input_path.stem + "_test.h5")
        print(f"Saving test HDF5 set: {test_h5_path}")
        save_hdf5_split(
            test_h5_path,
            test_rows,
            source_path=input_path,
            split_name="test",
            test_ratio=args.test_ratio,
            seed=args.seed,
            balance_report=balance_report,
        )
    
    # Save stripped (label-free) versions for unsupervised training
    train_stripped_path = output_dir / (input_path.stem + "_train_stripped.npz")
    test_stripped_path = output_dir / (input_path.stem + "_test_stripped.npz")
    
    print(f"\nSaving stripped train set (no labels): {train_stripped_path}")
    train_stripped_epoch_times = epoch_times[train_idx] if epoch_times is not None else None
    if is_hdf5_input and train_rows is not None:
        train_stripped_epoch_times = epoch_times_train
    np.savez(
        train_stripped_path,
        **_build_npz_payload(
            X_train,
            labels=None,
            epoch_times=train_stripped_epoch_times,
        ),
    )
    
    print(f"Saving stripped test set (no labels): {test_stripped_path}")
    test_stripped_epoch_times = epoch_times[test_idx] if epoch_times is not None else None
    if is_hdf5_input and test_rows is not None:
        test_stripped_epoch_times = epoch_times_test
    np.savez(
        test_stripped_path,
        **_build_npz_payload(
            X_test,
            labels=None,
            epoch_times=test_stripped_epoch_times,
        ),
    )
    
    # Save summary report
    report_path = output_dir / (input_path.stem + "_split_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Train/Test Split Report\n")
        f.write(f"Source: {input_path}\n")
        f.write(f"Test ratio: {args.test_ratio:.1%}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Train set (with labels): {train_path.name}\n")
        f.write(f"  Samples: {X_train.shape[0]}\n")
        f.write(f"  Features: {X_train.shape[1]}\n")
        f.write(f"\nTrain set (stripped - no labels): {train_stripped_path.name}\n")
        f.write(f"  Samples: {X_train.shape[0]}\n")
        f.write(f"  Features: {X_train.shape[1]}\n")
        f.write(f"\nTest set (with labels): {test_path.name}\n")
        f.write(f"  Samples: {X_test.shape[0]}\n")
        f.write(f"  Features: {X_test.shape[1]}\n")
        f.write(f"\nTest set (stripped - no labels): {test_stripped_path.name}\n")
        f.write(f"  Samples: {X_test.shape[0]}\n")
        f.write(f"  Features: {X_test.shape[1]}\n")
        f.write(f"\n{balance_report}\n")
    
    print(f"Saving report: {report_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
