#!/usr/bin/env python3
"""
Filter movement/artifact epochs from an input HDF5 sleep dataset.

The script preserves the original subject-group layout and writes a new HDF5
file next to the input file using the same base name plus the
"_artifact_filtered" suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


ARTIFACT_STAGE_LABELS = {5}


def _resolve_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_artifact_filtered{input_path.suffix}")


def _copy_group_attributes(source_group: h5py.Group, target_group: h5py.Group) -> None:
    for key, value in source_group.attrs.items():
        target_group.attrs[key] = value


def _filter_subject_group(source_group: h5py.Group) -> tuple[np.ndarray, np.ndarray | None, dict[str, object], int]:
    if "scores" not in source_group:
        raise ValueError("subject group is missing required 'scores' dataset")

    if "features" not in source_group:
        raise ValueError("subject group is missing required 'features' dataset")

    features = np.asarray(source_group["features"][:])
    scores = np.asarray(source_group["scores"][:])
    epoch_time_values = np.asarray(source_group["epochTime"][:]) if "epochTime" in source_group else None

    lengths = [features.shape[0], scores.shape[0]]
    if epoch_time_values is not None:
        lengths.append(epoch_time_values.shape[0])

    limit = min(lengths)
    features = features[:limit]
    scores = scores[:limit]
    if epoch_time_values is not None:
        epoch_time_values = epoch_time_values[:limit]

    keep_mask = ~np.isin(scores.astype(int), list(ARTIFACT_STAGE_LABELS))
    filtered_features = features[keep_mask]
    filtered_scores = scores[keep_mask]

    epoch_time = None
    if epoch_time_values is not None:
        epoch_time = epoch_time_values[keep_mask]

    attrs: dict[str, object] = {}
    for key, value in source_group.attrs.items():
        attrs[key] = value

    return filtered_features, filtered_scores, {"epochTime": epoch_time, "attrs": attrs}, int(np.size(scores) - np.count_nonzero(keep_mask))


def filter_artifact_epochs(input_hdf5_path: str, output_hdf5_path: str | None = None) -> Path:
    input_path = Path(input_hdf5_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_path}")

    output_path = Path(output_hdf5_path) if output_hdf5_path else _resolve_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    total_kept = 0

    with h5py.File(input_path, "r") as source_file, h5py.File(output_path, "w") as target_file:
        for key, value in source_file.attrs.items():
            target_file.attrs[key] = value
        target_file.attrs["source_file"] = str(input_path)
        target_file.attrs["filter_type"] = "artifact_epochs_removed"
        target_file.attrs["artifact_stage_labels"] = np.asarray(sorted(ARTIFACT_STAGE_LABELS), dtype=np.int32)

        for subject_name in source_file.keys():
            source_group = source_file[subject_name]
            if "features" not in source_group or "scores" not in source_group:
                print(f"Skipping {subject_name}: missing features or scores")
                continue

            filtered_features, filtered_scores, metadata, removed_count = _filter_subject_group(source_group)
            epoch_time = metadata["epochTime"]
            attrs = metadata["attrs"]

            target_group = target_file.create_group(subject_name)
            _copy_group_attributes(source_group, target_group)
            target_group.attrs["source_subject"] = subject_name
            target_group.attrs["artifact_filtered"] = True

            target_group.create_dataset("features", data=filtered_features, dtype=filtered_features.dtype)
            target_group.create_dataset("scores", data=filtered_scores, dtype=filtered_scores.dtype)
            if epoch_time is not None:
                target_group.create_dataset("epochTime", data=epoch_time, dtype=epoch_time.dtype)

            for attr_key, attr_value in attrs.items():
                if attr_key == "description_features":
                    continue
                target_group.attrs[attr_key] = attr_value

            total_removed += removed_count
            total_kept += filtered_features.shape[0]

            print(
                f"Processed {subject_name}: removed {removed_count} artifact epochs, kept {filtered_features.shape[0]}"
            )

    print(f"Saved artifact-filtered HDF5 to: {output_path}")
    print(f"Total kept: {total_kept}")
    print(f"Total removed: {total_removed}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove movement/artifact epochs from an HDF5 sleep dataset.")
    parser.add_argument("--input", required=True, help="Input HDF5 file path")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output HDF5 file path. Defaults to input name + _artifact_filtered.h5 in the same folder.",
    )
    args = parser.parse_args()

    filter_artifact_epochs(args.input, args.output)


if __name__ == "__main__":
    main()