"""
Extract selected Jalmar sleep features from an input HDF5 file into a new HDF5 file.

This script keeps the original subject-group layout and writes a new file that
contains only the requested feature columns.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


# ============================================================================
# CONFIGURATION SECTION - edit these paths before running
# ============================================================================
# INPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\sleep_features.h5"
# INPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\processing_test\sleep_features_preprocessed.h5"
# INPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\sleep_features_raw.h5"
INPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\15s_processing_test\sleep_features_raw.h5"

# OUTPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\sleep_features_N1_selection.h5"
# OUTPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\sleep_features_full_selection.h5"

OUTPUT_HDF5_PATH = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\15s_processing_test\raw_full\full_sleep_features_raw.h5"


# Columns to keep from the HDF5 feature tables.
# Matching is case-insensitive and ignores spaces/underscores/hyphens.
# TARGET_FEATURES = [
#     "Index_W",
#     "Index_1",
#     "Index_2",
#     "Index_3",
#     "Index_4",
#     "Delta",
#     "Theta",
#     "Aperiodic",
#     "DFA",
#     "MSE",
#     "EOG",
#     "Index_R_noEOG",
#     "Index_N_noEOG",
# ]
TARGET_FEATURES = [
    "Index_W",
    "Index_1",
    "Index_2",
    "Index_3",
    "Index_4",
    "Delta",
    "Theta",
    "Aperiodic",
    "DFA",
    "MSE",
    "EOG",
    "Index_R",
    "Index_N_noEOG",
]
# TARGET_FEATURES = ["Index_R_noEOG", "Index_N_noEOG", "Index_W", "EOG"]
# TARGET_FEATURES = ["Index_R", "Index_N_noEOG", "Index_W", "EOG"]

# If True, append a suffix to the output file rather than overwriting it.
OVERWRITE_OUTPUT = True
# ============================================================================


def _normalize_name(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")


def _parse_feature_names(description_attr: object, n_features: int) -> list[str]:
    if description_attr is None:
        return [f"Feature_{i + 1}" for i in range(n_features)]

    if isinstance(description_attr, bytes):
        description_attr = description_attr.decode("utf-8", errors="ignore")

    names = [part.strip() for part in str(description_attr).split(",") if part.strip()]
    if len(names) != n_features:
        return [f"Feature_{i + 1}" for i in range(n_features)]
    return names


def _find_feature_indices(feature_names: list[str], target_features: list[str]) -> tuple[list[int], list[str]]:
    normalized_to_index = {_normalize_name(name): idx for idx, name in enumerate(feature_names)}
    selected_indices: list[int] = []
    selected_names: list[str] = []

    for target_name in target_features:
        target_norm = _normalize_name(target_name)

        if target_norm in normalized_to_index:
            idx = normalized_to_index[target_norm]
            selected_indices.append(idx)
            selected_names.append(feature_names[idx])
            continue

        for idx, feature_name in enumerate(feature_names):
            if target_norm in _normalize_name(feature_name):
                selected_indices.append(idx)
                selected_names.append(feature_name)
                break

    return selected_indices, selected_names


def _copy_group_metadata(source_group: h5py.Group, target_group: h5py.Group) -> None:
    for key, value in source_group.attrs.items():
        if key == "description_features":
            continue
        target_group.attrs[key] = value


def extract_selected_features(input_hdf5_path: str, output_hdf5_path: str) -> None:
    input_path = Path(input_hdf5_path)
    output_path = Path(output_hdf5_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_path}")

    if output_path.exists():
        if OVERWRITE_OUTPUT:
            output_path.unlink()
        else:
            output_path = output_path.with_name(f"{output_path.stem}_selected{output_path.suffix}")
            if output_path.exists():
                output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as source_file, h5py.File(output_path, "w") as target_file:
        for subject_name in source_file.keys():
            # skip non-subject top-level groups created by preprocessing (stats, intermediate, etc.)
            if subject_name in ("stats", "intermediate"):
                continue

            source_group = source_file[subject_name]
            # ensure we have an HDF5 group with expected datasets
            if not isinstance(source_group, h5py.Group):
                continue

            if "features" not in source_group or "scores" not in source_group:
                print(f"Skipping {subject_name}: missing features or scores")
                continue

            features = np.asarray(source_group["features"][:], dtype=np.float32)
            scores = np.asarray(source_group["scores"][:])

            description = source_group.attrs.get("description_features")
            feature_names = _parse_feature_names(description, features.shape[1])

            selected_indices, selected_names = _find_feature_indices(feature_names, TARGET_FEATURES)

            if not selected_indices:
                print(f"Skipping {subject_name}: no matching features found")
                continue

            selected_indices = sorted(set(selected_indices))
            selected_features = features[:, selected_indices]
            selected_feature_names = [feature_names[idx] for idx in selected_indices]

            if len(scores) != len(selected_features):
                min_len = min(len(scores), len(selected_features))
                selected_features = selected_features[:min_len]
                scores = scores[:min_len]

            target_group = target_file.create_group(subject_name)
            target_group.create_dataset("features", data=selected_features, dtype="float32")
            target_group.create_dataset("scores", data=scores, dtype=scores.dtype)
            _copy_group_metadata(source_group, target_group)
            target_group.attrs["description_features"] = ", ".join(selected_feature_names)
            target_group.attrs["source_description_features"] = ", ".join(feature_names)
            target_group.attrs["selected_feature_targets"] = ", ".join(TARGET_FEATURES)
            target_group.attrs["selected_feature_indices"] = np.asarray(selected_indices, dtype=np.int32)

            print(
                f"Processed {subject_name}: {features.shape[1]} -> {selected_features.shape[1]} features "
                f"({', '.join(selected_feature_names)})"
            )

    print(f"Saved selected-feature HDF5 to: {output_path}")


if __name__ == "__main__":
    extract_selected_features(INPUT_HDF5_PATH, OUTPUT_HDF5_PATH)