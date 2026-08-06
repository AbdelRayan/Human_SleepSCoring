#!/usr/bin/env python3
"""
Export NPZ or HDF5 sleep-feature datasets to human-readable CSV files.

Supported inputs:
  - .npz files with a feature matrix stored under `d`
  - .h5 / .hdf5 files with subject groups containing `features` and optional
    `scores`, `label`, `epochTime`, or feature-name metadata

Behavior:
  - NPZ inputs are exported as a single CSV file.
  - HDF5 inputs are exported one CSV per subject group by default.
  - Use `--combined` to also write a single concatenated CSV.

Examples:
  python export_dataset_to_csv.py --input my_features.npz --output-dir csv_out
  python export_dataset_to_csv.py --input my_features.h5 --output-dir csv_out --combined
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


def _parse_feature_names(raw_names: object | None, n_features: int) -> list[str]:
    if raw_names is None:
        return [f"feature_{idx + 1}" for idx in range(n_features)]

    if isinstance(raw_names, bytes):
        raw_names = raw_names.decode("utf-8", errors="ignore")

    if isinstance(raw_names, np.ndarray):
        if raw_names.ndim == 0:
            raw_names = raw_names.item()
        else:
            raw_names = ",".join(str(item) for item in raw_names.tolist())

    names = [name.strip() for name in str(raw_names).split(",") if name.strip()]
    if len(names) != n_features:
        return [f"feature_{idx + 1}" for idx in range(n_features)]
    return names


def _subject_feature_names(group: h5py.Group, n_features: int) -> list[str]:
    for key in ("description_features", "feature_names", "features_names"):
        if key in group.attrs:
            return _parse_feature_names(group.attrs.get(key), n_features)
    for key in ("description_features", "feature_names", "features_names"):
        if key in group:
            value = group[key][()]
            return _parse_feature_names(value, n_features)
    return [f"feature_{idx + 1}" for idx in range(n_features)]


def _normalize_optional_array(values: object | None, length: int) -> np.ndarray | None:
    if values is None:
        return None

    array = np.asarray(values)
    if array.size == 0:
        return None

    array = array.reshape(-1)
    if array.shape[0] != length:
        return None
    return array


def _build_dataframe(
    features: np.ndarray,
    feature_names: list[str],
    subject_id: str | None = None,
    subject_ids: np.ndarray | None = None,
    row_index: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    epoch_times: np.ndarray | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(features, columns=feature_names)
    if subject_ids is not None:
        df.insert(0, "subject_id", subject_ids)
    elif subject_id is not None:
        df.insert(0, "subject_id", subject_id)

    if row_index is None:
        row_index = np.arange(len(df), dtype=int)
    df.insert(0 if subject_ids is None and subject_id is None else 1, "row_index", row_index)

    if labels is not None:
        df["label"] = labels
    if epoch_times is not None:
        df["epochTime"] = epoch_times
    return df


def export_npz(input_path: Path, output_dir: Path) -> Path:
    data = np.load(input_path, allow_pickle=True)
    if "d" not in data.files:
        raise ValueError(f"NPZ file does not contain required key 'd': {input_path}")

    features = np.asarray(data["d"])
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix in {input_path}, got shape {features.shape}")

    labels = _normalize_optional_array(data["epochsLinked"] if "epochsLinked" in data.files else None, features.shape[0])
    epoch_times = _normalize_optional_array(data["epochTime"] if "epochTime" in data.files else None, features.shape[0])
    subject_ids = None
    if "subject_id" in data.files:
        subject_ids = np.asarray(data["subject_id"]).reshape(-1)
        if subject_ids.shape[0] != features.shape[0]:
            subject_ids = None
    row_index = _normalize_optional_array(data["row_index"] if "row_index" in data.files else None, features.shape[0])

    feature_names = _parse_feature_names(
        data["feature_names"] if "feature_names" in data.files else (
            data["description_features"] if "description_features" in data.files else None
        ),
        features.shape[1],
    )

    df = _build_dataframe(
        features,
        feature_names,
        subject_ids=subject_ids,
        row_index=row_index,
        labels=labels,
        epoch_times=epoch_times,
    )
    out_path = output_dir / f"{input_path.stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _iter_hdf5_subjects(input_path: Path) -> Iterable[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]]:
    with h5py.File(input_path, "r") as handle:
        for subject_id in sorted(handle.keys()):
            group = handle[subject_id]
            if "features" not in group:
                continue

            features = np.asarray(group["features"][:])
            if features.ndim != 2:
                raise ValueError(
                    f"Expected 2D features in group {subject_id} of {input_path}, got shape {features.shape}"
                )

            labels = None
            if "scores" in group:
                labels = np.asarray(group["scores"][:]).reshape(-1)
            elif "label" in group:
                labels = np.asarray(group["label"][:]).reshape(-1)
            elif "labels" in group.attrs:
                labels = np.full(features.shape[0], group.attrs["labels"])

            labels = _normalize_optional_array(labels, features.shape[0])

            epoch_times = None
            if "epochTime" in group:
                epoch_times = np.asarray(group["epochTime"][:]).reshape(-1)
            epoch_times = _normalize_optional_array(epoch_times, features.shape[0])

            row_index = None
            if "row_index" in group:
                row_index = np.asarray(group["row_index"][:]).reshape(-1)
            elif "epoch_index" in group:
                row_index = np.asarray(group["epoch_index"][:]).reshape(-1)
            row_index = _normalize_optional_array(row_index, features.shape[0])

            feature_names = _subject_feature_names(group, features.shape[1])

            yield subject_id, features, labels, epoch_times, row_index, feature_names


def export_hdf5(input_path: Path, output_dir: Path, combined: bool = False) -> list[Path]:
    written_files: list[Path] = []
    combined_frames: list[pd.DataFrame] = []

    for subject_id, features, labels, epoch_times, row_index, feature_names in _iter_hdf5_subjects(input_path):
        df = _build_dataframe(
            features,
            feature_names,
            subject_id=subject_id,
            row_index=row_index,
            labels=labels,
            epoch_times=epoch_times,
        )

        out_path = output_dir / f"{input_path.stem}_{subject_id}.csv"
        df.to_csv(out_path, index=False)
        written_files.append(out_path)

        if combined:
            combined_frames.append(df)

    if combined and combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_path = output_dir / f"{input_path.stem}_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        written_files.append(combined_path)

    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NPZ/HDF5 feature datasets to human-readable CSV files.")
    parser.add_argument("--input", required=True, help="Path to the input .npz, .h5, or .hdf5 file")
    parser.add_argument("--output-dir", required=True, help="Directory where CSV files will be written")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="For HDF5 inputs, also write one combined CSV in addition to per-subject CSVs",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        written = [export_npz(input_path, output_dir)]
    elif suffix in {".h5", ".hdf5"}:
        written = export_hdf5(input_path, output_dir, combined=args.combined)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    manifest_path = output_dir / f"{input_path.stem}_csv_export_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(f"Source file: {input_path}\n")
        handle.write(f"Output directory: {output_dir}\n")
        handle.write(f"Combined output requested: {args.combined}\n")
        handle.write("Written files:\n")
        for path in written:
            handle.write(f"  - {path.name}\n")

    print(f"Export complete. Wrote {len(written)} CSV file(s).")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()