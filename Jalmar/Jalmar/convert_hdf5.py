"""Convert a single HDF5 file to a text file.

Reads an .h5 or .hdf5 file and writes a .txt file with the same base name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT_FILE = Path(r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\sleep_features_N1_selection.h5")


def _format_attrs(attrs: h5py.AttributeManager) -> str:
	if len(attrs) == 0:
		return "    attrs: {}\n"

	lines = ["    attrs:\n"]
	for key in attrs.keys():
		value = attrs[key]
		lines.append(f"      - {key}: {value}\n")
	return "".join(lines)


def _write_dataset(dataset: h5py.Dataset, out_f, max_values: int) -> None:
	out_f.write(f"  type: dataset\n")
	out_f.write(f"  shape: {dataset.shape}\n")
	out_f.write(f"  dtype: {dataset.dtype}\n")
	out_f.write(_format_attrs(dataset.attrs))

	try:
		data = dataset[()]
	except Exception as exc:
		out_f.write(f"  data: <failed to read: {exc}>\n")
		return

	data_arr = np.asarray(data)

	if data_arr.size == 0:
		out_f.write("  data: []\n")
		return

	if data_arr.ndim == 0:
		out_f.write(f"  data: {data_arr.item()}\n")
		return

	flat = data_arr.reshape(-1)
	if max_values > 0 and flat.size > max_values:
		clipped = flat[:max_values]
		out_f.write(
			"  data: "
			+ np.array2string(clipped, separator=", ")
			+ f" ... (truncated, showing {max_values}/{flat.size} values)\n"
		)
	else:
		out_f.write("  data: " + np.array2string(data_arr, separator=", ") + "\n")


def _write_group(group: h5py.Group, out_f) -> None:
	out_f.write("  type: group\n")
	out_f.write(_format_attrs(group.attrs))


def convert_hdf5_to_txt(hdf5_path: Path, txt_path: Path, max_values: int) -> None:
	"""Convert one HDF5 file to a text representation."""
	with h5py.File(hdf5_path, "r") as h5f, txt_path.open("w", encoding="utf-8") as out_f:
		out_f.write(f"source_file: {hdf5_path}\n")
		out_f.write(f"output_file: {txt_path}\n")
		out_f.write("=" * 80 + "\n")

		def visitor(name: str, obj) -> None:
			out_f.write(f"path: /{name}\n")
			if isinstance(obj, h5py.Dataset):
				_write_dataset(obj, out_f, max_values=max_values)
			elif isinstance(obj, h5py.Group):
				_write_group(obj, out_f)
			out_f.write("-" * 80 + "\n")

		h5f.visititems(visitor)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert a single HDF5 file to a TXT file with its structure and data."
	)
	parser.add_argument(
		"--input_file",
		type=Path,
		default=DEFAULT_INPUT_FILE,
		help="Path to the input .h5/.hdf5 file.",
	)
	parser.add_argument(
		"--output_file",
		type=Path,
		default=None,
		help="Path to the output .txt file. If not specified, uses input filename with .txt extension.",
	)
	parser.add_argument(
		"--max_values",
		type=int,
		default=5000,
		help="Max values to print per dataset before truncation (0 = no truncation).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_file: Path = args.input_file
	max_values: int = args.max_values

	if not input_file.exists() or not input_file.is_file():
		raise FileNotFoundError(f"Input file not found: {input_file}")

	# Determine output file path
	if args.output_file:
		txt_path = args.output_file
	else:
		txt_path = input_file.with_suffix(".txt")

	convert_hdf5_to_txt(input_file, txt_path, max_values=max_values)
	print(f"Converted: {input_file.name} -> {txt_path.name}")
	print("Done.")


if __name__ == "__main__":
	main()
