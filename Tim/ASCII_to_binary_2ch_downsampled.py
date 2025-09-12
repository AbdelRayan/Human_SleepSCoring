import numpy as np
import os
import shutil

def convert_brainvision_ascii(vhdr_file, out_dir="converted", out_prefix=None):
    """
    Convert BrainVision ASCII .dat (vectorized, channels in rows)
    to binary multiplexed format and patch .vhdr accordingly.

    Parameters
    ----------
    vhdr_file : str
        Path to the original .vhdr file.
    out_dir : str
        Directory where converted files will be stored.
    out_prefix : str or None
        Prefix for new files. If None, uses the original base name.

    Returns
    -------
    new_vhdr : str
        Path to the new patched .vhdr file (ready for MNE).
    """
    base_dir = os.path.dirname(vhdr_file)
    base_name = os.path.splitext(os.path.basename(vhdr_file))[0]
    if out_prefix is None:
        out_prefix = base_name

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Parse .vhdr to find data and marker files
    dat_file = None
    vmrk_file = None
    with open(vhdr_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("DataFile="):
                dat_file = line.split("=", 1)[1].strip()
            if line.startswith("MarkerFile="):
                vmrk_file = line.split("=", 1)[1].strip()

    if dat_file is None:
        raise ValueError("Could not find DataFile= in .vhdr")

    dat_file = os.path.join(base_dir, dat_file)
    if vmrk_file is not None:
        vmrk_file = os.path.join(base_dir, vmrk_file)

    # Output paths
    new_dat = os.path.join(out_dir, f"{out_prefix}.dat")
    new_vhdr = os.path.join(out_dir, f"{out_prefix}.vhdr")
    new_vmrk = os.path.join(out_dir, f"{out_prefix}.vmrk")

    # ---- Step 1: Convert ASCII .dat to binary multiplexed ----
    channels = []
    data = []

    with open(dat_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ch_name = parts[0]
            if ch_name == 'EOG1' or ch_name == 'Cz':
                if not os.path.exists(new_dat):
                    orig_values = np.array(parts[1:], dtype=np.float32)
                    n = len(orig_values) - (len(orig_values) % 4)
                    orig_values = orig_values[:n]
                    values = orig_values.reshape(-1, 4).mean(axis=1).astype(np.float32)
                    data.append(values)
                channels.append(ch_name)

    if not os.path.exists(new_dat):
        data = np.vstack(data)
        multiplexed = data.T.astype(np.float32).ravel(order="C")

        with open(new_dat, "wb") as f:
            f.write(multiplexed.tobytes())

        print(f"Converted {len(channels)} channels × {data.shape[1]} samples")
        print(f"Saved new binary .dat → {new_dat}")

    # ---- Step 2: Copy and patch .vhdr ----
    with open(vhdr_file, "r", encoding="utf-8", errors="ignore") as f_in:
        vhdr_lines = f_in.readlines()

    has_binary_infos = any(line.strip().startswith("[Binary Infos]") for line in vhdr_lines)

    with open(new_vhdr, "w", encoding="utf-8") as f_out:
        for line in vhdr_lines:
            if line.startswith("DataFile="):
                f_out.write(f"DataFile={os.path.basename(new_dat)}\n")
            elif line.startswith("MarkerFile="):
                # force new marker file
                f_out.write(f"MarkerFile={os.path.basename(new_vmrk)}\n")
            elif line.startswith("SamplingInterval="):
                f_out.write("SamplingInterval=4000\n")
            else:
                f_out.write(line)

        # If no MarkerFile line was present at all, add one in [Common Infos]
        if not any("MarkerFile=" in l for l in vhdr_lines):
            f_out.write(f"MarkerFile={os.path.basename(new_vmrk)}\n")

        # If no [Binary Infos] section, append one
        if not has_binary_infos:
            f_out.write("\n[Binary Infos]\n")
            f_out.write("DataFormat=BINARY\n")
            f_out.write("DataOrientation=MULTIPLEXED\n")
            f_out.write("BinaryFormat=IEEE_FLOAT_32\n")

    print(f"Patched .vhdr → {new_vhdr}")

    # ---- Step 3: Handle marker file ----
    if vmrk_file and os.path.exists(vmrk_file):
        shutil.copy(vmrk_file, new_vmrk)
        print(f"Copied .vmrk → {new_vmrk}")
    else:
        with open(new_vmrk, "w", encoding="utf-8") as f:
            f.write("Brain Vision Data Exchange Marker File, Version 1.0\n")
            f.write("; Created by converter\n")
            f.write("[Common Infos]\n")
            f.write("Codepage=UTF-8\n")
            f.write("[Marker Infos]\n")
            f.write("; no markers\n")
            f.write("[Marker Data]\n")
        print(f"Created empty .vmrk → {new_vmrk}")

    print(f"Patched .vhdr → {new_vhdr}")

    # ---- Step 3: Copy .vmrk if it exists ----
    if vmrk_file and new_vmrk:
        shutil.copy(vmrk_file, new_vmrk)
        print(f"Copied .vmrk → {new_vmrk}")
    else:
        print("No MarkerFile= found in .vhdr → skipping .vmrk copy")

    return new_vhdr


if __name__ == "__main__":
    vhdr_file = "D:/Intercranial_sleep_data/2/iEEG/2_night1_02.vhdr"
    binary_file = "D:/converted_sleep_data/2"
    convert_brainvision_ascii(vhdr_file, binary_file)
