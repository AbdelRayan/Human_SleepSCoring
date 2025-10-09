import re

import numpy as np
import os
import shutil
import mne

def convert_brainvision_ascii(vhdr_file, out_dir="converted", channel_select=None):
    """
    Convert BrainVision ASCII .dat (vectorized, channels in rows)
    to binary multiplexed format and patch .vhdr accordingly.

    Differences vs. original:
    - Channels are replaced with bipolar derivations:
        * EEG vs Cz (for all in channel_select)
        * EOG1-EOG2
        * EMG1-EMG2

    Parameters
    ----------
    vhdr_file : str
        Path to the original .vhdr file.
    out_dir : str
        Directory where converted files will be stored.
    channel_select : list[str] or None
        EEG channels to re-reference against Cz.
        EOG1/2 and EMG1/2 are always included automatically.

    Returns
    -------
    new_vhdr : str
        Path to the new patched .vhdr file (ready for MNE).
    """
    base_dir = os.path.dirname(vhdr_file)
    base_name = os.path.splitext(os.path.basename(vhdr_file))[0]
    out_prefix = base_name

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Parse .vhdr to find data and marker files
    dat_file, vmrk_file = None, None
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

    # ---- Step 1: Load ASCII .dat ----
    all_channels = {}
    with open(dat_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ch_name = parts[0]
            values = np.array(parts[1:], dtype=np.float32)
            n = len(values) - (len(values) % 4)
            values = values[:n].reshape(-1, 4).mean(axis=1).astype(np.float32)
            all_channels[ch_name] = values

    n_samples = len(next(iter(all_channels.values())))

    # ---- Step 2: Build bipolar derivations ----
    bipolar_data = []
    bipolar_names = []

    # EEG channels vs Cz
    if channel_select is not None and "Cz" in all_channels:
        for ch in channel_select :
            if ch in all_channels and ch not in ['TR01', 'TL01', 'TR10', 'TL10']:
                new_name = f"{ch}-Cz"
                bipolar_data.append(all_channels[ch] - all_channels["Cz"])
                bipolar_names.append(new_name)
            elif ch in ['TR01', 'TL01', 'TR10', 'TL10']:
                bipolar_data.append(all_channels[ch])
                bipolar_names.append(ch)

    # EOG
    if "EOG1" in all_channels and "EOG2" in all_channels:
        bipolar_data.append(all_channels["EOG1"] - all_channels["EOG2"])
        bipolar_data.append(all_channels["EOG1"])
        bipolar_data.append(all_channels["EOG2"])
        bipolar_names.append("EOG1-EOG2")
        bipolar_names.append("EOG1")
        bipolar_names.append("EOG2")

    # EMG
    # if "EMG1" in all_channels and "EMG2" in all_channels:
    #     bipolar_data.append(all_channels["EMG1"] - all_channels["EMG2"])
    #     bipolar_names.append("EMG1-EMG2")
    if "EMG1" in all_channels and "EMG2" in all_channels:
        bipolar_data.append(all_channels["EMG1"] - all_channels["EMG2"])
        bipolar_data.append(all_channels["EMG1"])
        bipolar_data.append(all_channels["EMG2"])
        bipolar_names.append("EMG1-EMG2")
        bipolar_names.append("EMG1")
        bipolar_names.append("EMG2")

    bipolar_data = np.vstack(bipolar_data)

    # ---- Step 3: Write binary multiplexed file ----
    multiplexed = bipolar_data.T.astype(np.float32).ravel(order="C")
    with open(new_dat, "wb") as f:
        f.write(multiplexed.tobytes())

    print(f"Converted {len(bipolar_names)} bipolar channels × {n_samples} samples")
    print(f"Saved new binary .dat → {new_dat}")

    # ---- Step 4: Write patched .vhdr ----
    with open(new_vhdr, "w", encoding="utf-8") as f_out:
        f_out.write("Brain Vision Data Exchange Header File Version 2.0\n")
        f_out.write("; Converted with bipolar derivations\n\n")
        f_out.write("[Common Infos]\n")
        f_out.write("Codepage=UTF-8\n")
        f_out.write(f"DataFile={os.path.basename(new_dat)}\n")
        f_out.write(f"MarkerFile={os.path.basename(new_vmrk)}\n")
        f_out.write("DataFormat=BINARY\n")
        f_out.write("DataOrientation=MULTIPLEXED\n")
        f_out.write("DataType=TIMEDOMAIN\n")
        f_out.write(f"NumberOfChannels={len(bipolar_names)}\n")
        f_out.write(f"DataPoints={n_samples}\n")
        f_out.write("SamplingInterval=4000\n")  # 1000 Hz → 250 Hz

        f_out.write("\n[Binary Infos]\n")
        f_out.write("BinaryFormat=IEEE_FLOAT_32\n")

        f_out.write("\n[Channel Infos]\n")
        for i, ch in enumerate(bipolar_names, start=1):
            f_out.write(f"Ch{i}={ch},,,µV\n")

    print(f"Patched .vhdr → {new_vhdr}")

    # ---- Step 5: Copy/patch .vmrk ----
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

    return new_vhdr

def convert_brainvision_ascii_average(vhdr_file, out_dir="converted", channel_select=None):
    """
    Convert BrainVision ASCII .dat (vectorized, channels in rows)
    to binary multiplexed format and patch .vhdr accordingly.

    Differences vs. original:
    - Channels are replaced with bipolar derivations:
        * EEG vs Cz (for all in channel_select)
        * EOG1-EOG2
        * EMG1-EMG2

    Parameters
    ----------
    vhdr_file : str
        Path to the original .vhdr file.
    out_dir : str
        Directory where converted files will be stored.
    channel_select : list[str] or None
        EEG channels to re-reference against Cz.
        EOG1/2 and EMG1/2 are always included automatically.

    Returns
    -------
    new_vhdr : str
        Path to the new patched .vhdr file (ready for MNE).
    """
    base_dir = os.path.dirname(vhdr_file)
    base_name = os.path.splitext(os.path.basename(vhdr_file))[0]
    out_prefix = base_name

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Parse .vhdr to find data and marker files
    dat_file, vmrk_file = None, None
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

    # ---- Step 1: Load ASCII .dat ----
    all_channels = {}
    with open(dat_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ch_name = parts[0]
            letters, numbers = re.match(r"([A-Za-z]+)(\d+)", ch_name).groups()
            values = np.array(parts[1:], dtype=np.float32)
            n = len(values) - (len(values) % 4)
            values = values[:n].reshape(-1, 4).mean(axis=1).astype(np.float32)
            if all_channels[letters]:
                all_channels[letters].append(values)
            else:
                all_channels[letters] = values


    n_samples = len(next(iter(all_channels.values())))

    # ---- Step 2: Build bipolar derivations ----
    bipolar_data = []
    bipolar_names = []

    # EEG channels vs Cz
    if channel_select is not None and "Cz" in all_channels:
        for ch in channel_select :
            if ch in all_channels and ch not in ['TR01', 'TL01', 'TR10', 'TL10']:
                new_name = f"{ch}-Cz"
                bipolar_data.append(all_channels[ch] - all_channels["Cz"])
                bipolar_names.append(new_name)
            elif ch in ['TR01', 'TL01', 'TR10', 'TL10']:
                bipolar_data.append(all_channels[ch])
                bipolar_names.append(ch)

    # EOG
    if "EOG1" in all_channels and "EOG2" in all_channels:
        bipolar_data.append(all_channels["EOG1"] - all_channels["EOG2"])
        bipolar_data.append(all_channels["EOG1"])
        bipolar_data.append(all_channels["EOG2"])
        bipolar_names.append("EOG1-EOG2")
        bipolar_names.append("EOG1")
        bipolar_names.append("EOG2")

    # EMG
    # if "EMG1" in all_channels and "EMG2" in all_channels:
    #     bipolar_data.append(all_channels["EMG1"] - all_channels["EMG2"])
    #     bipolar_names.append("EMG1-EMG2")
    if "EMG1" in all_channels and "EMG2" in all_channels:
        bipolar_data.append(all_channels["EMG1"] - all_channels["EMG2"])
        bipolar_data.append(all_channels["EMG1"])
        bipolar_data.append(all_channels["EMG2"])
        bipolar_names.append("EMG1-EMG2")
        bipolar_names.append("EMG1")
        bipolar_names.append("EMG2")

    bipolar_data = np.vstack(bipolar_data)

    # ---- Step 3: Write binary multiplexed file ----
    multiplexed = bipolar_data.T.astype(np.float32).ravel(order="C")
    with open(new_dat, "wb") as f:
        f.write(multiplexed.tobytes())

    print(f"Converted {len(bipolar_names)} bipolar channels × {n_samples} samples")
    print(f"Saved new binary .dat → {new_dat}")

    # ---- Step 4: Write patched .vhdr ----
    with open(new_vhdr, "w", encoding="utf-8") as f_out:
        f_out.write("Brain Vision Data Exchange Header File Version 2.0\n")
        f_out.write("; Converted with bipolar derivations\n\n")
        f_out.write("[Common Infos]\n")
        f_out.write("Codepage=UTF-8\n")
        f_out.write(f"DataFile={os.path.basename(new_dat)}\n")
        f_out.write(f"MarkerFile={os.path.basename(new_vmrk)}\n")
        f_out.write("DataFormat=BINARY\n")
        f_out.write("DataOrientation=MULTIPLEXED\n")
        f_out.write("DataType=TIMEDOMAIN\n")
        f_out.write(f"NumberOfChannels={len(bipolar_names)}\n")
        f_out.write(f"DataPoints={n_samples}\n")
        f_out.write("SamplingInterval=4000\n")  # 1000 Hz → 250 Hz

        f_out.write("\n[Binary Infos]\n")
        f_out.write("BinaryFormat=IEEE_FLOAT_32\n")

        f_out.write("\n[Channel Infos]\n")
        for i, ch in enumerate(bipolar_names, start=1):
            f_out.write(f"Ch{i}={ch},,,µV\n")

    print(f"Patched .vhdr → {new_vhdr}")

    # ---- Step 5: Copy/patch .vmrk ----
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

    return new_vhdr


if __name__ == "__main__":
    base_path = "D:/Intercranial_sleep_data/2/iEEG/"
    vhdr_files = [os.path.join(base_path,"2_night1_01.vhdr"),
                  os.path.join(base_path,"2_night1_02.vhdr"),
                  os.path.join(base_path,"2_night1_03.vhdr")]
    binary_file = "D:/converted_sleep_data/2/con_full/"
    for file in vhdr_files:
        print(f"converting {file}")
        convert_brainvision_ascii(file, binary_file)
