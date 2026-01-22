"""
By: Tim Veldema
Date: 19/11/2025

Utilities for converting BrainVision ASCII EEG recordings to
MNE-compatible binary format.

This module provides two high-level conversion workflows:

1. convert_brainvision_ascii
   Converts vectorized ASCII .dat files to multiplexed binary format and
   patches the associated .vhdr/.vmrk files. EEG channels can be optionally
   re-referenced to Cz (bipolarization for extra-cranial channels), while
   EOG and EMG channels are preserved or bipolarized as appropriate.
   Downsampling by 4 is performed in the same way as the native BrainVision
   ASCII export.

2. convert_brainvision_ascii_average
   Groups all contacts belonging to the same electrode (e.g. F3a, F3b, F3c)
   and produces a single averaged signal per electrode. This is useful when
   only electrode-level activity is needed rather than individual contact
   signals. EOG channels remain unipolar and EMG is output as EMG1–EMG2.

Both functions generate:
    - A binary multiplexed .dat file (IEEE_FLOAT_32)
    - A patched .vhdr header describing the new channel layout
    - A copied or minimal .vmrk marker file

The outputs are fully compatible with MNE-Python and standard BrainVision
readers.

Typical use cases:
    • Simplifying high-contact EEG montages
    • Creating Cz-referenced bipolar montages
    • Averaging depth/strip electrode contacts
    • Cleaning ASCII exports produced from clinical EEG systems
"""
import re
import numpy as np
import os
import shutil
import mne


def convert_brainvision_ascii(
    vhdr_file,
    out_dir="converted",
    channel_select=None,
    downsample_factor=4,
):
    """
    Convert a BrainVision ASCII .dat file (vectorized, channels in rows) into a
    binary multiplexed .dat file and generate a patched .vhdr compatible with MNE.

    Behavior
    --------
    - If ``channel_select`` is None:
        → All EEG channels present in the ASCII file are processed.
    - If ``channel_select`` is a list:
        → Only those EEG channels are processed (if present in the file).

    EOG and EMG channels are handled independently of ``channel_select``:
        - EOG1 and EOG2 (unipolar) are always appended if present.
        - EMG1–EMG2 is always appended as a bipolar channel if both are present.

    Extra-cranial EEG channels are defined as:
        F1–F5, Fz
        C1–C5, Cz
        T1–T5
        O1–O5, Oz

    Extra-cranial channels (except Cz) are bipolarized against Cz if Cz exists.
    If Cz does not exist, all channels are written unipolarly.
    Cz itself is always written as a unipolar channel.

    Non-extra-cranial EEG channels are always written unipolarly.

    Downsampling
    ------------
    If ``downsample_factor`` > 1, data are downsampled by averaging over
    non-overlapping blocks. Trailing samples that do not fit an integer number
    of blocks are discarded. ``None`` or 1 disables downsampling.

    Output details
    --------------
    - Output .dat is written as IEEE_FLOAT_32, multiplexed by time.
    - Channel order is:
        1. Selected EEG channels (file order or ``channel_select`` order)
        2. EOG1, EOG2 (if present)
        3. EMG1-EMG2 (if present)
    - Bipolar channels are named ``<channel>-Cz`` or ``EMG1-EMG2``.

    Note
    ----
    The output .vhdr uses a fixed ``SamplingInterval=4000`` (250 Hz) and does not
    adjust this value if downsampling is applied.

    Parameters
    ----------
    vhdr_file : str
        Path to the original BrainVision .vhdr file.
    out_dir : str
        Output directory for converted files.
    channel_select : list[str] or None
        EEG channels to process. EOG and EMG channels are handled separately.
    downsample_factor : int or None
        Downsampling factor. Use None or 1 for no downsampling.

    Returns
    -------
    new_vhdr : str
        Path to the patched .vhdr file.
    """
    # Extra-cranial EEG channel regex
    extracranial_re = re.compile(
        r"^(F[1-5]|Fz|C[1-5]|Cz|T[1-5]|O[1-5]|Oz)$",
        flags=re.IGNORECASE
    )

    # Helper function that determines whether a channel is extracranial
    def is_extracranial(ch):
        return extracranial_re.match(ch) is not None

    base_dir = os.path.dirname(vhdr_file)
    base_name = os.path.splitext(os.path.basename(vhdr_file))[0]
    out_prefix = base_name

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------- PARSE .vhdr -----------------------
    dat_file = vmrk_file = None
    with open(vhdr_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("DataFile="):
                dat_file = line.split("=", 1)[1].strip()
            elif line.startswith("MarkerFile="):
                vmrk_file = line.split("=", 1)[1].strip()

    if dat_file is None:
        raise ValueError("Missing DataFile= in header")

    dat_file = os.path.join(base_dir, dat_file)
    if vmrk_file:
        vmrk_file = os.path.join(base_dir, vmrk_file)

    new_dat = os.path.join(out_dir, f"{out_prefix}.dat")
    new_vhdr = os.path.join(out_dir, f"{out_prefix}.vhdr")
    new_vmrk = os.path.join(out_dir, f"{out_prefix}.vmrk")

    # ----------------------- LOAD ASCII .dat -----------------------
    all_channels = {}

    with open(dat_file, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            ch = parts[0]
            vals = np.asarray(parts[1:], dtype=np.float32)

            if downsample_factor and downsample_factor > 1:
                L = len(vals)
                Lcrop = L - (L % downsample_factor)
                vals = vals[:Lcrop].reshape(-1, downsample_factor).mean(axis=1)

            all_channels[ch] = vals.astype(np.float32)

    n_samples = len(next(iter(all_channels.values())))
    cz_available = "Cz" in all_channels

    # Determine which channels to process
    if channel_select is None:
        channels_to_use = list(all_channels.keys())
    else:
        channels_to_use = [ch for ch in channel_select if ch in all_channels]

    # ----------------------- BUILD OUTPUT DATA -----------------------
    bipolar_data = []
    bipolar_names = []

    for ch in channels_to_use:
        data = all_channels[ch]

        if is_extracranial(ch) and ch != "Cz" and cz_available:
            # bipolar if extracranial, derived using the Cz channel if its available
            bipolar_data.append(data - all_channels["Cz"])
            bipolar_names.append(f"{ch}-Cz")
        else:
            # unipolar
            bipolar_data.append(data)
            bipolar_names.append(ch)

    # Add EOG always
    for eog in ["EOG1", "EOG2"]:
        if eog in all_channels:
            bipolar_data.append(all_channels[eog])
            bipolar_names.append(eog)

    # Add EMG bipolar always
    if "EMG1" in all_channels and "EMG2" in all_channels:
        bipolar_data.append(all_channels["EMG1"] - all_channels["EMG2"])
        bipolar_names.append("EMG1-EMG2")

    bipolar_data = np.vstack(bipolar_data)

    # ----------------------- WRITE BINARY .dat -----------------------
    multiplexed = bipolar_data.T.astype(np.float32).ravel(order="C")
    with open(new_dat, "wb") as f:
        f.write(multiplexed.tobytes())

    # ----------------------- WRITE .vhdr -----------------------
    with open(new_vhdr, "w", encoding="utf-8") as f_out:
        f_out.write("Brain Vision Data Exchange Header File Version 2.0\n")
        f_out.write("; Converted with selective Cz-referencing\n\n")
        f_out.write("[Common Infos]\n")
        f_out.write("Codepage=UTF-8\n")
        f_out.write(f"DataFile={os.path.basename(new_dat)}\n")
        f_out.write(f"MarkerFile={os.path.basename(new_vmrk)}\n")
        f_out.write("DataFormat=BINARY\n")
        f_out.write("DataOrientation=MULTIPLEXED\n")
        f_out.write("DataType=TIMEDOMAIN\n")
        f_out.write(f"NumberOfChannels={len(bipolar_names)}\n")
        f_out.write(f"DataPoints={n_samples}\n")
        f_out.write("SamplingInterval=4000\n")

        f_out.write("\n[Binary Infos]\n")
        f_out.write("BinaryFormat=IEEE_FLOAT_32\n")

        f_out.write("\n[Channel Infos]\n")
        for i, ch in enumerate(bipolar_names, start=1):
            f_out.write(f"Ch{i}={ch},,,µV\n")

    # ----------------------- WRITE .vmrk -----------------------
    if vmrk_file and os.path.exists(vmrk_file):
        shutil.copy(vmrk_file, new_vmrk)
    else:
        with open(new_vmrk, "w", encoding="utf-8") as f:
            f.write("Brain Vision Data Exchange Marker File, Version 1.0\n")
            f.write("[Common Infos]\n")
            f.write("Codepage=UTF-8\n")
            f.write("[Marker Infos]\n")
            f.write("[Marker Data]\n")

    return new_vhdr

def convert_brainvision_ascii_average(vhdr_file, out_dir="converted"):
    """
    Does the same as convert_brainvision_ascii, but instead of treating each
    contact as a separate channel, this function:

    1. Selects all channels belonging to the same electrode
       ("full electrodes" → e.g. F3, F3a, F3b, etc.)
    2. Averages across all contacts belonging to that electrode.

    Outputs:
        - One averaged channel per electrode
        - EOG1, EOG2 kept unipolar
        - EMG1–EMG2 as bipolar
    """
    base_dir = os.path.dirname(vhdr_file)
    base_name = os.path.splitext(os.path.basename(vhdr_file))[0]
    out_prefix = base_name
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------- Parse .vhdr -------------------------
    dat_file, vmrk_file = None, None
    with open(vhdr_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DataFile="):
                dat_file = line.split("=", 1)[1].strip()
            elif line.startswith("MarkerFile="):
                vmrk_file = line.split("=", 1)[1].strip()

    if dat_file is None:
        raise ValueError("Could not find DataFile= in .vhdr")

    dat_file = os.path.join(base_dir, dat_file)
    if vmrk_file:
        vmrk_file = os.path.join(base_dir, vmrk_file)

    # Output paths
    new_dat = os.path.join(out_dir, f"{out_prefix}.dat")
    new_vhdr = os.path.join(out_dir, f"{out_prefix}.vhdr")
    new_vmrk = os.path.join(out_dir, f"{out_prefix}.vmrk")

    # ------------------------- Load ASCII data -------------------------
    electrode_groups = {}   # electrode_name → list of contact arrays
    eog_channels = {}
    emg_channels = {}

    contact_pattern = re.compile(r"^([A-Za-z]+)(\d*)$", flags=re.IGNORECASE)

    with open(dat_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            ch_name = parts[0]
            values = np.array(parts[1:], dtype=np.float32)

            # Downsample by 4 (exactly as original function)
            n = len(values) - (len(values) % 4)
            values = values[:n].reshape(-1, 4).mean(axis=1).astype(np.float32)

            # ---------- Special channels ----------
            if ch_name.startswith("EOG"):
                eog_channels[ch_name] = values
                continue

            if ch_name.startswith("EMG"):
                emg_channels[ch_name] = values
                continue

            # ---------- EEG electrodes ----------
            m = contact_pattern.match(ch_name)
            if not m:
                print(f"Skipping channel {ch_name} (not letter+digits)")
                continue

            electrode = m.group(1)   # e.g. F3a → "F"
            # Better grouping: electrode should be full letter+number prefix
            # Example: F3a → group as "F3"
            prefix = re.match(r"[A-Za-z]+\d+", ch_name)
            if prefix:
                electrode = prefix.group(0)

            if electrode not in electrode_groups:
                electrode_groups[electrode] = []

            electrode_groups[electrode].append(values)

    # ------------------------- Average contacts per electrode -------------------------
    averaged = {}
    for elec, arrs in electrode_groups.items():
        print(f"Averaging electrode {elec} from {len(arrs)} contacts")
        arrs = np.vstack(arrs)
        averaged[elec] = np.mean(arrs, axis=0)

    n_samples = len(next(iter(averaged.values())))

    # ------------------------- Build output matrix -------------------------
    out_data = []
    out_names = []

    # EEG averaged
    for elec in sorted(averaged.keys()):
        out_data.append(averaged[elec])
        out_names.append(elec)

    # EOG: unipolar
    if "EOG1" in eog_channels:
        out_data.append(eog_channels["EOG1"])
        out_names.append("EOG1")
    if "EOG2" in eog_channels:
        out_data.append(eog_channels["EOG2"])
        out_names.append("EOG2")

    # EMG bipolar
    if "EMG1" in emg_channels and "EMG2" in emg_channels:
        out_data.append(emg_channels["EMG1"] - emg_channels["EMG2"])
        out_names.append("EMG1-EMG2")

    out_data = np.vstack(out_data)

    # ------------------------- Write binary .dat -------------------------
    multiplexed = out_data.T.astype(np.float32).ravel(order="C")
    with open(new_dat, "wb") as f:
        f.write(multiplexed.tobytes())

    # ------------------------- Write new .vhdr -------------------------
    with open(new_vhdr, "w", encoding="utf-8") as f_out:
        f_out.write("Brain Vision Data Exchange Header File Version 2.0\n")
        f_out.write("; Averaged full-electrode signals\n\n")

        f_out.write("[Common Infos]\n")
        f_out.write("Codepage=UTF-8\n")
        f_out.write(f"DataFile={os.path.basename(new_dat)}\n")
        f_out.write(f"MarkerFile={os.path.basename(new_vmrk)}\n")
        f_out.write("DataFormat=BINARY\n")
        f_out.write("DataOrientation=MULTIPLEXED\n")
        f_out.write("DataType=TIMEDOMAIN\n")
        f_out.write(f"NumberOfChannels={len(out_names)}\n")
        f_out.write(f"DataPoints={n_samples}\n")
        f_out.write("SamplingInterval=4000\n")

        f_out.write("\n[Binary Infos]\n")
        f_out.write("BinaryFormat=IEEE_FLOAT_32\n")

        f_out.write("\n[Channel Infos]\n")
        for i, ch in enumerate(out_names, start=1):
            f_out.write(f"Ch{i}={ch},,,µV\n")

    # ------------------------- Write/Copy .vmrk -------------------------
    if vmrk_file and os.path.exists(vmrk_file):
        shutil.copy(vmrk_file, new_vmrk)
    else:
        with open(new_vmrk, "w", encoding="utf-8") as f:
            f.write("Brain Vision Data Exchange Marker File, Version 1.0\n")
            f.write("[Common Infos]\n")
            f.write("Codepage=UTF-8\n")
            f.write("[Marker Infos]\n")
            f.write("[Marker Data]\n")

    return new_vhdr


