import numpy as np
import mne
from scipy.signal import welch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mne.time_frequency import psd_array_welch

def rank_channels_by_bandpower(raw, bands=None, n_top=5):
    """
    Compute and rank EEG channels by average bandpower for given frequency bands.

    Parameters
    ----------
    raw : mne.io.Raw
        Preloaded raw EEG data.
    bands : dict
        Dictionary of frequency bands, e.g. {"delta": (0.5, 5), "theta": (6, 10)}.
        Defaults to common EEG bands.
    n_top : int
        Number of top channels to display per band.

    Returns
    -------
    bandpowers : dict
        Dictionary mapping band -> array of power per channel.
    """

    if bands is None:
        bands = {
            "delta": (0.5, 5),
            "theta": (6, 10),
            "sigma": (11, 17),
            "beta":  (22, 30),
            "gamma": (35, 45),
            "total": (0.5, 45)
        }

    # Get EEG data and channel names
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=eeg_picks)
    fs = raw.info['sfreq']
    ch_names = [raw.info['ch_names'][i] for i in eeg_picks]

    # Compute PSD
    psds, freqs = psd_array_welch(data, sfreq=fs, fmin=0.5, fmax=45, n_fft=int(fs*4))

    bandpowers = {}
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        power_vals = np.trapz(psds[:, mask], freqs[mask], axis=1)
        bandpowers[band] = power_vals

        # Rank + print top channels
        sorted_idx = np.argsort(power_vals)[::-1]
        print(f"\nTop {n_top} channels for {band}:")
        for i in range(min(n_top, len(sorted_idx))):
            print(f"  {ch_names[sorted_idx[i]]}: {power_vals[sorted_idx[i]]:.4f}")
    print(freqs[:10], freqs[-10:])
    return bandpowers


def channel_pca_multiband(raw, electrode_text, bands=None, n_components=2, standardize=True):
    """
    Run PCA on EEG channels for multiple frequency bands, print top channels,
    and plot PCA loadings in 3D using electrode coordinates.

    Parameters
    ----------
    raw : mne.io.Raw
        Preloaded raw EEG data.
    electrode_text : str
        Multiline string with electrode info, format: Name,X,Y,Z
    bands : dict
        Dictionary of frequency bands. Example:
        {"delta": (0.5,5), "theta": (6,10), "sigma": (11,17)}
        If None, defaults to common EEG bands.
    n_components : int
        Number of PCA components.
    standardize : bool
        Whether to z-score each channel before PCA.

    Returns
    -------
    pca_results : dict
        Dictionary with PCA loadings, top channels, and coordinates.
    """

    if bands is None:
        bands = {
            "delta": (0.5, 5),
            "theta": (6, 10),
            "sigma": (11, 17),
            "beta":  (22, 30),
            "gamma": (35, 45),
            "total": (0.5, 45)
        }

    # Pick EEG channels, exclude bads
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    ch_names = [raw.info['ch_names'][i] for i in eeg_picks]

    # Extract data
    data = raw.get_data(picks=eeg_picks).astype(np.float32)

    # Parse electrode coordinates
    lines = [line.strip() for line in electrode_text.strip().splitlines() if line.strip()]
    coord_dict = {}
    for line in lines:
        parts = line.split(",")
        name = parts[0]
        x, y, z = map(float, parts[1:4])
        coord_dict[name] = np.array([x, y, z])

    # Helper to find coordinate
    def find_coord(ch):
        if ch in coord_dict:
            return coord_dict[ch]
        m = re.search(r"(\D+)(\d+)$", ch)
        if m:
            base, num = m.groups()
            for alt in [f"{base}{int(num):02d}", f"{base}{int(num)}"]:
                if alt in coord_dict:
                    return coord_dict[alt]
        return None

    # Filter channels to those with coordinates
    coords_list = []
    valid_ch_names = []
    valid_picks = []

    for idx, ch in enumerate(ch_names):
        coord = find_coord(ch)
        if coord is not None:
            coords_list.append(coord)
            valid_ch_names.append(ch)
            valid_picks.append(idx)
        else:
            print(f"Warning: {ch} missing coordinates, excluding from PCA.")

    if len(valid_ch_names) < 2:
        raise RuntimeError("Not enough channels with coordinates to run PCA.")

    coords = np.array(coords_list)
    data = data[valid_picks, :]  # Only valid channels

    # Standardize channels if requested
    if standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(data)
    else:
        X_std = data

    # PCA
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_std.T)  # shape: (n_timepoints, n_components)

    # Identify top channels per component
    pc_top_channels = []
    for i in range(n_components):
        top_idx = np.argmax(np.abs(Z[:, i]))
        pc_top_channels.append(valid_ch_names[top_idx])
        print(f"PC{i+1} top channel: {valid_ch_names[top_idx]}")

        # Optional: 3D plot
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2],
                        c=Z[:, i], cmap='RdBu_r', s=100, edgecolors='k')
        plt.colorbar(sc, ax=ax, label=f'PC{i+1} score')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'PC{i+1} Scores - Top: {valid_ch_names[top_idx]}')
        ax.text(coords[top_idx,0], coords[top_idx,1], coords[top_idx,2],
                valid_ch_names[top_idx], color='gold', fontsize=12)
        plt.show()

    return {
        "pca_loadings": pca.components_,
        "top_channels": pc_top_channels,
        "coords": coords
    }

def parse_positions_with_mapping(text, final_names, to_meters=True):
    """
    Parse electrode coordinates and remap them to a given list of channel names.

    Parameters
    ----------
    text : str
        Multiline string with rows like:
        'TBAL01,-14.7597,-6.0460,-25.5850,3.0,0.0,0.0,1.0'
    final_names : list of str
        Correct channel names in the same order as rows in `text`.
    to_meters : bool
        If True, convert mm → m for MNE compatibility.

    Returns
    -------
    coords : dict
        Dictionary mapping `final_names` to (x, y, z) coordinates.
    """
    coords = {}
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) != len(final_names):
        raise ValueError(f"Number of lines ({len(lines)}) != number of final names ({len(final_names)})")

    for new_name, line in zip(final_names, lines):
        parts = line.split(",")
        if len(parts) < 4:
            continue
        x, y, z = map(float, parts[1:4])
        if to_meters:
            coords[new_name] = np.array([x, y, z]) / 1000.0
        else:
            coords[new_name] = np.array([x, y, z])
    return coords

def plot_ieeg_3d(electrode_text):
    """
    Plot intracranial electrode positions in 3D.

    Parameters
    ----------
    electrode_text : str
        Multiline string with electrode info, format:
        Name,X,Y,Z,other_columns...
    """
    lines = [line.strip() for line in electrode_text.strip().splitlines() if line.strip()]

    names, coords = [], []
    for line in lines:
        parts = line.split(",")
        name = parts[0]
        x, y, z = map(float, parts[1:4])  # take the first three coordinates
        names.append(name)
        coords.append([x, y, z])

    coords = np.array(coords)

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=100, c='skyblue', edgecolors='k')

    # Add labels
    for i, name in enumerate(names):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], name, fontsize=9, color='red')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Intracranial Electrode Positions')
    plt.show()