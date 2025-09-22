import mne
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import re

def channel_pca(raw, band):
    """
    Run PCA on EEG channels for a given frequency band, print top channels,
    and plot PCA loadings as topomaps with stars on top channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Preloaded raw EEG data with montage set.
    band : str
        One of ['noise','delta','theta','sigma','beta','gamma','total'].
    """
    # Define frequency bands
    bands = {
        'noise':[0,0.5], 'delta':[0.5,5],
        'theta':[6,10], 'sigma':[11,17],
        'beta':[22,30], 'gamma':[35,45],
        'total':[0,30]
    }

    if band not in bands:
        raise ValueError(f"Band '{band}' not recognized")

    low, high = bands[band]

    # Filter raw data
    raw_filt = raw.copy().filter(low, high, fir_design='firwin')

    # Pick EEG channels only
    eeg_picks = mne.pick_types(raw_filt.info, eeg=True)
    ch_names = [raw_filt.info['ch_names'][i] for i in eeg_picks]

    # Extract data and convert to float32 to save memory
    data = raw_filt.get_data(picks=eeg_picks).astype(np.float32)
    # shape = (n_channels, n_times)

    # Run Incremental PCA
    pca = IncrementalPCA(n_components=2, batch_size=10000)
    pca.fit(data.T)  # shape = (n_times × n_channels)
    loadings = pca.components_  # shape = (n_components × n_channels)

    # Identify top channels
    pc1_top = np.argmax(np.abs(loadings[0]))
    pc2_top = np.argmax(np.abs(loadings[1]))
    print("Top channels:")
    print("PC1 ->", ch_names[pc1_top])
    print("PC2 ->", ch_names[pc2_top])

    # Create EEG-only Info object and attach montage
    info_eeg = mne.create_info(ch_names=ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    info_eeg.set_montage(raw_filt.get_montage())

    # Plot topomaps for PC1 and PC2
    for i, pc_top in enumerate([pc1_top, pc2_top]):
        evoked = mne.EvokedArray(loadings[i][:, np.newaxis], info_eeg, tmin=0)
        evoked.comment = f"PC{i+1}"
        fig = evoked.plot_topomap(times=0, scalings=1, cmap="RdBu_r",
                                  size=3, time_format=f"PC{i+1}",
                                  outlines="head", show=False)

        # Overlay star on top channel
        montage_pos = info_eeg.get_montage().get_positions()["ch_pos"]
        xy = montage_pos[ch_names[pc_top]][:2]  # 2D xy projection
        plt.scatter(xy[0], xy[1], c="gold", s=200, marker="*", edgecolors="k")
        plt.show()
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