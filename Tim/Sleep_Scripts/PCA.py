import mne
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

def channel_pca(raw, band):
    bands = {
        'noise':[0,0.5], 'delta':[0.5,5],
        'theta':[6,10], 'sigma':[11,17],
        'beta':[22,30], 'gamma':[35,45],
        'total':[0,30]
    }

    low, high = bands[band]
    raw_filt = raw.copy().filter(low, high, fir_design='firwin')

    data, ch_names = raw.get_data(return_times=False, picks='eeg'), raw.ch_names
    data = data.astype(np.float32)
    # shape = (n_channels, n_times)

    pca = PCA(n_components=5)
    pca.fit(data.T)  # (time × channels)
    loadings = pca.components_  # (n_components × n_channels)

    # top channels
    pc1_top = np.argmax(np.abs(loadings[0]))
    pc2_top = np.argmax(np.abs(loadings[1]))

    print("Top channels:")
    print("PC1 ->", ch_names[pc1_top])
    print("PC2 ->", ch_names[pc2_top])

    # plot topomaps for PC1 and PC2
    for i, pc_top in enumerate([pc1_top, pc2_top]):
        evoked = mne.EvokedArray(loadings[i][:, np.newaxis], raw_filt.info, tmin=0)
        evoked.comment = f"PC{i+1}"
        fig = evoked.plot_topomap(times=0, scalings=1, cmap="RdBu_r",
                                  size=3, time_format=f"PC{i+1}",
                                  outlines="head", show=False)

        # overlay star on top channel
        pos = raw_filt.info["chs"][pc_top]["loc"][:3]
        montage_pos = raw_filt.get_montage().get_positions()["ch_pos"]
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