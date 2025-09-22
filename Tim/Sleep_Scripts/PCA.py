import mne
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D

def channel_pca(raw, band, electrode_text):
    """
    Run PCA on EEG channels for a given frequency band, print top channels,
    and plot PCA loadings in 3D using electrode coordinates from a text string.

    Parameters
    ----------
    raw : mne.io.Raw
        Preloaded raw EEG data with montage set.
    band : str
        One of ['noise','delta','theta','sigma','beta','gamma','total'].
    electrode_text : str
        Multiline string with electrode info, format:
        Name,X,Y,Z,other_columns...
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

    # Run Incremental PCA
    pca = IncrementalPCA(n_components=2, batch_size=10000)
    pca.fit(data.T)
    loadings = pca.components_  # shape = (n_components × n_channels)

    # Identify top channels
    pc1_top = np.argmax(np.abs(loadings[0]))
    pc2_top = np.argmax(np.abs(loadings[1]))
    print("Top channels:")
    print("PC1 ->", ch_names[pc1_top])
    print("PC2 ->", ch_names[pc2_top])

    # Parse electrode coordinates from text
    lines = [line.strip() for line in electrode_text.strip().splitlines() if line.strip()]
    coord_dict = {}
    for line in lines:
        parts = line.split(",")
        name = parts[0]
        x, y, z = map(float, parts[1:4])
        coord_dict[name] = np.array([x, y, z])

    # Build arrays for plotting, matching ch_names order
    coords = np.array([coord_dict[ch] for ch in ch_names])

    # 3D plot with PCA loadings
    for i, pc_top in enumerate([pc1_top, pc2_top]):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2],
                        c=loadings[i], cmap='RdBu_r', s=100, edgecolors='k')
        plt.colorbar(sc, ax=ax, label=f'PC{i+1} Loading (AU)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.title(f'PC{i+1} Loadings')

        # Highlight top channel
        ax.text(coords[pc_top,0], coords[pc_top,1], coords[pc_top,2],
                ch_names[pc_top], color='gold', fontsize=12)

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