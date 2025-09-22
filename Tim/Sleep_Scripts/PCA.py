import mne
import numpy as np
from sklearn.decomposition import PCA

def channel_pca(raw, band):
    bands = {
        'noise':[0,0.5], 'delta':[0.5,5],
        'theta':[6,10], 'sigma':[11,17],
        'beta':[22,30], 'gamma':[35,45],
        'total':[0,30]
    }

    low, high = bands[band]
    raw_filt = raw.copy().filter(low, high, fir_design='firwin')

    data, ch_names = raw_filt.get_data(return_times=False), raw_filt.ch_names
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
