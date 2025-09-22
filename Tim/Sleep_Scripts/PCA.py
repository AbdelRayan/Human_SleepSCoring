import mne
import numpy as np
from sklearn.decomposition import PCA

def channel_pca(raw, band):
    bands = {'noise':[0,0.5], 'delta':[0.5,5],
    'theta':[6,10], 'sigma':[11,17], 'beta':[22,30],
    'gamma':[35,45], 'total':[0,30]}

    low, high = bands[band][0], bands[band][1]

    raw.filter(low, high, fir_design='firwin')

    data, ch_names = raw.get_data(return_times=False), raw.ch_names
    # shape = (n_channels, n_times)

    pca = PCA(n_components=5)  # you only need first 2 PCs
    pca.fit(data.T)  # transpose to (time × channels)

    # 5. Inspect loadings (components_)
    loadings = pca.components_  # shape (n_components, n_channels)

    # Pick the channel with max abs weight for each PC
    pc1_top = np.argmax(np.abs(loadings[0]))
    pc2_top = np.argmax(np.abs(loadings[1]))

    print("Top channels:")
    print("PC1 ->", ch_names[pc1_top])
    print("PC2 ->", ch_names[pc2_top])
