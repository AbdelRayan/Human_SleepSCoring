import mne
import numpy as np
from sklearn.decomposition import PCA

def channel_pca(raw, band):
    bands = {'noise_band':[0,0.5], 'delta_band':[0.5,5],
    'theta_band':[6,10], 'sigma_band':[11,17], 'beta_band':[22,30],
    'gamma_band':[35,45], 'total_band':[0,30]}

    # 2. Bandpass filter to target band (example: sigma 11–16 Hz)
    raw.filter(11, 16, fir_design='firwin')

    # 3. Get data matrix (channels × time)
    data, ch_names = raw.get_data(return_times=False), raw.ch_names
    # shape = (n_channels, n_times)

    # 4. Run PCA
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
