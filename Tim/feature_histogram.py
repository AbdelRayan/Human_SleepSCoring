import DeepNeuralNetworkSleep.hdf5_files.datasets_extraction as DE
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

file = 'D:/EEG_Data_stage/datasets/training_features.npz'
output = 'D:/EEG_Data_stage/datasets/'

features = {"W Index":np.array([]),"R Index":np.array([]),"N Index":np.array([]),"1 Index":np.array([]),
            "2 Index":np.array([]),"3 Index":np.array([])
            ,"4 Index":np.array([]),"Noise":np.array([]),"Theta":np.array([]),"Delta":np.array([])
            ,"Aperiodic Fit":np.array([]),"DFA":np.array([]),"MSE":np.array([])}


data = np.load(file, allow_pickle=True)
print(len(data['d']))
print(data['d'])

# Access the features array
raw_features = data['d']  # shape: (num_samples, num_features)

# Append each column to your features dict
for key, col_idx in zip(features.keys(), range(raw_features.shape[1])):
    features[key] = np.append(features[key], raw_features[:, col_idx])

static_colors = {
    "W Index":        "#1f77b4",  # blue
    "R Index":        "#ff7f0e",  # orange
    "N Index":        "#2ca02c",  # green
    "1 Index":        "#d62728",  # red
    "2 Index":        "#9467bd",  # purple
    "3 Index":        "#8c564b",  # brown
    "4 Index":        "#e377c2",  # pink
    "Noise":          "#7f7f7f",  # gray
    "Theta":          "#bcbd22",  # olive
    "Delta":          "#17becf",  # teal
    "Aperiodic Fit":  "#aec7e8",  # light blue
    "DFA":            "#ffbb78",  # light orange
    "MSE":            "#98df8a",  # light green
}

fig, axes = plt.subplots(3, 5, figsize=(30, 18))
axes = axes.flatten()

for idx, (key, values) in enumerate(features.items()):
    ax = axes[idx]

    ax.hist(values, bins=40, range=(0,1),
            color=static_colors[key],
            edgecolor="black")

    ax.set_xlim(0, 1)

    ax.set_ylabel(key, fontsize=26)
    ax.set_xlabel("Value", fontsize=26)

    ax.tick_params(axis="both", which="major", labelsize=24)

    ax.grid(True)

# Turn off leftover subplot
for j in range(len(features), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(output + "feature_histograms_training.svg")
plt.close()



