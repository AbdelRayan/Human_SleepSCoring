import DeepNeuralNetworkSleep.hdf5_files.datasets_extraction as DE
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

file = 'D:/EEG_Data_stage/datasets/intra_cranial_dataset.h5'
output = 'D:/EEG_Data_stage/datasets/'

features = {"W Index":np.array([]),"R Index":np.array([]),"N Index":np.array([]),"1 Index":np.array([]),
            "2 Index":np.array([]),"3 Index":np.array([])
            ,"4 Index":np.array([]),"Noise":np.array([]),"Theta":np.array([]),"Delta":np.array([])
            ,"Aperiodic Fit":np.array([]),"DFA":np.array([]),"MSE":np.array([])}

with h5py.File(file, "r") as f:
    # Loop over nights
    for group_name in f.keys():
        group = f[group_name]
        print("Group:", group_name)

        # Check dataset exists inside the group
        if "Features" not in group:
            print("  NO 'Features' dataset found in", group_name)
            continue

        raw_features = group["Features"][:]  # <--- FIXED PATH

        # Assign each column
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
plt.savefig(output + "feature_histograms.pdf")
plt.close()



