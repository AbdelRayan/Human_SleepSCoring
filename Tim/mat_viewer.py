from scipy.io import loadmat
import h5py
import numpy as np

mat = loadmat('post_trial1_2017-09-28_11-30-59-states.mat')
mat_hpc = loadmat('HPC_100_CH46.continuous.mat')

with h5py.File("Rat3.h5", "r") as f:
    # List top-level groups
    print("Keys:", list(f.keys()))

    # Explore a subgroup
    group = f["HPC_100_CH46.continuous"]
    print("Subkeys:", list(group.keys()))
    for key in group.keys():
        print(group[key])

print(len(mat['states']))
print(len(mat_hpc['HPC'])/2500)