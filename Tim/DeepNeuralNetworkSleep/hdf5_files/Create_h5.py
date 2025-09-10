import numpy as np
import os
import h5py
import multiprocessing as mp
from joblib import Parallel, delayed

from hdf5_creation import prepare_for_hdf5, update_hdf5

fs = 2500 # EEG sampling frequency
epoch_length = 2 #in seconds

path_to_pt5 = 'C:/Users/timmi/Documents/Rat2'   # Input folder path
hdf5_path = 'C:/Users/timmi/Documents/rath5/Rat2.h5'   #Name of the new hdf5 file to create

Database = h5py.File(hdf5_path, 'w')  # Output directory path

files = np.ravel(os.listdir(path_to_pt5))
# Create recording triplets (HPC, PFC, states)
files = files[:len(files) // 3 * 3].reshape(-1, 3)

num_processes = mp.cpu_count()
print('Number of processes :', num_processes)

results = (
    Parallel(
        n_jobs=min(num_processes, len(files)), verbose = 10)
           (
        delayed(
               prepare_for_hdf5)(recording, fs, path_to_pt5, epoch_length) for recording in files))

for result in results:
    print(result[2])
    update_hdf5(result, hdf5_path)