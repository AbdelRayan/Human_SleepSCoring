# HDF5 files
This directory is used to create a HDF5 file from the .mat files created in the pre-processing directory. Followed by extracting the training and testing set from this created HDF5 file.

## Create HDF5
The required HDF5 file can be created by running the "create_hdf5.ipynb" script. Within this script in the third cell you can manually adjust the parameters, such as: epoch length, and the paths to retrieve the .mat files and where to save the hdf5 file to.

Built into the script is to match sampling frequency to the specific dataset. For this project the Sleep-EDF dataset had a sampling frequency of 100, while the other dataset had a sampling frequency of 250.

## Extract datasets
After creating the HDF5 file, to extract the training and testing set, you must run the "dataset_extraction.ipynb" script. Within this script you can adjust the main variables and paths. Output is required to end up under "mcRBM/sample_data/input/". Dataset_name can be changed according to what you want to call your current dataset. This script was built with the 5 states of human sleep in mind. Therefore it won't work on data that do not have these 5 states.