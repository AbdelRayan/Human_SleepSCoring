# HDF5 files
This directory is used to create a HDF5 file from the .mat files created in the pre-processing directory. Followed by extracting the training and testing set from this created HDF5 file.

## _hdf5_creation.ipynb_
After having created the .mat files, the next step is to create the HDF5 file with all feature and annotation data.

Default parameters:
- **epoch_length** <- length of each datapoint
- **pattern** <- regex pattern specific for current datasets

Paths:
- **path_to_mat** <- path to where .mat files are saved
- **hdf5_path** <- path to save HDF5 file to

Requirements:
- **Complete .mat file collections. So all 5 files (Fpz-Cz, Pz-Oz, EMG, EOG and States) for each subject. Missing files may cause errors**

From the _"hdf5_files"_ directory, run the _"create_hdf5.ipynb"_ script to create a hdf5 file from your collection of .mat files. After adjusting the paths you can press run all to run the script. 

## _dataset_extraction.ipynb_
From this HDF5 file a training and testing dataset will be extracted to be used for machine learning. This will create a upsampled dataset by resampling.

Default parameters:
- **dataset_name** <- name of dataset to save the training and testing data to
- **states** <- states of interest (Wake, N1, N2, N3 and REM)

Paths:
- **file_path** <- path to HDF5 file
- **partial_output_path** <- path to input folder of mcRBM subdirectory ( will be combined with dataset name for own subset)

Requirements:
- **Have everything from the mcRBM subdirectory downloaded. These datasets are saved to the _"mcRBM/sample_data/input/"_ directory**

From the _"hdf5_files"_ directory, run the _"dataset_extraction.ipynb"_ script to create a training and testing dataset from your HDF5 file. After adjusting the paths and variables you can press run all to run the script.  This creates both a training and testing dataset within their specific folders structures as such: 

    {dataset_name}/
            |
            ├──training/
            |    |
            |    ├──features.npy 
            |    ├──states.mat
            |
            ├──testing/
                |
                ├──features.npy 
                ├──states.mat

**Make sure these files end up under _"mcRBM/sample_data/input/{category_name}/"_.**