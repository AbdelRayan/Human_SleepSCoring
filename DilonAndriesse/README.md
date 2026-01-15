# Uncovering latent states of human sleep through unsupervised machine learning.
Based on features extracted from EEG data from two seperate datasets. One of which a publicly available dataset: Sleep-EDF Database Expanded [1].

## Description
This project contains all scripts neccesary to recreate the project to uncover 
latent states in human sleep. From pre-processing, feature computation and 
dataset extraction to machine learning and latent state inference.

## Installation
### 1. Clone the repository
The repository url can be found in the main GIT page, under the "Code" tab.

    git -clone <repository_url>
    
### 2. Install dependencies
The following command can be run in to recreate the conda environment required to run all scripts.

    conda env create -f environment_minimal.yml

## Usage
Base explanation of the script needed to get from .EDF files to inferred latent states. A more in depth view of the different scripts can be found in their corresponding directories.

Scripts that need to manually be run:

    *pre_processing/egi_to_edf.ipynb
    pre_processing/extract_data_edf.ipynb
    hdf5_files/create_hdf5.ipynb
    hdf5_files/dataset_extraction.ipynb
    mcRBM/sample_data/mcRBM_input_features.ipynb
    mcRBM/run_model/train_model.ipynb
    mcRBM/run_model/infer_states_test.ipynb
    mcRBM/run_model/latent_states_analysis.ipynb

_* -This file is only required if the dataset is in EGI format_

### 0. General
Most Jupyter Notebook scripts (except for visualize_data.ipynb) contain cells near the top in the following formats:

#### Paths

    # description of directory
    specific_path_name = r"custom/path/to/files"

#### Variables

    # Description of variable
    variable_name = variable
    # Description of another variable
    another_variable_name = variable

These cells are where you can manually adjust the paths to the required files and set your own custom variables for each of the scripts.

### 1. Pre-processing
If you have EEG data available in EDF format skip "egi_to_edf.ipynb". Otherwise see pre-processing README for more information.

From the "pre_processing" directory run, "extract_data_edf.ipynb" to extract all required data from .EDF files into .mat files that are required for feature computation.

### 2. HDF5 creation
From the "hdf5_files" directory, run the "create_hdf5.ipynb" script to create a hdf5 file from your collection of .mat files. 

To create data that can be used for machine learning, run the "dataset_extraction.ipynb" script. This creates both a training and testing dataset within their specific folders as such: 

    training/
        |
        ├──features.npy 
        ├──states.mat

    testing/
        |
        ├──features.npy 
        ├──states.mat

Make sure these files end up under "mcRBM/sample_data/input/{category_name}/".

### 3. mcRBM
Config files for the mcRBM model can be found under "mcRBM/configuration_files/".
Within exp_details and exp_details, adjust the paths to your current setup and give the model a name. Make sure to use the training data for exp_details and the testing data for exp_details_test.

First run the "mcRBM_input_features.ipynb" script from the "mcRBM/sample_data/" directory to create a features.npz file, which is required for the training of the model.

Now run the "train_model.ipynb" script from the "mcRBM/run_model/" directory to start training the model. Depending on the amount of datapoints in your dataset training could take a while.

After training you can run the following two scripts from "mcRBM/run_model/" successively: "infer_states_test.ipynb" followed by "latent_states_analysis.ipynb". The results from these scripts can be found under "mcRBM/sample_data/experiments/{model_name}/analysis/".

## References
Link to Sleep-EDF Database Expanded dataset: https://physionet.org/content/sleep-edfx/1.0.0/

1. Kemp, B., & Roessen, M. (2018). Sleep-EDF Expanded (Version 1.0.0) [Data set]. PhysioNet. https://physionet.org/content/sleep-edfx/1.0.0/
