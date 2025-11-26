# Uncovering latent states of human sleep through unsupervised machine learning.
Based on features extracted from EEG data from two seperate datasets. One of which a publicly available dataset: Sleep-EDF Database Expanded [1].

## Description
This project contains all scripts neccesary to recreate the project to uncover 
latent states in human sleep. From pre-processing, feature computation and 
dataset extraction to machine learning and latent state inference.

## Installation
### 1. Clone the repository
    (Add code required for cloning here)
### 2. Install dependencies
    (create conda environment file and add the instruction to create new conda environment here)

## Usage
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
If you have EEG data available in EDF format skip step "0. Convert EGI to EDF"
#### 0. Convert EGI to EDF
To use most of the pre-processing scripts, it is required that your EEG data is available in EDF format.
If your data is a .RAW EGI file you can use the egi_to_edf.ipynb Jupyter Notebook to convert.
This script only works when the EGI files are available in a specific format like this:

    main_directory (EGI)/
        |
        ├── Subject1/
        |    |
        |    ├──"datasetname_subject1_night1 dateofrecording.RAW"
        |    ├──"datasetname_subject1_night2 dateofrecording.RAW"
        | 
        ├── Subject2/
        |    |
        |    ├──"datasetname_subject1_night1 dateofrecording.RAW"
        |    ├──"datasetname_subject1_night2 dateofrecording.RAW"
        |
        ├── Etc...

An example of a file name would be something like this: "CURRENTSTUDY_S35_1 2 20201214 2207.RAW"
From this file the subject is extract with the following naming convention: "subject_night" i.e. "S35_1".
        
#### 1. Extract data
Most of the scripts available form previous research working using .mat file types. This step applies some preprocessing steps such as a bandpass and creates required channels and converts that data in .mat file types to work with the following steps. To do this use run all in the "extract_data_edf.ipynb" Jupyter. This adds all .mat files from all subject into a singular directory. There are 5 files per subject, the Fpz-Cz, Pz-Oz, EMG and EOG channels, and the sleep states. 

Layout looks like this:

    mat_files/
        |
        ├──S35_1_Fpz-Cz.mat
        ├──S35_1_Pz-Oz.mat
        ├──S35_1_EMG.mat
        ├──S35_1_EOG.mat
        ├──S35_1_states.mat
        ├──S35_2_Fpz-Cz.mat
        ├──S35_2_Pz-Oz.mat
        ├──S35_2_EMG.mat
        ├──S35_2_EOG.mat
        ├──S35_2_states.mat
        ├──Etc...

### 2 Data visualization
This directory is used to visualize the input data and features, to manually check wether these feature properly represent the different sleep states.
#### 1. Input data
This Notebook visualizes the input data. This script has a config file in which you can manually set variables to be used for the visualization. This file is called "visualization_config.yaml".
#### 2. Rodent features
This notebook visualizes the main features used in the rodent research. These images are saved to a custom path you have to set. There these images are saved per subject. The structure may look something like this:

    feature_visualization/
        |
        ├──Category_name (i.e. "10s-epochs")
            ├──S35_1
            |    ├──normalized_emg.svg
            |    ├──wei_all-indices.svg
            |    ├──wei-indices_vs_new-indices.svg
            |    ├──Etc...
            ├──S35_2
            |    ├──normalized_emg.svg
            |    ├──wei_all-indices.svg
            |    ├──wei-indices_vs_new-indices.svg
            |    ├──Etc...
            ├──Etc...
#### 3. New features
#### 4. Recreation of paper on fractal cycle analysis
### 3. HDF5 creation
### 4. mcRBM
#### 1. Configuration
#### 2. Training
#### 3. Analysis

