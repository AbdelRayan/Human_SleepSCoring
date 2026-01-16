# Uncovering latent states of human sleep through unsupervised machine learning.
Internship project.

## Description
The dataset used during this research is based on features extracted from EEG data of two unrelated datasets. One of which a publicly available dataset called "Sleep-EDF Database Expanded" [1]. See the references for a link to the site where this dataset can be downloaded.

This GIT project contains the scripts necessary to go from raw EGI or EDF data files to create a dataset with 14 features that can be used to train the mcRBM machine learning model and infer latent states. At the current moment in time it is unlikely the model is able to uncover latent states that properly represent human sleep. Thus need for further revision is required.

## Installation
### 1. Clone the repository
You can either manually download the GIT zip file or clone the GIT page using the following command to gain access to the code.

    git clone <repository_url>

The repository URL can be found in the main GIT page called "Human_SleepSCoring", under the "Code" tab.
    
### 2. Install dependencies
The following commands can be run to recreate the conda environment required to run all scripts.

    cd Human_SleepSCoring/DilonAndriesse
    conda env create -f env.yml

## Usage
General workflow required to pre-process the data, create the dataset, train the model and infer latent states.

The follow Notebooks need to be run for this process:

    *pre_processing/egi_to_edf.ipynb
    pre_processing/extract_data_edf.ipynb
    hdf5_files/create_hdf5.ipynb
    hdf5_files/dataset_extraction.ipynb
    mcRBM/sample_data/mcRBM_input_features.ipynb
    mcRBM/run_model/train_model.ipynb
    mcRBM/run_model/infer_states_test.ipynb
    mcRBM/run_model/latent_states_analysis.ipynb

_* -This file is only required if the dataset is in EGI format_

### 0. Additional information
Most Jupyter Notebook scripts (except for visualize_data.ipynb) contain cells near the top of the code in the following formats:

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
_**We assume you have data available in the EDF format, if this is not the case go to the pre_processing README for more information.**_

#### 2.1 _extract_data_edf.ipynb_

Default parameters:
- **pattern** <- regex pattern specific for current datasets
- **wake_time** <- time (s) to save of wake data before and after sleep
- **channels** <- list of channels to be extracted from the data EDF dataset
- **sleep_edf_stage_id** <- dictionary of EDf sleep stages and their corresponding ids
- **motorwp4_stage_id** <- dictionary of extra dataset sleep stages and their corresponding ids
- **edf_sf** <- sampling frequency for EDF dataset
- **motor_sf** <- sampling frequency for motorwp dataset
- **emg_bandpass** <- high and low pass for EMG
- **eog_bandpass** <- high and low pass for EOG

Paths:
- **main_data_path** <- Path to where all EDF files are saved
- **sleep_edf_anno** <- Path to where all EDF annotation files are saved
- **motorwp4_anno** <- Path to where all MOTORWP4 annotation files are saved
- **output_path** <- Path to save the .mat files to

Requirements:
- **All EDF files are in a single directory**
- **All EDF annotation files are in a seperate single directory**
- **Output folder for the mat files is manually made**
- **Paths to both these directories have to be manually adjusted in the Notebook**

From the _"pre_processing"_ directory run _"extract_data_edf.ipynb"_ to extract the necessary channel data from the .EDF files into .mat files. After adjusting the paths you can press run all to run the script.

These mat files will be used to compute the features and create a HDF5 containing the dataset.

### 2. dataset creation
#### 2.1 _hdf5_creation.ipynb_
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

#### 2.2 _dataset_extraction.ipynb_
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

#### 2.3 _mcRBM_input_features.ipynb_
This script is run to create the final version of the dataset in a specific format required to train the mcRBM model. 

Default parameters:
- **feature_npy_file_name** <- name of the feature file created in the previous step
- **feature_npz_file_name** <- name of the feature file that will be created in this script
- **states_file_name** <- name of the states file created in the previous step

Paths:
- **data_path** <- Path to the training or testing dataset created in the previous step (e.g. _"mcRBM/sample_data/input/training/"_)

Requirements:
- **Followed previous step to have data in proper structure**

From the _"mcRBM/sample_data"_ directory, run the _"mcRBM_input_features.ipynb"_ script to create the data required for mcRBM machine learning. After adjusting the paths you can press run all to run the script. This needs to be done for both the training and testing subdirectory.

### 3. mcRBM
#### 3.1 _Configuration files_
Config files for the mcRBM model can be found under _"mcRBM/configuration_files/"_.

Files:
- **input_configuration.txt** <- File with input parameters for the mcRBM model
- **exp_details.txt** <- File with parameters and paths required for training the model
- **exp_details_test.txt** <- File with parameters and paths required for infering latent states

Within _"exp_details"_ adjust the _"dsetDir"_ path to your input training subdirecty, adjust the _"expsDir"_ path to the _"mcRBM/sample_data/experiments"_ subdirectory. Finally set _"expID"_ to a model name as wanted.

Change these same settings within the _"exp_details_test"_, but change _"dsetDir"_ to your testing dataset instead of training. Set the _"modelDirName"_ to the model name and _"modelName"_ to the correct weight file.

#### 3.2 _train_model.ipynb_
To Actually train the model the Cupy package is required. For this research Cupy for CUDA 12 was used. This package needs to manually be installed on a PC with an NVIDIA graphics card that supports CUDA.

Change the paths from the top cells (_"sys.path.append({path})"_ and _"os.chdir"_) to your complete path to the _"mcRBM/scripts"_ subdirectory.

After adjusting all config files you can run the _"train_model.
ipynb"_ script from the _"mcRBM/run_model/"_ subdirectory to start training the model. Depending on the amount of datapoints in your dataset training could take a while (e.g. model with 800.000 datapoints took 30+ hours).

#### 3.3 _infer_states_test.ipynb_
After training has finished you must run the _"infer_state_test.ipynb"_ script.

Change the paths from the top cells (_"os.chdir"_) to your complete path to the _"mcRBM/scripts"_ subdirectory.

Paths:
- **root_dir** <- path to the configuration files
- **base_dir** <- path to the model directory
- **model_dir** <- path to the specific weights

Variables:
- **command** <- specific statement to run in subprocess

In general it is important to change all paths to their corresponding ones on your PC.

#### 3.4 _latent_states_analysis.ipynb_
To do some analysis on the inferred latent states the _"latent_states_analysis.ipynb"_ can be run.

Same as before it is important to change the filepaths within the Notebook.

In this file one of the parameters is very important. A variable called model is created as followed: _"model = StatesAnalysis("C:/Users/andri/school/bio-informatics/internship/donders/vsc/Human_SleepSCoring/DilonAndriesse/mcRBM/", expFile, 9999, 1, False, "L2", "ratios", ["Extracranial Human"])"_. The variable 9999 refers to the weights file that is used, ensure that this number is the same as the number in the weights file from the _"exp_details_test"_.

The results from the analysis can be found under _"mcRBM/sample_data/experiments/{model_name}/analysis/"_.

## References
Link to Sleep-EDF Database Expanded dataset: https://physionet.org/content/sleep-edfx/1.0.0/

1. Kemp, B., & Roessen, M. (2018). Sleep-EDF Expanded (Version 1.0.0) [Data set]. PhysioNet. https://physionet.org/content/sleep-edfx/1.0.0/
