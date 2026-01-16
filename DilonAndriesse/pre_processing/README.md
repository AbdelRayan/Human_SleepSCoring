# Pre-processing
## _"egi_to_edf.ipynb"_
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

Paths:
- **raw_egi_files** <- main directory where all subdirectories with .RAW files are saved
- **output_path** <- Path to save the EDF files to

Default parameters:
- **channel_selection** <- a selection of all channels to extract from the .RAW EGI file
- **fpz** <- specific channel for this electrode position
- **cz** <- collection of channels to take average of for this electrode position
- **pz** <-specific channel for this electrode position
- **oz** <-specific channel for this electrode position
- **channel_types** <- dictionary with channel types for each channel

An example of a file name would be like this: "CURRENTSTUDY_S35_1 2 20201214 2207.RAW"
From this file the subject is extract with the following naming convention: "subject_night" i.e. "S35_1".
        
## _extract_data_edf.ipynb_

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