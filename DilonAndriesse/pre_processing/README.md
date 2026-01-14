# Pre-processing
## Convert EGI to EDF
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
        
## Extract data
Most of the scripts available form previous research working using .mat file types. This step applies some preprocessing steps such as a bandpass and creates required channels and converts that data in .edf file types to work with the following steps. To do this use run all in the "extract_data_edf.ipynb" Jupyter. This adds all .mat files from all subject into a singular directory. There are 5 files per subject, the Fpz-Cz, Pz-Oz, EMG and EOG channels, and the sleep states. 

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
